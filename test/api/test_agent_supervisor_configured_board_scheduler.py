"""Tests for the sealed scheduler-config runtime adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    execution_plan as execution_plan_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    local_profile as local_profile_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.execution_plan import (
    ExecutionClaimConflictError,
    ExecutionPlanError,
    ProductionParallelPlanAdapter,
    load_plan_revision_store_binding,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
    export_local_profile_lifecycle_witness,
    initialize_local_profile,
    lifecycle_root_identity_did,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    ConfiguredBoardError,
    configured_board_launch_plan,
    load_configured_board,
    materialize_configured_board_execution_plan,
    preflight_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PlanRevisionStore,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KITA_CONFIG = (
    REPO_ROOT
    / "config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json"
)
V3_BOARD_NAMESPACE = "agent-supervisor-prompt-only-self-improvement-v3"
V3_ROUTE_ID = (
    "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
)
V3_AUTHORIZATION_PATH = Path(
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "provider_fallback_policy_authorization_20260808.json"
)
V3_ROOT_PIN_PATH = Path(
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "local_profile_lifecycle_root_pin_20260808.json"
)
V3_WITNESS_PATH = Path(
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "local_profile_lifecycle_witness_20260808.json"
)
_TEST_SEALED_DESCRIPTORS: list[int] = []


@pytest.fixture(autouse=True)
def _isolated_local_profile_lifecycle_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setattr(
        local_profile_module,
        "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE",
        tmp_path / "local-profile-root-registry",
    )
    yield
    for descriptor in _TEST_SEALED_DESCRIPTORS:
        try:
            os.close(descriptor)
        except OSError:
            pass
    _TEST_SEALED_DESCRIPTORS.clear()


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_addressed_mapping(
    value: dict[str, Any],
    *,
    identity_field: str,
) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _test_sealed_control_plane(
    tmp_path: Path,
    *,
    source_head: str,
    source_tree: str,
) -> tuple[
    llm_router.AgentImplementationControlPlanePin,
    llm_router.AgentImplementationSealedControlPlane,
]:
    """Build a production-shaped capsule from this test process's sources."""

    root = tmp_path / "accepted-control-plane-capsule"
    root.mkdir(mode=0o700)
    relative_files = set(llm_router._AGENT_CONTROL_PLANE_RELATIVE_FILES)
    relative_files.update(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (
            REPO_ROOT / "ipfs_accelerate_py/agent_supervisor"
        ).rglob("*.py")
    )
    digests: dict[str, str] = {}
    for relative in sorted(relative_files):
        source = REPO_ROOT / relative
        payload = source.read_bytes()
        target = root / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o400)
        digests[relative] = "sha256:" + hashlib.sha256(payload).hexdigest()
    manifest: dict[str, Any] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "materialized-control-plane@1"
        ),
        "source_head": source_head,
        "source_tree": source_tree,
        "files": digests,
    }
    manifest["capsule_id"] = _content_addressed_mapping(
        manifest,
        identity_field="capsule_id",
    )
    manifest_path = root / ".agent-control-plane-manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest) + b"\n")
    manifest_path.chmod(0o400)
    for directory in sorted(
        (entry for entry in root.rglob("*") if entry.is_dir()),
        key=lambda entry: len(entry.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    root.chmod(0o500)
    pin = llm_router.build_agent_implementation_control_plane_pin(
        runner_path=(
            root
            / "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py"
        ),
        capsule_root=root,
    )
    sealed = llm_router.seal_agent_implementation_control_plane_capsule(pin)
    _TEST_SEALED_DESCRIPTORS.append(sealed.descriptor)
    return pin, sealed


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def _configure_git(repo: Path) -> None:
    _git(repo, "config", "user.name", "Configured Board Test")
    _git(repo, "config", "user.email", "configured-board@example.invalid")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_configured_repo(tmp_path: Path) -> tuple[Path, Path]:
    child = tmp_path / "dependency-source"
    child.mkdir()
    _git(child, "init", "-b", "main")
    _configure_git(child)
    _write(child / "dependency.txt", "dependency\n")
    _git(child, "add", "dependency.txt")
    _git(child, "commit", "-m", "seed dependency")
    child_revision = _git(child, "rev-parse", "HEAD").stdout.strip()

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _configure_git(repo)
    _write(repo / "README.md", "configured board fixture\n")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "dependency",
    )
    _git(repo, "add", "README.md", ".gitmodules", "dependency")
    _git(repo, "commit", "-m", "seed repository")
    ancestor = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _write(repo / "docs/plan.md", "plan\n")
    _write(repo / "docs/objectives.md", "# Objectives\n")
    _write(repo / "docs/tasks.md", "# Tasks\n")
    _write(
        repo / "scripts/validate_board.py",
        (
            "import json\n"
            "print(json.dumps({'valid': True, 'errors': []}, sort_keys=True))\n"
        ),
    )
    _write(
        repo
        / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
        "raise SystemExit(0)\n",
    )
    config_path = repo / "config/scheduler.json"
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "configured_board_test.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "plan_path": "docs/plan.md",
        "validator_path": "scripts/validate_board.py",
        "task_prefix": "TEST-",
        "goal_prefix": "TEST-G",
        "board_namespace": "configured-board-test",
        "merge_target_branch": "main",
        "source_binding": {
            "accelerator_required_ancestor": ancestor,
            "accelerator_required_branch": "main",
            "dependency_submodule_path": "dependency",
            "dependency_planning_revision": child_revision,
        },
        "max_lanes": 2,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "poll_interval_seconds": 5,
        "daemon_interval_seconds": 60,
        "check_interval_seconds": 30,
        "stale_seconds": 1800,
        "watchdog_startup_grace_seconds": 300,
        "max_restarts": 3,
        "max_task_attempts": 3,
        "implementation_retry_budget": 3,
        "validation_retry_budget": 3,
        "merge_retry_budget": 3,
        "implementation_timeout_seconds": 7200,
        "implementation_max_timeout_seconds": 21600,
        "implementation_log_stall_seconds": 1200,
        "worktree_submodule_paths": ["dependency"],
        "protected_paths": [
            "config/scheduler.json",
            "docs/plan.md",
            "docs/objectives.md",
            "docs/tasks.md",
            "scripts/validate_board.py",
        ],
        "runtime_paths": {
            "root": "data/configured-board",
            "state": "data/configured-board/state",
            "worktrees": "data/configured-board/worktrees",
            "merge_queue": "data/configured-board/merge-queue",
            "logs": "data/configured-board/logs",
        },
        "lanes": [
            {
                "index": 0,
                "name": "test-lane-0",
                "strict_shard_remainder": 0,
            },
            {
                "index": 1,
                "name": "test-lane-1",
                "strict_shard_remainder": 1,
            },
        ],
        "provider": {
            "provider_id": "codex",
            "model_id": "test-model",
            "max_concurrency": 2,
        },
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(
        repo,
        "add",
        "config/scheduler.json",
        "docs/plan.md",
        "docs/objectives.md",
        "docs/tasks.md",
        "scripts/validate_board.py",
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
    )
    _git(repo, "commit", "-m", "add configured board")
    return repo, config_path


def _commit_v3_route_authorization(
    repo: Path,
    config_path: Path,
    payload: dict[str, object],
) -> None:
    source_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    source_tree = _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_identity = ed25519_did_key(reviewer_key.public_key())
    profile_dir = repo.parent / f"{repo.name}-reviewer-profile"
    lifecycle_dir = repo.parent / f"{repo.name}-reviewer-lifecycle"
    profile = initialize_local_profile(
        repository_cid=f"repository:{repo.name}",
        baseline_commit=source_head,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        signing_key=reviewer_key.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        ),
        effect_bounds=("edit", "isolated_worktree", "test"),
        budget_cid="budget:configured-board-fixture",
        resource_cid="resource:configured-board-fixture",
        route_id=V3_ROUTE_ID,
        reviewer_identity=reviewer_identity,
        reviewer_provider="local_operator",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_reasoning_effort="high",
    )
    root_identity_did = lifecycle_root_identity_did()
    authorized_at_ms = int(time.time()) * 1000
    root_pin: dict[str, Any] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "local-profile-lifecycle-root-pin@1"
        ),
        "board_namespace": V3_BOARD_NAMESPACE,
        "base_head": source_head,
        "base_tree": source_tree,
        "root_identity_did": root_identity_did,
        "pinned_at_ms": authorized_at_ms,
    }
    root_pin["pin_id"] = _content_addressed_mapping(
        root_pin,
        identity_field="pin_id",
    )
    root_pin_path = repo / V3_ROOT_PIN_PATH
    root_pin_path.parent.mkdir(parents=True, exist_ok=True)
    root_pin_path.write_bytes(_canonical_json_bytes(root_pin))
    root_pin_path.chmod(0o400)
    _git(repo, "add", V3_ROOT_PIN_PATH.as_posix())
    _git(repo, "commit", "-m", "pin fixture lifecycle root")

    witness_nonce = "witness:" + hashlib.sha256(
        str(repo).encode("utf-8")
    ).hexdigest()
    witness = export_local_profile_lifecycle_witness(
        repository_cid=f"repository:{repo.name}",
        board_namespace=V3_BOARD_NAMESPACE,
        base_head=source_head,
        base_tree=source_tree,
        nonce=witness_nonce,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        observed_at_ms=authorized_at_ms,
        expires_at_ms=authorized_at_ms + 10 * 60 * 1000,
    )
    witness_path = repo / V3_WITNESS_PATH
    witness_path.write_bytes(_canonical_json_bytes(witness))
    witness_path.chmod(0o400)
    witness_sha256 = "sha256:" + hashlib.sha256(
        witness_path.read_bytes()
    ).hexdigest()
    root_pin_sha256 = "sha256:" + hashlib.sha256(
        root_pin_path.read_bytes()
    ).hexdigest()
    authority_bounds: dict[str, Any] = {
        "repository_cid": f"repository:{repo.name}",
        "baseline_commit": source_head,
        "effects": ["edit", "isolated_worktree", "test"],
        "budget_cid": "budget:configured-board-fixture",
        "resource_cid": "resource:configured-board-fixture",
        "authority_cid": profile.content_id,
    }
    route = {
        "route_id": V3_ROUTE_ID,
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
        "allowed_trigger_classes": [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ],
    }
    review_payload = llm_router.agent_implementation_route_review_payload(
        board_namespace=V3_BOARD_NAMESPACE,
        authorization_kind="explicit_operator_override",
        source_head=source_head,
        source_tree=source_tree,
        route=route,
        authority_bounds=authority_bounds,
        reviewer_identity=reviewer_identity,
        reviewer_provider="local_operator",
        reviewer_profile_id=profile.profile_id,
        reviewer_profile_content_id=profile.content_id,
        reviewer_lifecycle_anchor_id=profile.lifecycle_anchor_id,
        reviewer_lifecycle_generation=profile.lifecycle_generation,
        reviewer_witness_path=V3_WITNESS_PATH.as_posix(),
        reviewer_witness_sha256=witness_sha256,
        lifecycle_root_identity_did=root_identity_did,
        lifecycle_witness_nonce=witness_nonce,
        lifecycle_root_pin_path=V3_ROOT_PIN_PATH.as_posix(),
        lifecycle_root_pin_sha256=root_pin_sha256,
        authorized_at_ms=authorized_at_ms,
        fallback_implementer_identity="codex",
    )
    signature = base64.b64encode(
        reviewer_key.sign(_canonical_json_bytes(review_payload))
    ).decode("ascii")
    artifact = repo / V3_AUTHORIZATION_PATH
    authorization = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-policy-authorization@2"
        ),
        "board_namespace": V3_BOARD_NAMESPACE,
        "authorization_source": {
            "kind": "explicit_operator_override",
            "source_head": source_head,
            "source_tree": source_tree,
            "prospective_only": True,
            "requires_descendant_tree": True,
        },
        "route": route,
        "ownership_contract": {
            "canonical_route_plan_owner": "ipfs_accelerate_py.llm_router",
            "typed_fallback_decision_owner": "ipfs_accelerate_py.llm_router",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed": False,
        },
        "bootstrap_route_guarantees": {
            "explicit_codex_review_conflict_denied": True,
        },
        "reviewer": {
            "identity": reviewer_identity,
            "provider": "local_operator",
            "profile_id": profile.profile_id,
            "profile_content_id": profile.content_id,
            "lifecycle_anchor_id": profile.lifecycle_anchor_id,
            "generation": profile.lifecycle_generation,
            "witness_path": V3_WITNESS_PATH.as_posix(),
            "witness_sha256": witness_sha256,
            "signature": signature,
        },
        "authority_bounds": authority_bounds,
        "fallback_implementer_identity": "codex",
        "lifecycle_root_identity_did": root_identity_did,
        "lifecycle_witness_nonce": witness_nonce,
        "lifecycle_root_pin_path": V3_ROOT_PIN_PATH.as_posix(),
        "lifecycle_root_pin_sha256": root_pin_sha256,
        "authorized_at_ms": authorized_at_ms,
    }
    payload["board_namespace"] = V3_BOARD_NAMESPACE
    provider = payload["provider"]
    assert isinstance(provider, dict)
    provider["route_authorization_path"] = V3_AUTHORIZATION_PATH.as_posix()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(_canonical_json_bytes(authorization))
    artifact.chmod(0o400)
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(
        repo,
        "add",
        config_path.relative_to(repo).as_posix(),
        V3_AUTHORIZATION_PATH.as_posix(),
        V3_WITNESS_PATH.as_posix(),
    )
    _git(repo, "commit", "-m", "authorize scoped high route")


def _task_block(
    task_id: str,
    *,
    status: str = "todo",
    depends_on: tuple[str, ...] = (),
    output: str | None = None,
    schedulable: bool = True,
) -> str:
    return "\n".join(
        (
            f"## {task_id} {task_id} fixture",
            "",
            f"- Status: {status}",
            "- Completion: automatic",
            f"- Is schedulable: {'true' if schedulable else 'false'}",
            "- Priority: P0",
            "- Track: fixture",
            f"- Depends on: {', '.join(depends_on)}",
            f"- Outputs: {output or f'src/{task_id.lower()}.py'}",
            "- Validation: python -m pytest -q",
            "- Resource class: cpu-small",
            "",
        )
    )


def _seed_v3_task_repo(
    tmp_path: Path,
    blocks: tuple[str, ...],
) -> tuple[Path, Path, scheduler_module.ConfiguredBoard]:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_or_auth_unavailable",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
    }
    _write(repo / "docs/tasks.md", "# Tasks\n\n" + "\n".join(blocks))
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "docs/tasks.md", "config/scheduler.json")
    _git(repo, "commit", "-m", "seed v3 task population")
    _commit_v3_route_authorization(repo, config_path, payload)
    board = load_configured_board(config_path, repo_root=repo)
    return repo, config_path, board


PLAN_NOW = 1_000_000


def _host_capacity(*, lanes: int = 2) -> dict[str, object]:
    return {
        "observed_at_ms": PLAN_NOW,
        "worker_limit": lanes,
        "available_worker_capacity": lanes,
        "active_workers": 0,
        "cpu_percent": 1,
        "memory_percent": 1,
        "disk_percent": 1,
        "memory_total_bytes": 1_000_000_000,
        "memory_available_bytes": 900_000_000,
        "disk_total_bytes": 1_000_000_000,
        "disk_available_bytes": 900_000_000,
        "capabilities": ["cpu"],
        "resource_classes": ["cpu-small", "coordinator"],
    }


def _provider_capacity(
    *,
    lanes: int = 2,
    active: int = 0,
    observed_at_ms: int = PLAN_NOW,
    primary_healthy: bool = True,
    fallback_healthy: bool = True,
) -> tuple[dict[str, object], ...]:
    def observation(provider_id: str, healthy: bool) -> dict[str, object]:
        return {
            "provider_id": provider_id,
            "healthy": healthy,
            "quota_remaining": 10,
            "latency_ms": 25,
            "context_window_tokens": 100_000,
            "token_budget_remaining": 100_000,
            "max_concurrency": lanes,
            "active_requests": active,
            "capabilities": ["implementation"],
            "observed_at_ms": observed_at_ms,
            "retry_after_ms": 0,
            "available_concurrency": max(0, lanes - active),
        }

    return (
        observation("grok_cli", primary_healthy),
        observation("codex_cli", fallback_healthy),
    )


def _common_args(plan: dict[str, object]) -> list[str]:
    prefix = "--common-arg="
    return [
        item[len(prefix) :]
        for item in plan["argv"]
        if isinstance(item, str) and item.startswith(prefix)
    ]


def _fenced_plan_children(
    tmp_path: Path,
) -> tuple[
    Path,
    scheduler_module.ConfiguredBoard,
    object,
    multi_runner_module.PlanBoundSupervisorChild,
    multi_runner_module.PlanBoundSupervisorChild,
    subprocess.Popen[bytes],
]:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"), _task_block("TEST-B")),
    )
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(),
        provider_capacity_snapshots=_provider_capacity(),
        task_state_snapshots=(),
    )
    assert receipt is not None
    launch = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        stamp="20260808T040404Z",
        parallelism_receipt=receipt,
    )
    argv = list(launch["argv"])
    records = [
        argv[index + 1]
        for index, token in enumerate(argv[:-1])
        if token == "--implementation-plan-bound-track"
    ]
    children = tuple(
        multi_runner_module.PlanBoundSupervisorChild.from_cli_record(record)
        for record in records
    )
    assert len(children) == 2
    donor, recipient = children
    track = donor.track(stamp="20260808T040404Z").resolve(repo)
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        *track.extra_args,
    ]
    state_root = track.supervisor_pid_path.parent.resolve()
    profile = multi_runner_module.LifecycleProfile(
        target_id=f"supervisor-track:{donor.name}",
        run_id="test-plan-bound-reassignment",
        configuration_root="test-plan-bound-reassignment-config",
        repository_root=str(repo.resolve()),
        state_root=str(state_root),
        run_root=str(state_root / "lifecycle-runs" / donor.name),
        argv=tuple(command),
        cwd=str(repo.resolve()),
    )
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=profile.launch_environment(0),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    process_identity = multi_runner_module.LinuxProcessAdapter()._identity(
        process.pid,
        profile,
    )
    setattr(process, "_agent_supervisor_lifecycle_profile", profile)
    setattr(process, "_agent_supervisor_process_identity", process_identity)
    birth_cid = multi_runner_module._persist_plan_bound_process_birth(
        profile=profile,
        process_identity=process_identity,
        repo_root=repo,
    )
    setattr(process, "_agent_supervisor_process_birth_cid", birth_cid)
    fenced, _member_pids = multi_runner_module._terminate_managed_process(
        process,
        grace_seconds=1.0,
    )
    assert fenced is True
    process.wait(timeout=5)
    return repo, board, receipt, donor, recipient, process


def test_plan_bound_identity_capture_failure_fences_before_child_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_head = _git(REPO_ROOT, "rev-parse", "HEAD").stdout.strip()
    source_tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}").stdout.strip()
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=source_head,
        source_tree=source_tree,
    )
    runtime_relative = Path("data/agent_supervisor") / (
        "plan-bound-gate-test-" + tmp_path.name
    )
    runtime_root = REPO_ROOT / runtime_relative
    lane_relative = runtime_relative / "lane-0"
    lane_root = REPO_ROOT / lane_relative
    supervisor_pid = lane_root / "supervisor.pid"
    store_relative = runtime_relative / "plan-revision-store"
    plan_args = (
        "--state-dir", str(lane_relative),
        "--plan-bound-dispatch",
        "--plan-revision-store-path", str(store_relative),
        "--plan-bound-revision-cid", "revision:test",
        "--plan-bound-plan-root-cid", "plan-root:test",
        "--plan-bound-execution-plan-cid", "execution-plan:test",
        "--plan-bound-capacity-snapshot-id", "capacity:test",
        "--plan-bound-slice-manifest-cid", "manifest:test",
        "--plan-bound-slice-id", "slice:test",
        "--plan-bound-source-head", source_head,
        "--plan-bound-source-tree", source_tree,
        "--plan-bound-task-source-revision", "task-source:test",
        "--plan-bound-configuration-root", "configuration:test",
        "--plan-bound-accepted-tree-root", str(REPO_ROOT),
        "--plan-bound-lane-id", "lane-0",
        "--execution-slice-task-id", "TEST-A",
        "--execution-slice-task-cid", "task-cid:test-a",
    )
    track = multi_runner_module.SupervisorTrack(
        name="identity-capture-failure",
        script_path=Path(multi_runner_module.PLAN_BOUND_ACCEPTED_ENTRY_PATH),
        log_path=lane_root / "supervisor.log",
        supervisor_pid_path=supervisor_pid,
        daemon_pid_path=lane_root / "daemon.pid",
        supervisor_status_path=lane_root / "supervisor-status.json",
        extra_args=plan_args,
    )
    original_identity = multi_runner_module.LinuxProcessAdapter._identity

    def fail_identity(_self, _pid, _profile):
        raise multi_runner_module.ProcessIdentityMismatch(
            "deterministic process-birth capture failure"
        )

    monkeypatch.setattr(
        multi_runner_module.LinuxProcessAdapter,
        "_identity",
        fail_identity,
    )
    monkeypatch.setattr(
        multi_runner_module,
        "_validate_plan_bound_accepted_tree",
        lambda **_kwargs: None,
    )
    try:
        with pytest.raises(
            multi_runner_module.PlanBoundProcessBirthError,
            match="launch remained gated",
        ) as captured:
            multi_runner_module.start_track(
                track,
                repo_root=REPO_ROOT,
                common_args=(),
                python_executable=sys.executable,
                accepted_control_plane_pin=control_plane_pin,
                accepted_control_plane_descriptor=(
                    control_plane_launch.descriptor
                ),
                output=lambda _message: None,
            )

        failure = captured.value
        assert failure.all_trees_fenced is True
        assert not supervisor_pid.exists()
        assert not Path(f"/proc/{failure.pid}").exists()
        monkeypatch.setattr(
            multi_runner_module.LinuxProcessAdapter,
            "_identity",
            original_identity,
        )
        assert not multi_runner_module.LinuxProcessAdapter().snapshot(
            failure.profile
        ).members
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)


def test_accepted_tree_entries_ignore_hostile_python_import_authority(
    tmp_path: Path,
) -> None:
    shadow_root = tmp_path / "shadow"
    shadow_package = shadow_root / "ipfs_accelerate_py"
    shadow_package.mkdir(parents=True)
    sentinel = tmp_path / "shadow-imported"
    (shadow_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    (shadow_root / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    shadow_runner = (
        shadow_package / "agent_supervisor/runtime/multi_supervisor_runner.py"
    )
    shadow_runner.parent.mkdir(parents=True)
    shadow_runner.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    hostile_environment = dict(os.environ)
    hostile_environment.update(
        {
            "PYTHONPATH": str(shadow_root),
            "PYTHONUSERBASE": str(shadow_root),
            "PYTHONSTARTUP": str(shadow_package / "__init__.py"),
        }
    )
    source_head = _git(REPO_ROOT, "rev-parse", "HEAD").stdout.strip()
    source_tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}").stdout.strip()
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=source_head,
        source_tree=source_tree,
    )
    sealed_modules = (
        multi_runner_module.PLAN_BOUND_LAUNCH_GATE_MODULE,
        (
            "ipfs_accelerate_py.agent_supervisor.runtime."
            "configured_board_scheduler"
        ),
        (
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_supervisor"
        ),
    )
    for sealed_module in sealed_modules:
        sealed_command = (
            multi_runner_module.build_sealed_control_plane_module_command(
                python_executable=sys.executable,
                pin=control_plane_pin,
                descriptor=control_plane_launch.descriptor,
                module_name=sealed_module,
                argv=("--help",),
            )
        )
        sealed_result = subprocess.run(
            sealed_command,
            cwd=shadow_root,
            env=hostile_environment,
            pass_fds=(control_plane_launch.descriptor,),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert sealed_result.returncode == 0, sealed_result.stderr
        assert "usage:" in sealed_result.stdout
        assert not sentinel.exists()
    entries = (
        REPO_ROOT
        / "scripts/ops/agent_supervisor/configured_board_scheduler.py",
        REPO_ROOT
        / multi_runner_module.PLAN_BOUND_GATE_ENTRY_PATH,
    )
    for entry in entries:
        result = subprocess.run(
            [sys.executable, "-I", str(entry), "--help"],
            cwd=shadow_root,
            env=hostile_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        assert not sentinel.exists()


def test_detached_coordinator_pid_projection_rejects_symlink_and_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    state_dir = board.path(board.runtime_paths["state"])
    log_dir = board.path(board.runtime_paths["logs"])
    state_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    pid_path = state_dir / "configured-board-master.pid"
    outside = tmp_path / "outside-pid-target"
    outside.write_text(f"{os.getpid()}\n", encoding="utf-8")
    module_path = (
        repo
        / "ipfs_accelerate_py/agent_supervisor/runtime/"
        "configured_board_scheduler.py"
    )
    monkeypatch.setattr(scheduler_module, "__file__", str(module_path))
    monkeypatch.setattr(
        scheduler_module,
        "_git_identity",
        lambda _root: ("a" * 40, "b" * 40),
    )
    monkeypatch.setattr(
        scheduler_module,
        "_tracked_head_snapshot",
        lambda **_kwargs: (b"tracked", "sha256:tracked"),
    )
    spawned = False

    def forbidden_spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("unsafe PID projection must fence before spawn")

    monkeypatch.setattr(scheduler_module.subprocess, "Popen", forbidden_spawn)

    pid_path.symlink_to(outside)
    with pytest.raises(ConfiguredBoardError, match="symbolic link"):
        scheduler_module._launch_detached_plan_bound_coordinator(
            board,
            implement=True,
            duration_seconds=1.0,
        )
    assert scheduler_module._remove_owned_coordinator_pid(board) is False
    assert outside.read_text(encoding="utf-8") == f"{os.getpid()}\n"
    assert spawned is False
    pid_path.unlink()

    os.link(outside, pid_path)
    try:
        with pytest.raises(ConfiguredBoardError, match="hardlinked"):
            scheduler_module._launch_detached_plan_bound_coordinator(
                board,
                implement=True,
                duration_seconds=1.0,
            )
        assert scheduler_module._remove_owned_coordinator_pid(board) is False
        assert outside.read_text(encoding="utf-8") == f"{os.getpid()}\n"
        assert spawned is False
    finally:
        pid_path.unlink()

    descriptor, identity = scheduler_module._reserve_coordinator_pid_projection(
        pid_path
    )
    try:
        scheduler_module._publish_reserved_coordinator_pid(
            pid_path,
            descriptor,
            identity,
            os.getpid(),
        )
    finally:
        os.close(descriptor)
    os.chmod(pid_path, 0o644)
    assert scheduler_module._remove_owned_coordinator_pid(board) is False
    os.chmod(pid_path, 0o600)
    assert scheduler_module._remove_owned_coordinator_pid(board) is True
    assert not pid_path.exists()


def test_plan_bound_wave_and_supervisor_pid_projections_reject_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(lanes=1),
        provider_capacity_snapshots=_provider_capacity(lanes=1),
        task_state_snapshots=(),
    )
    assert receipt is not None
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=receipt.slice_manifest.source_head,
        source_tree=receipt.slice_manifest.repository_tree_id,
    )
    launch_plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        stamp="20260809T-pid-projection",
        parallelism_receipt=receipt,
        accepted_control_plane_pin=control_plane_pin,
        accepted_control_plane_descriptor=control_plane_launch.descriptor,
    )
    argv = list(launch_plan["argv"])
    record_index = argv.index("--implementation-plan-bound-track")
    child = multi_runner_module.PlanBoundSupervisorChild.from_cli_record(
        argv[record_index + 1]
    )
    track = child.track(stamp="20260809T-pid-projection")
    resolved_track = track.resolve(repo)
    resolved_track.supervisor_pid_path.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-plan-bound-pid"
    outside.write_text("31337\n", encoding="ascii")
    spawned = False

    def forbidden_spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("unsafe PID authority reached process birth")

    monkeypatch.setattr(
        multi_runner_module,
        "_validate_plan_bound_accepted_tree",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        multi_runner_module.subprocess,
        "Popen",
        forbidden_spawn,
    )

    for link_kind in ("symbolic", "hard"):
        if link_kind == "symbolic":
            resolved_track.supervisor_pid_path.symlink_to(outside)
        else:
            os.link(outside, resolved_track.supervisor_pid_path)
        try:
            expected = "symbolic link" if link_kind == "symbolic" else "hardlink"
            with pytest.raises(ValueError, match=expected):
                multi_runner_module.start_track(
                    track,
                    repo_root=repo,
                    common_args=(),
                    python_executable=sys.executable,
                    accepted_control_plane_pin=control_plane_pin,
                    accepted_control_plane_descriptor=(
                        control_plane_launch.descriptor
                    ),
                    output=lambda _message: None,
                )
        finally:
            resolved_track.supervisor_pid_path.unlink()
        assert outside.read_text(encoding="ascii") == "31337\n"
        assert spawned is False

    master_pid = board.path(board.runtime_paths["state"]) / "wave-master.pid"
    for link_kind in ("symbolic", "hard"):
        if link_kind == "symbolic":
            master_pid.symlink_to(outside)
        else:
            os.link(outside, master_pid)
        try:
            expected = "symbolic link" if link_kind == "symbolic" else "hardlink"
            with pytest.raises(ValueError, match=expected):
                multi_runner_module.run_supervisor_tracks(
                    (track,),
                    repo_root=repo,
                    common_args=(),
                    duration_seconds=0.1,
                    master_pid_path=master_pid,
                    plan_bound_children=(child,),
                    accepted_control_plane_pin=control_plane_pin,
                    accepted_control_plane_descriptor=(
                        control_plane_launch.descriptor
                    ),
                    output=lambda _message: None,
                )
        finally:
            master_pid.unlink()
        assert outside.read_text(encoding="ascii") == "31337\n"
        assert spawned is False


def test_kita_config_maps_to_four_strict_existing_supervisor_lanes() -> None:
    board = load_configured_board(KITA_CONFIG, repo_root=REPO_ROOT)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260801T000000Z",
    )
    args = plan["argv"]
    common = _common_args(plan)

    lane_flag = args.index("--implementation-supervisor-lanes-per-track")
    assert args[lane_flag + 1] == "4"
    assert "--implementation-supervisor-strict-task-sharding" in args
    assert "--exit-when-all-tracks-terminal" in args
    assert "--detach" in args
    assert "--implement" in common
    assert "--strict-task-sharding" in common
    assert "--objective-refill-scan" not in common
    assert "--codebase-refill-scan" not in common
    assert "--no-objective-task-janitor" in common
    assert common.count("--worktree-submodule-path") == 2
    assert set(board.worktree_submodule_paths).issubset(common)
    assert common.count("--implementation-protected-path") == len(
        board.protected_paths
    )
    assert plan["environment"] == {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
            "primary_quota_exhausted"
        ),
        "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.5",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "medium",
    }


def test_ordered_provider_contract_requires_complete_unambiguous_fields(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "max_concurrency": 2,
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        ConfiguredBoardError,
        match="fallback_model_id",
    ):
        load_configured_board(config_path, repo_root=repo)

    payload["provider"]["fallback_model_id"] = "gpt-5.6-terra"
    payload["provider"]["fallback_trigger"] = (
        "primary_quota_or_auth_unavailable"
    )
    payload["provider"]["fallback_reasoning_effort"] = "high"
    payload["provider"]["provider_id"] = "auto"
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        ConfiguredBoardError,
        match="cannot be mixed",
    ):
        load_configured_board(config_path, repo_root=repo)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("primary_provider_id", "claude"),
        ("primary_model_id", "grok-4"),
        ("fallback_provider_id", "openai"),
        ("fallback_model_id", "gpt-5.6"),
        ("fallback_trigger", "primary_unavailable"),
        ("fallback_reasoning_effort", "medium"),
    ),
)
def test_ordered_provider_contract_seals_fallback_authority(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_or_auth_unavailable",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
    }
    payload["provider"][field] = value
    _commit_v3_route_authorization(repo, config_path, payload)

    with pytest.raises(ConfiguredBoardError, match=field):
        load_configured_board(config_path, repo_root=repo)


def test_ordered_provider_contract_accepts_legacy_quota_medium_tuple(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "medium",
        "max_concurrency": 2,
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    board = load_configured_board(config_path, repo_root=repo)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260808T000000Z",
    )

    assert plan["environment"][scheduler_module.FALLBACK_TRIGGER_ENV] == (
        "primary_quota_exhausted"
    )
    assert plan["environment"][scheduler_module.CODEX_REASONING_EFFORT_ENV] == (
        "medium"
    )


def test_ordered_provider_contract_rejects_hybrid_legacy_trigger_high_effort(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ConfiguredBoardError, match="reviewed legacy"):
        load_configured_board(config_path, repo_root=repo)


def test_legacy_provider_launch_environment_remains_backward_compatible(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    board = load_configured_board(config_path, repo_root=repo)

    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260803T000000Z",
    )

    assert plan["environment"] == {
        scheduler_module.PROVIDER_ENV: "codex",
        scheduler_module.CODEX_MODEL_ENV: "test-model",
    }


def test_launch_config_overrides_ambient_provider_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_or_auth_unavailable",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
    }
    _commit_v3_route_authorization(repo, config_path, payload)
    expected_environment = configured_board_launch_plan(
        load_configured_board(config_path, repo_root=repo),
        implement=True,
        detach=False,
        stamp="20260808T000000Z",
    )["environment"]
    assert isinstance(expected_environment, dict)
    observed: dict[str, str | None] = {}
    controlled_names = scheduler_module.SCHEDULER_PROVIDER_ENV_NAMES
    for name in controlled_names:
        monkeypatch.setenv(name, "ambient-value")

    scheduler_module._apply_configured_board_environment(
        {"environment": expected_environment}
    )
    observed.update(
        {
            name: scheduler_module.os.environ.get(name)
            for name in controlled_names
        }
    )
    assert observed == {
        name: expected_environment.get(name) for name in controlled_names
    }


def test_sparse_legacy_launch_clears_stale_ordered_route_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {"max_concurrency": 2}
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "use sparse legacy provider config")
    for name in scheduler_module.SCHEDULER_PROVIDER_ENV_NAMES:
        monkeypatch.setenv(name, "stale-ordered-value")
    observed: dict[str, str | None] = {}

    def fake_multi_supervisor_main(_argv: list[str]) -> int:
        observed.update(
            {
                name: scheduler_module.os.environ.get(name)
                for name in scheduler_module.SCHEDULER_PROVIDER_ENV_NAMES
            }
        )
        return 0

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner.main",
        fake_multi_supervisor_main,
    )

    result = scheduler_module.main(
        [
            "--repo-root",
            str(repo),
            "--config",
            str(config_path),
            "launch",
            "--implement",
            "--foreground",
            "--duration-seconds",
            "1",
        ]
    )

    assert result == 0
    assert observed == {
        name: None for name in scheduler_module.SCHEDULER_PROVIDER_ENV_NAMES
    }


def test_v3_materializer_uses_canonical_ready_and_attempt_admissible_set(
    tmp_path: Path,
) -> None:
    _repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (
            _task_block("TEST-A", status="completed"),
            _task_block("TEST-B", depends_on=("TEST-A",)),
            _task_block("TEST-C", schedulable=False),
            _task_block("TEST-D", depends_on=("TEST-C",)),
            _task_block("TEST-E"),
            _task_block("TEST-F"),
        ),
    )
    population = scheduler_module._configured_board_task_population(
        board,
        source_head=_git(board.repo_root, "rev-parse", "HEAD").stdout.strip(),
        task_state_snapshots=({"implementation_attempts": {"TEST-E": 3}},),
    )
    assert population.completed_task_ids == ("TEST-A",)
    assert population.attempt_limited_task_ids == ("TEST-E",)
    assert tuple(item["task_id"] for item in population.ready_records) == (
        "TEST-B",
        "TEST-F",
    )
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(),
        provider_capacity_snapshots=_provider_capacity(),
        task_state_snapshots=({"implementation_attempts": {"TEST-E": 3}},),
    )
    assert receipt is not None
    assert {
        task_id
        for execution_slice in receipt.slice_manifest.slices
        for task_id in execution_slice.task_ids
    } == {"TEST-B", "TEST-F"}


def test_v3_materializer_rejects_unsafe_or_ambiguous_attempt_state(
    tmp_path: Path,
) -> None:
    _repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    state_root = board.path(board.runtime_paths["state"])
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "unsafe_task_state.json"
    common = {
        "now_ms": PLAN_NOW,
        "host_capacity_snapshot": _host_capacity(lanes=1),
        "provider_capacity_snapshots": _provider_capacity(lanes=1),
    }

    target = state_root / "state-target.json"
    _write(
        target,
        json.dumps({"implementation_attempts": {"TEST-A": 3}}) + "\n",
    )
    state_path.symlink_to(target)
    with pytest.raises(ConfiguredBoardError, match="projection.*symbolic"):
        materialize_configured_board_execution_plan(board, **common)
    state_path.unlink()
    target.unlink()

    outside_state = tmp_path / "outside-state"
    outside_state.mkdir()
    _write(
        outside_state / "escaped_task_state.json",
        json.dumps({"implementation_attempts": {"TEST-A": 3}}) + "\n",
    )
    linked_state = state_root / "linked-state"
    linked_state.symlink_to(outside_state, target_is_directory=True)
    with pytest.raises(ConfiguredBoardError, match="projection.*symbolic"):
        materialize_configured_board_execution_plan(board, **common)
    linked_state.unlink()

    _write(
        state_path,
        (
            '{"implementation_attempts":{"TEST-A":3},'
            '"implementation_attempts":{},'
            '"implementation_attempts_by_cid":{}}\n'
        ),
    )
    with pytest.raises(ConfiguredBoardError, match="projection is unreadable"):
        materialize_configured_board_execution_plan(board, **common)

    _write(
        state_path,
        json.dumps(
            {
                "implementation_attempts": {"TEST-A": 0.1},
                "implementation_attempts_by_cid": {},
            },
            sort_keys=True,
        )
        + "\n",
    )
    with pytest.raises(ConfiguredBoardError, match="attempt count is invalid"):
        materialize_configured_board_execution_plan(board, **common)


def test_v3_materializer_requires_fresh_unsaturated_provider_evidence(
    tmp_path: Path,
) -> None:
    _repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    common = {
        "now_ms": PLAN_NOW,
        "host_capacity_snapshot": _host_capacity(lanes=1),
        "task_state_snapshots": (),
    }
    with pytest.raises(ConfiguredBoardError, match="provider capacity"):
        materialize_configured_board_execution_plan(
            board, provider_capacity_snapshots=(), **common
        )
    with pytest.raises(ExecutionPlanError):
        materialize_configured_board_execution_plan(
            board,
            provider_capacity_snapshots=_provider_capacity(
                lanes=1,
                observed_at_ms=PLAN_NOW - 20_000,
            ),
            **common,
        )
    with pytest.raises(ExecutionPlanError):
        materialize_configured_board_execution_plan(
            board,
            provider_capacity_snapshots=_provider_capacity(
                lanes=1,
                active=1,
            ),
            **common,
        )


def test_route_capacity_projection_is_router_owned_strict_and_fallback_only(
    tmp_path: Path,
) -> None:
    _repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    fallback_only = _provider_capacity(
        lanes=2,
        primary_healthy=False,
        fallback_healthy=True,
    )
    capacity, route = scheduler_module.configured_board_route_capacity_projection(
        board,
        provider_capacity_snapshots=fallback_only,
        now_ms=PLAN_NOW,
    )
    assert capacity["schema"].endswith("implementation-route-capacity@2")
    assert capacity["route_id"] == route.route_id == V3_ROUTE_ID
    assert capacity["provider_id"] == route.route_id
    assert capacity["healthy"] is True
    assert capacity["schedulable"] is True
    assert capacity["available_concurrency"] == 2
    assert capacity["profile_id"].startswith("sha256:")
    lanes = {item["role"]: item for item in capacity["lanes"]}
    assert lanes["primary"]["capacity_available"] is False
    assert lanes["typed_fallback_capacity_only"] == {
        **lanes["typed_fallback_capacity_only"],
        "provider_id": "codex",
        "model_id": "gpt-5.6-terra",
        "reasoning_effort": "high",
        "capacity_available": True,
        "dispatch_authorized": False,
    }
    assert all(item["dispatch_authorized"] is False for item in lanes.values())

    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(lanes=2),
        provider_capacity_snapshots=fallback_only,
        task_state_snapshots=(),
    )
    assert receipt is not None
    assert receipt.slice_manifest.capacity_snapshot_id == (
        receipt.binding.capacity_snapshot_id
    )

    both_unavailable = _provider_capacity(
        lanes=2,
        primary_healthy=False,
        fallback_healthy=False,
    )
    denied, _route = scheduler_module.configured_board_route_capacity_projection(
        board,
        provider_capacity_snapshots=both_unavailable,
        now_ms=PLAN_NOW,
    )
    assert denied["schedulable"] is False
    assert denied["available_concurrency"] == 0
    with pytest.raises(ExecutionPlanError):
        materialize_configured_board_execution_plan(
            board,
            now_ms=PLAN_NOW,
            host_capacity_snapshot=_host_capacity(lanes=2),
            provider_capacity_snapshots=both_unavailable,
            task_state_snapshots=(),
        )

    canonical = [dict(item) for item in _provider_capacity(lanes=1)]
    extra = [dict(item) for item in canonical]
    extra[0]["scheduler_guess"] = True
    wrong_type = [dict(item) for item in canonical]
    wrong_type[0]["observed_at_ms"] = float(PLAN_NOW)
    mixed = [dict(item) for item in canonical]
    mixed[1]["provider_id"] = "grok"
    adversarial = (
        canonical[:1],
        [*canonical, dict(canonical[0])],
        extra,
        wrong_type,
        mixed,
    )
    for observations in adversarial:
        with pytest.raises(ConfiguredBoardError, match="router rejected"):
            scheduler_module.configured_board_route_capacity_projection(
                board,
                provider_capacity_snapshots=observations,
                now_ms=PLAN_NOW,
            )

    aliases = [dict(item) for item in canonical]
    aliases[0]["provider_id"] = "grok"
    aliases[1]["provider_id"] = "codex"
    aliased, _route = scheduler_module.configured_board_route_capacity_projection(
        board,
        provider_capacity_snapshots=aliases,
        now_ms=PLAN_NOW,
    )
    assert aliased["route_id"] == V3_ROUTE_ID
    assert aliased["schedulable"] is True

    stale, _route = scheduler_module.configured_board_route_capacity_projection(
        board,
        provider_capacity_snapshots=_provider_capacity(
            lanes=1,
            observed_at_ms=PLAN_NOW - 20_000,
        ),
        now_ms=PLAN_NOW,
    )
    assert stale["schedulable"] is False

    legacy_route = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    )
    legacy = llm_router.project_agent_implementation_route_capacity(
        legacy_route,
        observations=[dict(item) for item in fallback_only],
        now_ms=PLAN_NOW,
        max_age_ms=5_000,
    ).as_compiler_snapshot()
    assert legacy["schedulable"] is False
    assert all(item["dispatch_authorized"] is False for item in legacy["lanes"])


def test_v3_materializer_denies_same_id_cid_and_same_namespace_config_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = tmp_path / "task-race"
    task_case.mkdir()
    _repo, _config_path, board = _seed_v3_task_repo(
        task_case,
        (_task_block("TEST-A", output="src/original.py"),),
    )
    taskboard = board.path(board.taskboard_path)
    canonical_snapshot = scheduler_module._tracked_head_snapshot
    task_reads = 0

    def change_same_id_before_publish(**kwargs):
        nonlocal task_reads
        if Path(kwargs["path"]) == taskboard:
            task_reads += 1
            if task_reads == 2:
                _write(
                    taskboard,
                    "# Tasks\n\n"
                    + _task_block("TEST-A", output="src/identity-drift.py"),
                )
        return canonical_snapshot(**kwargs)

    monkeypatch.setattr(
        scheduler_module,
        "_tracked_head_snapshot",
        change_same_id_before_publish,
    )
    with pytest.raises(ConfiguredBoardError, match="authority|differs|changed"):
        materialize_configured_board_execution_plan(
            board,
            now_ms=PLAN_NOW,
            host_capacity_snapshot=_host_capacity(lanes=1),
            provider_capacity_snapshots=_provider_capacity(lanes=1),
            task_state_snapshots=(),
        )
    store_root = board.path(board.runtime_paths["state"]) / "plan-revision-store"
    assert PlanRevisionStore(store_root).get_active() is None

    config_case = tmp_path / "config-race"
    config_case.mkdir()
    _repo, config_path, board = _seed_v3_task_repo(
        config_case,
        (_task_block("TEST-A"),),
    )
    canonical_load = scheduler_module.load_configured_board
    config_loads = 0

    def change_same_namespace_before_publish(*args, **kwargs):
        nonlocal config_loads
        loaded = canonical_load(*args, **kwargs)
        config_loads += 1
        if config_loads == 1:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["poll_interval_seconds"] = 6
            _write(
                config_path,
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )
        return loaded

    with monkeypatch.context() as config_context:
        config_context.setattr(
            scheduler_module,
            "load_configured_board",
            change_same_namespace_before_publish,
        )
        with pytest.raises(
            ConfiguredBoardError,
            match="configuration|differs|changed",
        ):
            materialize_configured_board_execution_plan(
                board,
                now_ms=PLAN_NOW,
                host_capacity_snapshot=_host_capacity(lanes=1),
                provider_capacity_snapshots=_provider_capacity(lanes=1),
                task_state_snapshots=(),
            )
    store_root = board.path(board.runtime_paths["state"]) / "plan-revision-store"
    assert not store_root.exists()


def test_v3_revision_adopts_exact_wave_then_steers_on_capacity_and_head_drift(
    tmp_path: Path,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"), _task_block("TEST-B")),
    )

    def materialize(lanes: int):
        result = materialize_configured_board_execution_plan(
            board,
            now_ms=PLAN_NOW,
            host_capacity_snapshot=_host_capacity(lanes=lanes),
            provider_capacity_snapshots=_provider_capacity(lanes=lanes),
            task_state_snapshots=(),
        )
        assert result is not None
        return result

    first = materialize(2)
    adopted = materialize(2)
    narrowed = materialize(1)
    assert adopted.binding.revision_cid == first.binding.revision_cid
    assert adopted.slice_manifest_cid == first.slice_manifest_cid
    assert narrowed.binding.semantic_revision == first.binding.semantic_revision + 1
    assert len(first.slice_manifest.slices) == 2
    assert len(narrowed.slice_manifest.slices) == 1
    narrowed_revision = scheduler_module.PlanRevisionStore(
        board.path(board.runtime_paths["state"]) / "plan-revision-store"
    ).load_revision(narrowed.binding.revision_cid)
    assert narrowed_revision.parent_plan_root == first.binding.plan_root_cid
    assert narrowed_revision.origin.value == "steer"

    _write(repo / "README.md", "configured board fixture\nnew head\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "advance integration head")
    head_replan = materialize(1)
    assert head_replan.binding.semantic_revision == narrowed.binding.semantic_revision + 1
    assert head_replan.binding.repository_tree_id != narrowed.binding.repository_tree_id


def test_v3_launch_uses_only_exact_plan_slices_and_empty_wave_has_no_child(
    tmp_path: Path,
) -> None:
    _repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"), _task_block("TEST-B")),
    )
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(),
        provider_capacity_snapshots=_provider_capacity(),
        task_state_snapshots=(),
    )
    assert receipt is not None
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260808T010101Z",
        parallelism_receipt=receipt,
    )
    argv = list(plan["argv"])
    records = [
        json.loads(argv[index + 1])
        for index, token in enumerate(argv[:-1])
        if token == "--implementation-plan-bound-track"
    ]
    assert len(records) == len(receipt.slice_manifest.nonempty) == 2
    assert {tuple(item["task_ids"]) for item in records} == {
        ("TEST-A",),
        ("TEST-B",),
    }
    assert "--implementation-supervisor-lanes-per-track" not in argv
    assert "--implementation-supervisor-strict-task-sharding" not in argv
    assert "--detach" not in argv
    assert plan["effective_strict_task_sharding"] is False

    empty = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        stamp="20260808T010102Z",
        parallelism_receipt=None,
    )
    assert "--plan-bound-wave" in empty["argv"]
    assert "--implementation-plan-bound-track" not in empty["argv"]
    assert empty["admitted_lanes"] == 0


@pytest.mark.parametrize(
    "bridge_scenario", ("normal", "scope_drift", "tamper", "mode_tamper")
)
def test_plan_bound_child_bootstraps_existing_daemon_preclaim_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bridge_scenario: str,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(lanes=1),
        provider_capacity_snapshots=_provider_capacity(lanes=1),
        task_state_snapshots=(),
    )
    assert receipt is not None
    execution_slice = receipt.slice_manifest.slices[0]
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=receipt.slice_manifest.source_head,
        source_tree=receipt.slice_manifest.repository_tree_id,
    )
    state_dir = board.path(board.runtime_paths["state"]) / "lane-0"
    config = supervisor_module.PortalSupervisorConfig(
        todo_path=board.path(board.taskboard_path),
        state_path=state_dir / "test_task_state.json",
        strategy_path=state_dir / "test_strategy.json",
        events_path=state_dir / "test_supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix=board.task_header_prefix,
        state_prefix="test_lane_0",
        implement=True,
        max_task_attempts=board.payload["max_task_attempts"],
        worktree_root=board.path(board.runtime_paths["worktrees"]),
        merge_target_branch=board.merge_target_branch,
        merge_queue_dir=board.path(board.runtime_paths["merge_queue"]),
        task_shard_count=1,
        task_shard_index=0,
        strict_task_sharding=False,
        scheduler_config_path=_config_path,
        execution_slice_task_ids=execution_slice.task_ids,
        execution_slice_task_cids=execution_slice.task_cids,
        plan_bound_dispatch=True,
        plan_revision_store_path=(
            board.path(board.runtime_paths["state"]) / "plan-revision-store"
        ),
        plan_bound_revision_cid=receipt.binding.revision_cid,
        plan_bound_plan_root_cid=receipt.binding.plan_root_cid,
        plan_bound_execution_plan_cid=receipt.binding.execution_plan_cid,
        plan_bound_capacity_snapshot_id=receipt.binding.capacity_snapshot_id,
        plan_bound_slice_manifest_cid=receipt.slice_manifest_cid,
        plan_bound_slice_id=execution_slice.slice_id,
        plan_bound_lane_id=execution_slice.lane_id,
        plan_bound_source_head=receipt.slice_manifest.source_head,
        plan_bound_source_tree=receipt.slice_manifest.repository_tree_id,
        plan_bound_task_source_revision=receipt.slice_manifest.task_source_revision,
        plan_bound_configuration_root=receipt.slice_manifest.configuration_root,
        plan_bound_accepted_tree_root=repo,
        accepted_control_plane_pin=control_plane_pin,
        accepted_control_plane_descriptor=control_plane_launch.descriptor,
    )
    supervisor = supervisor_module.PortalImplementationSupervisor(config)
    with monkeypatch.context() as build_context:
        build_context.setattr(
            supervisor_module.PortalImplementationSupervisor,
            "_validated_plan_bound_slice",
            lambda _self: None,
        )
        command = supervisor._build_daemon_command()
    marker = supervisor_module.PLAN_BOUND_DAEMON_CHILD_MARKER
    assert Path(command[0]).samefile(sys.executable)
    assert command[1:4] == [
        "-I",
        "-c",
        multi_runner_module.SEALED_CONTROL_PLANE_BOOTSTRAP,
    ]
    assert command[4] == str(control_plane_launch.descriptor)
    assert json.loads(command[5]) == control_plane_pin.as_dict()
    assert command[6] == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_supervisor"
    )
    assert command[7] == (
        multi_runner_module.SEALED_CONTROL_PLANE_BOOTSTRAP_SHA256
    )
    assert marker in command
    assert supervisor_module.PLAN_BOUND_DAEMON_ENTRYPOINT in command
    assert command.count("--execution-slice-task-id") == 0
    assert command.count("--execution-slice-task-cid") == 1
    assert "--strict-task-sharding" not in command
    assert command[command.index("--task-shard-count") + 1] == "1"
    assert command[command.index("--task-shard-index") + 1] == "0"
    assert "--once" in command

    captured: dict[str, object] = {}

    def bounded_run_once(self):
        tasks = daemon_module.parse_task_file(
            self.todo_path,
            task_header_prefix=self.task_header_prefix,
        )
        assert [task.task_id for task in tasks] == ["TEST-A"]
        assert tasks[0].canonical_task_cid == execution_slice.task_cids[0]
        assert self._canonical_ref(tasks[0]) == execution_slice.task_cids[0]
        decision = self._require_plan_runtime_before_claim(
            tasks[0],
            tasks=tasks,
        )
        captured["decision"] = decision
        captured["claim"] = self._build_implementation_task_claim_metadata(
            tasks[0],
            1,
            "2026-08-08T00:00:00+00:00",
        )
        claim_path = self._implementation_task_claim_path(
            tasks[0].task_id,
            canonical_task_cid=execution_slice.task_cids[0],
        )
        acquired, reason, _existing = (
            self._try_acquire_implementation_task_claim(
                claim_path,
                captured["claim"],
            )
        )
        assert acquired is True, reason
        workspace = self.worktree_root / "execution-lease-bridge-workspace"
        if bridge_scenario == "scope_drift":
            workspace.parent.mkdir(parents=True, exist_ok=True)
            _git(repo, "worktree", "add", "--detach", str(workspace), "HEAD")
        lifecycle = self.worktree_lifecycle.begin_preparing(
            task_id=tasks[0].task_id,
            canonical_task_cid=execution_slice.task_cids[0],
            attempt=1,
            lane_id=self._worktree_lifecycle_lane_id(),
            workspace_path=workspace,
            branch="implementation/execution-lease-bridge",
            merge_target=self._main_branch_name(),
            state_dir=str(self.state_path.parent.resolve()),
        )
        self._active_worktree_lifecycle = lifecycle
        captured["command"] = self._build_implementation_command(
            workspace,
            task=tasks[0],
        )
        lifecycle = self.worktree_lifecycle.mark_active(
            lifecycle.workspace_path,
            lease_id=lifecycle.lease_id,
            expected_fence=lifecycle.fence,
        )
        self._active_worktree_lifecycle = lifecycle
        captured["provider_called"] = False

        def provider_effect():
            captured["provider_called"] = True
            return "provider-effect-called"

        captured["plan_store"] = self.plan_revision_store
        captured["require_active"] = self.require_active_plan_revision
        if bridge_scenario in {"tamper", "mode_tamper"}:
            lifecycle_path = self.worktree_lifecycle.workspace_path_for(
                Path(lifecycle.workspace_path)
            )
            if bridge_scenario == "tamper":
                tampered = json.loads(
                    lifecycle_path.read_text(encoding="utf-8")
                )
                tampered["fence"] = int(tampered["fence"]) + 1
                lifecycle_path.write_text(
                    json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            else:
                lifecycle_path.chmod(0o644)
        try:
            captured["provider_result"] = self._decision_runtime_mutation(
                "command_invocation",
                {
                    "operation": "implementation_provider",
                    "task_id": tasks[0].task_id,
                    "attempt": 1,
                    "command": tuple(captured["command"]),
                    "workspace_path": lifecycle.workspace_path,
                    "branch": lifecycle.branch,
                    "pre_implementation_receipt_cid": "receipt:test",
                },
                provider_effect,
            )
        except supervisor_module.PlanBoundDispatchError:
            self._release_implementation_task_claim(
                claim_path,
                captured["claim"],
            )
            raise
        if bridge_scenario == "scope_drift":
            _write(workspace / "src/test-a.py", "VALUE = 'declared'\n")
            _write(workspace / "src/undeclared-shared.py", "VALUE = 'drift'\n")
            try:
                self._validate_implementation_patch(
                    workspace,
                    tasks[0],
                    baseline_ref="HEAD",
                    allow_scope_adjudication=False,
                )
            finally:
                self._release_implementation_task_claim(
                    claim_path,
                    captured["claim"],
                )
        assert self._release_implementation_task_claim(
            claim_path,
            captured["claim"],
        )
        return {"reason": "bounded_plan_bootstrap_test"}

    monkeypatch.setattr(daemon_module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        supervisor_module,
        "__file__",
        str(
            repo
            / "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_supervisor.py"
        ),
    )
    original_capacity_observation = (
        scheduler_module.configured_board_capacity_observation
    )

    def capacity_observation(current_board, **kwargs):
        if kwargs:
            return original_capacity_observation(current_board, **kwargs)
        return (
            _host_capacity(lanes=1),
            _provider_capacity(lanes=1),
            PLAN_NOW,
        )

    monkeypatch.setattr(
        scheduler_module,
        "configured_board_capacity_observation",
        capacity_observation,
    )
    canonical_evaluate_plan_runtime_dispatch = (
        daemon_module.evaluate_plan_runtime_dispatch
    )

    def capture_plan_runtime_dispatch(*args, **kwargs):
        captured["passed_task_cid"] = kwargs.get("task_cid")
        return canonical_evaluate_plan_runtime_dispatch(*args, **kwargs)

    monkeypatch.setattr(
        daemon_module,
        "evaluate_plan_runtime_dispatch",
        capture_plan_runtime_dispatch,
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "run_once",
        bounded_run_once,
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_build_implementation_command",
        lambda _self, _workspace, *, task=None: ["provider-test-command"],
    )
    store_path = board.path(board.runtime_paths["state"]) / "plan-revision-store"
    launch_argv = (
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        "--plan-revision-store-path",
        str(store_path),
        "--plan-bound-revision-cid",
        receipt.binding.revision_cid,
        "--plan-bound-plan-root-cid",
        receipt.binding.plan_root_cid,
        "--plan-bound-execution-plan-cid",
        receipt.binding.execution_plan_cid,
        "--plan-bound-capacity-snapshot-id",
        receipt.binding.capacity_snapshot_id,
        "--plan-bound-slice-manifest-cid",
        receipt.slice_manifest_cid,
        "--plan-bound-slice-id",
        execution_slice.slice_id,
        "--plan-bound-lane-id",
        execution_slice.lane_id,
        "--plan-bound-source-head",
        receipt.slice_manifest.source_head,
        "--plan-bound-source-tree",
        receipt.slice_manifest.repository_tree_id,
        "--plan-bound-task-source-revision",
        receipt.slice_manifest.task_source_revision,
        "--plan-bound-configuration-root",
        receipt.slice_manifest.configuration_root,
        "--plan-bound-accepted-tree-root",
        str(repo),
        "--execution-slice-task-id",
        execution_slice.task_ids[0],
        "--execution-slice-task-cid",
        execution_slice.task_cids[0],
    )
    state_root = state_dir.parent
    launch_profile = multi_runner_module.LifecycleProfile(
        target_id="supervisor-track:execution-lease-bridge",
        run_id="test-execution-lease-bridge",
        configuration_root="test-execution-lease-bridge-config",
        repository_root=str(repo),
        state_root=str(state_root),
        run_root=str(state_root / "lifecycle-runs/execution-lease-bridge"),
        argv=launch_argv,
        cwd=str(repo),
    )
    launch_process = subprocess.Popen(
        launch_argv,
        cwd=repo,
        env=launch_profile.launch_environment(0),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    launch_identity = multi_runner_module.LinuxProcessAdapter()._identity(
        launch_process.pid,
        launch_profile,
    )
    multi_runner_module._persist_plan_bound_process_birth(
        profile=launch_profile,
        process_identity=launch_identity,
        repo_root=repo,
    )
    helper_argv = command[command.index(marker) + 1 :]
    try:
        if bridge_scenario in {"scope_drift", "tamper", "mode_tamper"}:
            with pytest.raises(
                supervisor_module.PlanBoundDispatchError,
                match=(
                    "scope drift fenced before merge enqueue"
                    if bridge_scenario == "scope_drift"
                    else (
                        "lease or fence changed before provider"
                        if bridge_scenario == "tamper"
                        else "exactly mode 0600"
                    )
                ),
            ):
                supervisor_module._run_plan_bound_daemon_child(helper_argv)
        else:
            assert supervisor_module._run_plan_bound_daemon_child(helper_argv) == 0
    finally:
        launch_process.terminate()
        launch_process.wait(timeout=5)
        workspace = board.path(board.runtime_paths["worktrees"]) / (
            "execution-lease-bridge-workspace"
        )
        if workspace.exists():
            _git(repo, "worktree", "remove", "--force", str(workspace))
    assert captured["passed_task_cid"] == execution_slice.task_cids[0]
    assert captured["decision"] is None
    claim = captured["claim"]
    assert isinstance(claim, dict)
    assert claim["compiled_claim_acquired_before_publish"] is True
    assert claim["plan_revision_cid"] == receipt.binding.revision_cid
    assert claim["compiled_lease_id"]
    assert claim["compiled_worktree_id"]
    assert str(claim["compiled_fence_token"]).startswith("fence:sha256:")
    assert int(claim["pid"]) > 0
    assert captured["require_active"] is True
    assert captured["command"] == ["provider-test-command"]
    if bridge_scenario in {"tamper", "mode_tamper"}:
        assert captured.get("provider_called") is not True
        assert "provider_result" not in captured
    else:
        assert captured["provider_result"] == "provider-effect-called"
    execution_store = PlanRevisionStore(store_path)
    with execution_store._thread_lock:
        with execution_store._guard():
            current_execution = (
                execution_plan_module._load_plan_bound_execution_lease_locked(
                    execution_store,
                    revision_cid=receipt.binding.revision_cid,
                    slice_id=execution_slice.slice_id,
                    lane_id=execution_slice.lane_id,
                )
            )
    assert current_execution is not None
    assert current_execution[1].phase == {
        "normal": "provider_ready",
        "scope_drift": "scope_drift",
        "tamper": "workspace_prepared",
        "mode_tamper": "workspace_prepared",
    }[bridge_scenario]
    assert current_execution[1].canonical_claim_cid
    assert current_execution[1].workspace_path.endswith(
        "execution-lease-bridge-workspace"
    )
    assert current_execution[1].workspace_lease_id
    assert current_execution[1].workspace_fence >= 1
    if bridge_scenario == "scope_drift":
        assert current_execution[1].merge_enqueue_reached is False
        assert current_execution[1].proposal_id
        assert current_execution[1].proposal_receipt_id
        assert "path_outside_scope" in (
            current_execution[1].proposal_reason_codes
        )
        assert "src/undeclared-shared.py" in (
            current_execution[1].actual_changed_paths
        )

    retry_process = subprocess.Popen(
        launch_argv,
        cwd=repo,
        env=launch_profile.launch_environment(1),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    retry_deadline = time.monotonic() + 2.0
    while True:
        try:
            retry_identity = (
                multi_runner_module.LinuxProcessAdapter()._identity(
                    retry_process.pid,
                    launch_profile,
                )
            )
            break
        except multi_runner_module.ProcessIdentityMismatch:
            if time.monotonic() >= retry_deadline:
                raise
            time.sleep(0.01)
    try:
        with pytest.raises(
            ValueError,
            match=(
                "provider boundary"
                if bridge_scenario in {"normal", "scope_drift"}
                else "daemon process is not provably dead"
            ),
        ):
            multi_runner_module._persist_plan_bound_process_birth(
                profile=launch_profile,
                process_identity=retry_identity,
                repo_root=repo,
            )
    finally:
        retry_process.terminate()
        retry_process.wait(timeout=5)
    with execution_store._thread_lock:
        with execution_store._guard():
            after_retry = (
                execution_plan_module._load_plan_bound_execution_lease_locked(
                    execution_store,
                    revision_cid=receipt.binding.revision_cid,
                    slice_id=execution_slice.slice_id,
                    lane_id=execution_slice.lane_id,
                )
            )
    assert after_retry == current_execution

    launch_plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        stamp="20260808T-scope-drift",
        parallelism_receipt=receipt,
        accepted_control_plane_pin=control_plane_pin,
        accepted_control_plane_descriptor=control_plane_launch.descriptor,
    )
    child_records = [
        launch_plan["argv"][index + 1]
        for index, token in enumerate(launch_plan["argv"][:-1])
        if token == "--implementation-plan-bound-track"
    ]
    assert len(child_records) == 1
    child = multi_runner_module.PlanBoundSupervisorChild.from_cli_record(
        child_records[0]
    )
    assert child.revision_cid == receipt.binding.revision_cid
    assert child.slice_id == execution_slice.slice_id
    assert child.lane_id == execution_slice.lane_id
    assert repo / child.plan_revision_store_path == store_path
    drift_receipt = multi_runner_module._plan_bound_scope_drift_receipt(child)
    if bridge_scenario == "scope_drift":
        assert drift_receipt is not None
        assert drift_receipt["merge_enqueue_reached"] is False
        assert drift_receipt["proposal_receipt_id"] == (
            current_execution[1].proposal_receipt_id
        )
        with monkeypatch.context() as runner_patch:
            runner_patch.setattr(
                multi_runner_module,
                "start_track",
                lambda *_args, **_kwargs: launch_process,
            )
            runner_patch.setattr(
                multi_runner_module,
                "_terminate_managed_process",
                lambda *_args, **_kwargs: (True, []),
            )
            runner_patch.setattr(
                multi_runner_module,
                "stop_tracks",
                lambda *_args, **_kwargs: {
                    "stopped_count": 1,
                    "all_trees_fenced": True,
                    "removed_runtime_markers": [],
                },
            )
            runner_result = multi_runner_module.run_supervisor_tracks(
                (child.track(stamp="20260808T-scope-drift-runner"),),
                repo_root=repo,
                common_args=(),
                duration_seconds=0.1,
                heartbeat_interval_seconds=0.01,
                stop_grace_seconds=0.01,
                exit_when_all_tracks_terminal=True,
                plan_bound_children=(child,),
                accepted_control_plane_pin=control_plane_pin,
                accepted_control_plane_descriptor=(
                    control_plane_launch.descriptor
                ),
                output=lambda _message: None,
            )
        assert runner_result["replan_required"] is True
        assert runner_result["scope_drift_receipts"] == [drift_receipt]
        assert runner_result["all_trees_fenced"] is True
        assert runner_result["completed"] is False

        with monkeypatch.context() as main_patch:
            main_patch.setattr(
                multi_runner_module,
                "seal_ordered_implementation_provider_route",
                lambda **_kwargs: {},
            )
            main_patch.setattr(
                "ipfs_accelerate_py.agent_supervisor.runtime."
                "provider_command_binding.preflight_provider_entry_module",
                lambda _module: None,
            )
            main_patch.setattr(
                multi_runner_module,
                "run_supervisor_tracks",
                lambda *_args, **_kwargs: runner_result,
            )
            assert (
                multi_runner_module.main(list(launch_plan["argv"]))
                == multi_runner_module.PLAN_BOUND_REPLAN_RETURN_CODE
            )
    else:
        assert drift_receipt is None

    replacement = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(lanes=2),
        provider_capacity_snapshots=_provider_capacity(lanes=2),
        task_state_snapshots=(),
    )
    assert replacement is not None
    assert replacement.binding.revision_cid != receipt.binding.revision_cid
    if bridge_scenario == "scope_drift":
        assert len(replacement.slice_manifest.slices) == 1
        replacement_revision = execution_store.load_revision(
            replacement.binding.revision_cid
        )
        assert replacement_revision.origin.value == "steer"
        assert replacement_revision.conflict_contract.conflict_surface_cid
        assert "src/undeclared-shared.py" in (
            replacement_revision.conflict_contract.predicted_files
        )
    with pytest.raises(
        supervisor_module.PlanBoundDispatchError,
        match="revision|fence",
    ):
        supervisor._build_daemon_command()


def test_fenced_slice_reassignment_has_one_cas_winner_and_recipient_adopts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board, receipt, donor, recipient, process = _fenced_plan_children(
        tmp_path
    )
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=receipt.slice_manifest.source_head,
        source_tree=receipt.slice_manifest.repository_tree_id,
    )
    config_path = board.config_path
    assert getattr(process, "_agent_supervisor_process_identity", None) is not None
    assert process.returncode not in (None, 0)

    def attempt_reassignment():
        return multi_runner_module.reassign_fenced_plan_bound_child(
            donor=donor,
            recipient=recipient,
            donor_process=process,
            repo_root=repo,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt_reassignment) for _ in range(2)]
    adopted = []
    rejected = []
    for future in futures:
        try:
            adopted.append(future.result())
        except Exception as exc:  # noqa: BLE001 - assert exact one-winner boundary
            rejected.append(exc)
    assert len(adopted) == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], ExecutionClaimConflictError)
    adopted_child = adopted[0]
    assert adopted_child.revision_cid == donor.revision_cid
    assert adopted_child.task_ids == donor.task_ids
    assert adopted_child.task_cids == donor.task_cids
    assert adopted_child.lane_id == recipient.lane_id
    assert adopted_child.reassignment_cid

    adopted_store_path = Path(adopted_child.plan_revision_store_path)
    if not adopted_store_path.is_absolute():
        adopted_store_path = repo / adopted_store_path
    store = PlanRevisionStore(adopted_store_path.resolve())
    adapter = ProductionParallelPlanAdapter(store)
    current = adapter.load_slice_reassignment(
        revision_cid=donor.revision_cid,
        slice_id=donor.slice_id,
    )
    assert current is not None
    assert current[0] == adopted_child.reassignment_cid
    reassignment = current[1]
    assert reassignment.generation == 1
    process_evidence = store.get_cas(reassignment.donor_process_birth_cid)
    attempt_evidence = store.get_cas(reassignment.attempt_absence_cid)
    claim_evidence = store.get_cas(reassignment.claim_absence_cid)
    assert process_evidence["process_birth"]["identity_id"]
    assert process_evidence["fenced_tree"]["members"] == []
    assert attempt_evidence["never_attempted"] is True
    assert attempt_evidence["state_identity"]["state"] == "absent"
    assert claim_evidence["task_ids"] == list(donor.task_ids)
    assert claim_evidence["task_cids"] == list(donor.task_cids)
    assert {item["state"] for item in claim_evidence["claims"]} == {"absent"}
    with pytest.raises(ExecutionPlanError, match="owner|lane|CAS"):
        adapter.validate_slice_owner(
            revision_cid=donor.revision_cid,
            slice_manifest_cid=donor.slice_manifest_cid,
            slice_id=donor.slice_id,
            lane_id=donor.lane_id,
        )
    assert adapter.validate_slice_owner(
        revision_cid=adopted_child.revision_cid,
        slice_manifest_cid=adopted_child.slice_manifest_cid,
        slice_id=adopted_child.slice_id,
        lane_id=adopted_child.lane_id,
        reassignment_cid=adopted_child.reassignment_cid,
    ).task_ids == donor.task_ids

    def child_config(
        child: multi_runner_module.PlanBoundSupervisorChild,
    ) -> supervisor_module.PortalSupervisorConfig:
        state_dir = Path(child.state_dir)
        if not state_dir.is_absolute():
            state_dir = repo / state_dir
        store_path = Path(child.plan_revision_store_path)
        if not store_path.is_absolute():
            store_path = repo / store_path
        return supervisor_module.PortalSupervisorConfig(
            todo_path=board.path(board.taskboard_path),
            state_path=state_dir / f"{child.state_prefix}_task_state.json",
            strategy_path=state_dir / f"{child.state_prefix}_strategy.json",
            events_path=state_dir / f"{child.state_prefix}_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix=board.task_header_prefix,
            state_prefix=child.state_prefix,
            implement=True,
            max_task_attempts=board.payload["max_task_attempts"],
            worktree_root=board.path(board.runtime_paths["worktrees"]),
            merge_target_branch=board.merge_target_branch,
            merge_queue_dir=board.path(board.runtime_paths["merge_queue"]),
            task_shard_count=1,
            task_shard_index=0,
            strict_task_sharding=False,
            scheduler_config_path=config_path,
            execution_slice_task_ids=child.task_ids,
            execution_slice_task_cids=child.task_cids,
            plan_bound_dispatch=True,
            plan_revision_store_path=store_path,
            plan_bound_revision_cid=child.revision_cid,
            plan_bound_plan_root_cid=child.plan_root_cid,
            plan_bound_execution_plan_cid=child.execution_plan_cid,
            plan_bound_capacity_snapshot_id=child.capacity_snapshot_id,
            plan_bound_slice_manifest_cid=child.slice_manifest_cid,
            plan_bound_slice_id=child.slice_id,
            plan_bound_lane_id=child.lane_id,
            plan_bound_reassignment_cid=child.reassignment_cid,
            plan_bound_source_head=child.source_head,
            plan_bound_source_tree=child.source_tree,
            plan_bound_task_source_revision=child.task_source_revision,
                plan_bound_configuration_root=child.configuration_root,
                plan_bound_accepted_tree_root=Path(child.accepted_tree_root),
                accepted_control_plane_pin=control_plane_pin,
                accepted_control_plane_descriptor=(
                    control_plane_launch.descriptor
                ),
            )

    # The unit repository stands in for the accepted import tree.  Preserve
    # that production invariant while exercising owner loss and adoption.
    monkeypatch.setattr(
        supervisor_module,
        "__file__",
        str(
            repo
            / "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_supervisor.py"
        ),
    )
    with pytest.raises(
        supervisor_module.PlanBoundDispatchError,
        match="own|canonical",
    ):
        supervisor_module.PortalImplementationSupervisor(
            child_config(donor)
        )._build_daemon_command()

    adopted_supervisor = supervisor_module.PortalImplementationSupervisor(
        child_config(adopted_child)
    )
    with monkeypatch.context() as build_context:
        build_context.setattr(
            supervisor_module.PortalImplementationSupervisor,
            "_validated_plan_bound_slice",
            lambda _self: None,
        )
        command = adopted_supervisor._build_daemon_command()
    assert command[command.index("--plan-bound-lane-id") + 1] == recipient.lane_id
    assert command[command.index("--plan-bound-reassignment-cid") + 1] == (
        adopted_child.reassignment_cid
    )

    captured: dict[str, object] = {}

    def bounded_run_once(self):
        tasks = daemon_module.parse_task_file(
            self.todo_path,
            task_header_prefix=self.task_header_prefix,
        )
        task = next(item for item in tasks if item.task_id == donor.task_ids[0])
        captured["decision"] = self._require_plan_runtime_before_claim(
            task,
            tasks=tasks,
        )
        captured["claim"] = self._build_implementation_task_claim_metadata(
            task,
            1,
            "2026-08-08T00:00:00+00:00",
        )
        return {"reason": "reassigned_recipient_bootstrap_test"}

    monkeypatch.setattr(daemon_module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        scheduler_module,
        "configured_board_capacity_observation",
        lambda _board: (_host_capacity(), _provider_capacity(), PLAN_NOW),
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "run_once",
        bounded_run_once,
    )
    marker = supervisor_module.PLAN_BOUND_DAEMON_CHILD_MARKER
    helper_argv = command[command.index(marker) + 1 :]
    assert supervisor_module._run_plan_bound_daemon_child(helper_argv) == 0
    assert captured["decision"] is None
    claim = captured["claim"]
    assert isinstance(claim, dict)
    assert claim["plan_revision_cid"] == receipt.binding.revision_cid
    assert claim["compiled_claim_acquired_before_publish"] is True


def test_reassignment_rejects_dangling_hardlink_and_swapped_claim_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _board, _receipt, donor, recipient, process = (
        _fenced_plan_children(tmp_path)
    )
    claim_probe = object.__new__(daemon_module.PortalImplementationDaemon)
    claim_probe.repo_root = repo
    claim_path = daemon_module.PortalImplementationDaemon._implementation_task_claim_path(
        claim_probe,
        donor.task_ids[0],
        canonical_task_cid=donor.task_cids[0],
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)

    def reassign():
        return multi_runner_module.reassign_fenced_plan_bound_child(
            donor=donor,
            recipient=recipient,
            donor_process=process,
            repo_root=repo,
        )

    claim_path.symlink_to(claim_path.with_name("missing-claim-target"))
    with pytest.raises(ExecutionClaimConflictError, match="unsafe"):
        reassign()
    claim_path.unlink()

    hardlink_source = claim_path.with_name("hardlink-source.json")
    _write(hardlink_source, json.dumps({"kind": "stale"}) + "\n")
    os.link(hardlink_source, claim_path)
    with pytest.raises(ExecutionClaimConflictError, match="unsafe"):
        reassign()
    claim_path.unlink()
    hardlink_source.unlink()

    _write(claim_path, json.dumps({"kind": "before-swap"}) + "\n")
    replacement = claim_path.with_name("replacement-claim.json")
    _write(replacement, json.dumps({"kind": "after-swap"}) + "\n")
    original_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if Path(path) == claim_path and not swapped:
            swapped = True
            claim_path.unlink()
            replacement.replace(claim_path)
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(multi_runner_module.os, "open", swapping_open)
    with pytest.raises(ExecutionClaimConflictError, match="unsafe"):
        reassign()
    assert swapped is True

    store_path = Path(donor.plan_revision_store_path)
    if not store_path.is_absolute():
        store_path = repo / store_path
    assert ProductionParallelPlanAdapter(
        PlanRevisionStore(store_path.resolve())
    ).load_slice_reassignment(
        revision_cid=donor.revision_cid,
        slice_id=donor.slice_id,
    ) is None


def test_reassignment_rejects_prior_claim_and_consumed_provider_attempt(
    tmp_path: Path,
) -> None:
    repo, _board, _receipt, donor, recipient, process = (
        _fenced_plan_children(tmp_path)
    )
    claim_probe = object.__new__(daemon_module.PortalImplementationDaemon)
    claim_probe.repo_root = repo
    claim_path = daemon_module.PortalImplementationDaemon._implementation_task_claim_path(
        claim_probe,
        donor.task_ids[0],
        canonical_task_cid=donor.task_cids[0],
    )
    _write(
        claim_path,
        json.dumps(
            {
                "kind": daemon_module.IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
                "pid": 999_999_999,
                "repo_root": str(repo),
                "task_id": donor.task_ids[0],
                "canonical_task_cid": donor.task_cids[0],
            },
            sort_keys=True,
        )
        + "\n",
    )

    def reassign():
        return multi_runner_module.reassign_fenced_plan_bound_child(
            donor=donor,
            recipient=recipient,
            donor_process=process,
            repo_root=repo,
        )

    with pytest.raises(ExecutionClaimConflictError, match="already published"):
        reassign()
    claim_path.unlink()

    state_dir = Path(donor.state_dir)
    if not state_dir.is_absolute():
        state_dir = repo / state_dir
    state_path = state_dir / f"{donor.state_prefix}_task_state.json"
    _write(
        state_path,
        (
            '{"active_attempt":0,"implementation_in_progress":false,'
            f'"implementation_attempts":{{"{donor.task_ids[0]}":1}},'
            '"implementation_attempts":{},'
            '"implementation_attempts_by_cid":{}}\n'
        ),
    )
    with pytest.raises(
        ExecutionClaimConflictError,
        match="prove canonical donor attempt state pristine",
    ):
        reassign()

    _write(
        state_path,
        json.dumps(
            {
                "active_attempt": 0.1,
                "implementation_in_progress": False,
                "implementation_attempts": {},
                "implementation_attempts_by_cid": {},
            },
            sort_keys=True,
        )
        + "\n",
    )
    with pytest.raises(ExecutionClaimConflictError, match="active attempt is malformed"):
        reassign()

    _write(
        state_path,
        json.dumps(
            {
                "active_attempt": 0,
                "implementation_in_progress": "false",
                "implementation_attempts": {},
                "implementation_attempts_by_cid": {},
            },
            sort_keys=True,
        )
        + "\n",
    )
    with pytest.raises(
        ExecutionClaimConflictError,
        match="implementation-in-progress flag is malformed",
    ):
        reassign()

    _write(
        state_path,
        json.dumps(
            {
                # Canonical crash recovery may release an unfinished attempt
                # ordinal back to zero.  These implementation-start markers
                # are saved before provider dispatch, so any provider-started
                # attempt necessarily retains this no-replay evidence.
                "active_task_id": "",
                "active_task_cid": "",
                "active_attempt": 0,
                "implementation_in_progress": False,
                "implementation_attempts": {},
                "implementation_attempts_by_cid": {},
                "last_implementation_task_id": donor.task_ids[0],
                "last_implementation_task_cid": donor.task_cids[0],
                "last_implementation_started_at": (
                    "2026-08-08T00:00:00+00:00"
                ),
            },
            sort_keys=True,
        )
        + "\n",
    )
    with pytest.raises(ExecutionClaimConflictError, match="consumed|active"):
        reassign()

    store_path = Path(donor.plan_revision_store_path)
    if not store_path.is_absolute():
        store_path = repo / store_path
    assert ProductionParallelPlanAdapter(
        PlanRevisionStore(store_path.resolve())
    ).load_slice_reassignment(
        revision_cid=donor.revision_cid,
        slice_id=donor.slice_id,
    ) is None


def test_plan_bound_child_record_rejects_arbitrary_entry_and_authority_smuggling(
    tmp_path: Path,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path, (_task_block("TEST-A"),)
    )
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(lanes=1),
        provider_capacity_snapshots=_provider_capacity(lanes=1),
        task_state_snapshots=(),
    )
    assert receipt is not None
    launch = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        stamp="20260808T030303Z",
        parallelism_receipt=receipt,
    )
    argv = list(launch["argv"])
    records = [
        argv[index + 1]
        for index, token in enumerate(argv[:-1])
        if token == "--implementation-plan-bound-track"
    ]
    children = tuple(
        multi_runner_module.PlanBoundSupervisorChild.from_cli_record(record)
        for record in records
    )
    assert len(children) == 1
    child = children[0]
    with pytest.raises(ValueError, match="accepted entry"):
        replace(child, script_path="scripts/arbitrary_child.py")
    with pytest.raises(ValueError, match="safe relative path"):
        replace(child, state_dir="../escaped-state")

    payload = child.to_dict()
    payload["script_path"] = "scripts/arbitrary_child.py"
    with pytest.raises(ValueError, match="accepted entry"):
        multi_runner_module.PlanBoundSupervisorChild.from_cli_record(
            json.dumps(payload)
        )
    duplicate = child.cli_record()[:-1] + ',"lane_id":"other"}'
    with pytest.raises(ValueError, match="duplicate key"):
        multi_runner_module.PlanBoundSupervisorChild.from_cli_record(duplicate)
    payload = child.to_dict()
    payload["extra"] = "authority-smuggling"
    with pytest.raises(ValueError, match="fields are not exact"):
        multi_runner_module.PlanBoundSupervisorChild.from_cli_record(
            json.dumps(payload)
        )


def test_plan_bound_supervisor_paths_reject_links_before_any_outside_write(
    tmp_path: Path,
) -> None:
    repo, config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    state_root = repo / board.runtime_paths["state"]
    state_root.mkdir(parents=True)
    outside = tmp_path / "outside-runtime-authority"

    def construct(state_dir: Path, store_path: Path) -> None:
        supervisor_module.PortalSupervisorConfig(
            todo_path=repo / board.taskboard_path,
            state_path=state_dir / "lane_task_state.json",
            strategy_path=state_dir / "lane_strategy.json",
            events_path=state_dir / "lane_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            scheduler_config_path=config_path,
            plan_bound_dispatch=True,
            plan_revision_store_path=store_path,
            plan_bound_accepted_tree_root=repo,
        )

    lane_dir = state_root / "lane-0"
    lane_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(
        supervisor_module.PlanBoundDispatchError,
        match="symbolic link",
    ):
        construct(lane_dir, state_root / "plan-revision-store")
    assert not outside.exists()
    lane_dir.unlink()

    lane_dir.mkdir()
    store_path = state_root / "plan-revision-store"
    store_path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(
        supervisor_module.PlanBoundDispatchError,
        match="symbolic link",
    ):
        construct(lane_dir, store_path)
    assert not outside.exists()


def test_plan_store_authority_rejects_duplicate_hardlink_and_swap_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path, (_task_block("TEST-A"),)
    )
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(lanes=1),
        provider_capacity_snapshots=_provider_capacity(lanes=1),
        task_state_snapshots=(),
    )
    assert receipt is not None
    store = PlanRevisionStore(
        board.path(board.runtime_paths["state"]) / "plan-revision-store"
    )
    adapter = ProductionParallelPlanAdapter(store)
    execution_slice = receipt.slice_manifest.slices[0]

    active_bytes = store.active_path.read_bytes()
    store.active_path.write_bytes(
        active_bytes.rstrip()[:-1] + b',"revision_cid":"foreign"}\n'
    )
    with pytest.raises(ExecutionPlanError, match="duplicate JSON key"):
        load_plan_revision_store_binding(store)
    store.active_path.write_bytes(active_bytes)

    active_alias = tmp_path / "active-hardlink.json"
    os.link(store.active_path, active_alias)
    try:
        with pytest.raises(ExecutionPlanError, match="single-link"):
            load_plan_revision_store_binding(store)
    finally:
        active_alias.unlink()

    os.chmod(store.active_path, 0o644)
    try:
        with pytest.raises(ExecutionPlanError, match="exactly 0600"):
            load_plan_revision_store_binding(store)
    finally:
        os.chmod(store.active_path, 0o600)

    original_lstat = execution_plan_module.os.lstat

    def foreign_owner_lstat(path: os.PathLike[str] | str):
        observed = original_lstat(path)
        if Path(path) == store.active_path:
            raw = list(observed)
            raw[4] = os.geteuid() + 1
            return os.stat_result(raw)
        return observed

    with monkeypatch.context() as owner_patch:
        owner_patch.setattr(
            execution_plan_module.os, "lstat", foreign_owner_lstat
        )
        with pytest.raises(ExecutionPlanError, match="effective user"):
            load_plan_revision_store_binding(store)

    continuation_paths = tuple(store.continuations_dir.glob("*.json"))
    assert len(continuation_paths) == 1
    continuation_path = continuation_paths[0]
    continuation_bytes = continuation_path.read_bytes()
    continuation_record = json.loads(continuation_bytes)
    continuation_record["updated_at_ns"] += 1
    continuation_path.write_text(
        json.dumps(continuation_record, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(ExecutionPlanError, match="identity is invalid"):
            execution_plan_module._secure_store_continuation(
                store,
                continuation_record["idempotency_key"],
            )
    finally:
        continuation_path.write_bytes(continuation_bytes)

    manifest_path = store.cas_dir / receipt.slice_manifest_cid
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(
        manifest_bytes.rstrip()[:-1] + b',"payload":{}}\n'
    )
    with pytest.raises(ExecutionPlanError, match="duplicate JSON key"):
        adapter.validate_slice_owner(
            revision_cid=receipt.binding.revision_cid,
            slice_manifest_cid=receipt.slice_manifest_cid,
            slice_id=execution_slice.slice_id,
            lane_id=execution_slice.lane_id,
        )
    manifest_path.write_bytes(manifest_bytes)

    manifest_alias = tmp_path / "manifest-hardlink.json"
    os.link(manifest_path, manifest_alias)
    try:
        with pytest.raises(ExecutionPlanError, match="single-link"):
            adapter.validate_slice_owner(
                revision_cid=receipt.binding.revision_cid,
                slice_manifest_cid=receipt.slice_manifest_cid,
                slice_id=execution_slice.slice_id,
                lane_id=execution_slice.lane_id,
            )
    finally:
        manifest_alias.unlink()

    os.chmod(manifest_path, 0o660)
    try:
        with pytest.raises(ExecutionPlanError, match="exactly 0600"):
            adapter.validate_slice_owner(
                revision_cid=receipt.binding.revision_cid,
                slice_manifest_cid=receipt.slice_manifest_cid,
                slice_id=execution_slice.slice_id,
                lane_id=execution_slice.lane_id,
            )
    finally:
        os.chmod(manifest_path, 0o600)

    original_fstat = execution_plan_module.os.fstat
    manifest_inode = manifest_path.stat().st_ino
    manifest_fstat_calls = 0

    def swap_after_read(descriptor: int):
        nonlocal manifest_fstat_calls
        observed = original_fstat(descriptor)
        if observed.st_ino == manifest_inode:
            manifest_fstat_calls += 1
        if observed.st_ino == manifest_inode and manifest_fstat_calls == 2:
            replacement = manifest_path.with_name("manifest-swap")
            replacement.write_bytes(manifest_bytes)
            os.chmod(replacement, 0o600)
            os.replace(replacement, manifest_path)
        return observed

    monkeypatch.setattr(execution_plan_module.os, "fstat", swap_after_read)
    with pytest.raises(ExecutionPlanError, match="pathname changed"):
        adapter.validate_slice_owner(
            revision_cid=receipt.binding.revision_cid,
            slice_manifest_cid=receipt.slice_manifest_cid,
            slice_id=execution_slice.slice_id,
            lane_id=execution_slice.lane_id,
        )


def test_actual_scope_drift_is_replanned_as_new_serialized_revision(
    tmp_path: Path,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"), _task_block("TEST-B")),
    )

    def materialize():
        result = materialize_configured_board_execution_plan(
            board,
            now_ms=PLAN_NOW,
            host_capacity_snapshot=_host_capacity(),
            provider_capacity_snapshots=_provider_capacity(),
            task_state_snapshots=(),
        )
        assert result is not None
        return result

    original = materialize()
    assert len(original.slice_manifest.slices) == 2
    _write(
        repo / "docs/tasks.md",
        "# Tasks\n\n"
        + "\n".join(
            (
                _task_block("TEST-A", output="src/observed-shared.py"),
                _task_block("TEST-B", output="src/observed-shared.py"),
            )
        ),
    )
    _git(repo, "add", "docs/tasks.md")
    _git(repo, "commit", "-m", "replan observed overlapping scope")
    steered = materialize()
    assert steered.binding.semantic_revision == original.binding.semantic_revision + 1
    assert steered.binding.revision_cid != original.binding.revision_cid
    assert len(steered.slice_manifest.slices) == 1


def test_v3_coordinator_replans_second_wave_from_new_head_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (
            _task_block("TEST-A"),
            _task_block("TEST-B", depends_on=("TEST-A",)),
        ),
    )
    original_materialize = materialize_configured_board_execution_plan

    def materialize(current_board):
        return original_materialize(
            current_board,
            now_ms=PLAN_NOW,
            host_capacity_snapshot=_host_capacity(),
            provider_capacity_snapshots=_provider_capacity(),
            task_state_snapshots=(),
        )

    monkeypatch.setattr(
        scheduler_module,
        "materialize_configured_board_execution_plan",
        materialize,
    )
    launched: list[dict[str, object]] = []

    def fake_multi_supervisor_main(argv: list[str]) -> int:
        records = [
            json.loads(argv[index + 1])
            for index, token in enumerate(argv[:-1])
            if token == "--implementation-plan-bound-track"
        ]
        assert len(records) == 1
        launched.append(records[0])
        if len(launched) == 1:
            return multi_runner_module.PLAN_BOUND_REPLAN_RETURN_CODE
        if len(launched) == 2:
            blocks = (
                _task_block("TEST-A", status="completed"),
                _task_block("TEST-B", depends_on=("TEST-A",)),
            )
        else:
            blocks = (
                _task_block("TEST-A", status="completed"),
                _task_block(
                    "TEST-B", status="completed", depends_on=("TEST-A",)
                ),
            )
        _write(repo / "docs/tasks.md", "# Tasks\n\n" + "\n".join(blocks))
        _git(repo, "add", "docs/tasks.md")
        _git(repo, "commit", "-m", f"complete coordinator wave {len(launched)}")
        return 0

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner.main",
        fake_multi_supervisor_main,
    )
    assert scheduler_module._run_plan_bound_coordinator(
        board,
        implement=True,
        duration_seconds=30,
    ) == 0
    assert [item["task_ids"] for item in launched] == [
        ["TEST-A"],
        ["TEST-A"],
        ["TEST-B"],
    ]
    assert launched[0]["source_head"] == launched[1]["source_head"]
    assert launched[0]["revision_cid"] == launched[1]["revision_cid"]
    assert launched[1]["source_head"] != launched[2]["source_head"]
    assert launched[1]["revision_cid"] != launched[2]["revision_cid"]


def test_preflight_accepts_exact_committed_binding_then_rejects_drift(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    board = load_configured_board(config_path, repo_root=repo)

    report = preflight_configured_board(board)
    assert report["valid"] is True, report["errors"]

    _write(repo / "docs/plan.md", "dirty plan\n")
    dirty_report = preflight_configured_board(board)
    assert dirty_report["valid"] is False
    assert any(
        error.startswith("checkout_clean:")
        for error in dirty_report["errors"]
    )

    _write(repo / "docs/plan.md", "plan\n")
    assert not _git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    _git(repo, "submodule", "deinit", "-f", "--", "dependency")
    uninitialized_report = preflight_configured_board(board)
    assert uninitialized_report["valid"] is False
    submodule_check = next(
        check
        for check in uninitialized_report["checks"]
        if check["name"] == "configured_submodules"
    )
    assert submodule_check["passed"] is False
    assert submodule_check["detail"][0]["exact_worktree"] is False


def test_preflight_accepts_only_descendant_submodule_progress(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    child_worktree = repo / "dependency"
    child_source = Path(
        _git(child_worktree, "remote", "get-url", "origin").stdout.strip()
    )

    _write(child_source / "dependency.txt", "advanced dependency\n")
    _git(child_source, "add", "dependency.txt")
    _git(child_source, "commit", "-m", "advance dependency")
    advanced_revision = _git(
        child_source,
        "rev-parse",
        "HEAD",
    ).stdout.strip()
    _git(child_worktree, "fetch", "origin")
    _git(child_worktree, "checkout", advanced_revision)
    _git(repo, "add", "dependency")
    _git(repo, "commit", "-m", "record dependency progress")

    board = load_configured_board(config_path, repo_root=repo)
    advanced_report = preflight_configured_board(board)
    assert advanced_report["valid"] is True, advanced_report["errors"]
    advanced_check = next(
        check
        for check in advanced_report["checks"]
        if check["name"] == "configured_submodules"
    )
    assert advanced_check["detail"][0][
        "planning_revision_is_ancestor"
    ] is True

    _git(child_source, "checkout", "--orphan", "divergent")
    _write(child_source / "dependency.txt", "divergent dependency\n")
    _git(child_source, "add", "dependency.txt")
    _git(child_source, "commit", "-m", "diverge dependency")
    divergent_revision = _git(
        child_source,
        "rev-parse",
        "HEAD",
    ).stdout.strip()
    _git(child_worktree, "fetch", "origin", "divergent")
    _git(child_worktree, "checkout", divergent_revision)
    _git(repo, "add", "dependency")
    _git(repo, "commit", "-m", "record divergent dependency")

    divergent_report = preflight_configured_board(board)
    assert divergent_report["valid"] is False
    divergent_check = next(
        check
        for check in divergent_report["checks"]
        if check["name"] == "configured_submodules"
    )
    assert divergent_check["passed"] is False
    assert divergent_check["detail"][0][
        "planning_revision_is_ancestor"
    ] is False


def test_preflight_rejects_missing_submodule_planning_revision(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    del payload["source_binding"]["dependency_planning_revision"]
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "remove dependency planning revision")

    board = load_configured_board(config_path, repo_root=repo)
    report = preflight_configured_board(board)
    submodule_check = next(
        check
        for check in report["checks"]
        if check["name"] == "configured_submodules"
    )

    assert report["valid"] is False
    assert submodule_check["passed"] is False
    assert submodule_check["detail"][0]["planning_revision"] == ""
    assert submodule_check["detail"][0][
        "planning_revision_is_ancestor"
    ] is False


def test_preflight_rejects_submodule_head_gitlink_mismatch(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    child_worktree = repo / "dependency"
    child_source = Path(
        _git(child_worktree, "remote", "get-url", "origin").stdout.strip()
    )
    _write(child_source / "dependency.txt", "unrecorded advance\n")
    _git(child_source, "add", "dependency.txt")
    _git(child_source, "commit", "-m", "unrecorded dependency advance")
    revision = _git(child_source, "rev-parse", "HEAD").stdout.strip()
    _git(child_worktree, "fetch", "origin")
    _git(child_worktree, "checkout", revision)

    board = load_configured_board(config_path, repo_root=repo)
    report = preflight_configured_board(board)
    submodule = next(
        check
        for check in report["checks"]
        if check["name"] == "configured_submodules"
    )["detail"][0]

    assert report["valid"] is False
    assert submodule["valid"] is False
    assert submodule["head"] != submodule["gitlink"]


def test_preflight_rejects_dirty_submodule_worktree(tmp_path: Path) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    _write(repo / "dependency" / "dependency.txt", "dirty dependency\n")

    board = load_configured_board(config_path, repo_root=repo)
    report = preflight_configured_board(board)
    submodule = next(
        check
        for check in report["checks"]
        if check["name"] == "configured_submodules"
    )["detail"][0]

    assert report["valid"] is False
    assert submodule["valid"] is False
    assert submodule["dirty"]


def test_loader_rejects_runtime_path_escape(tmp_path: Path) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["runtime_paths"]["state"] = "../escaped-state"
    payload["protected_paths"][0] = "config/unsafe.json"
    unsafe = repo / "config/unsafe.json"
    _write(unsafe, json.dumps(payload))

    with pytest.raises(ConfiguredBoardError, match="unsafe relative path"):
        load_configured_board(unsafe, repo_root=repo)
