"""Tests for the sealed scheduler-config runtime adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
_TEST_PROCESSES: list[subprocess.Popen[Any]] = []


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
    for process in reversed(_TEST_PROCESSES):
        if process.poll() is None:
            try:
                process_group = os.getpgid(process.pid)
                if process_group == process.pid:
                    os.killpg(process_group, signal.SIGTERM)
                else:
                    process.terminate()
            except (OSError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    process_group = os.getpgid(process.pid)
                    if process_group == process.pid:
                        os.killpg(process_group, signal.SIGKILL)
                    else:
                        process.kill()
                except (OSError, ProcessLookupError):
                    pass
        try:
            process.wait(timeout=1)
        except (ChildProcessError, subprocess.TimeoutExpired):
            pass
    _TEST_PROCESSES.clear()


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _capture_test_process_identity(
    process: subprocess.Popen[Any],
    profile: Any,
    *,
    timeout_seconds: float = 2.0,
) -> Any:
    """Wait only for the synthetic child to finish exec'ing its fixed env."""

    adapter = multi_runner_module.LinuxProcessAdapter()
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return adapter._identity(process.pid, profile)
        except multi_runner_module.ProcessIdentityMismatch:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _spawn_test_process(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
    """Register a real test child before any fallible identity observation."""

    process = subprocess.Popen(*args, **kwargs)
    _TEST_PROCESSES.append(process)
    return process


def _test_lifecycle_token(tmp_path: Path, label: str) -> str:
    return hashlib.sha256(
        f"{tmp_path.resolve()}:{label}".encode("utf-8")
    ).hexdigest()[:16]


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
    # ASE3-030 owns the production manifest closure.  Include its already-
    # reviewed identity leaf in this moving fixture so the remaining red edge
    # is specifically the absent sealed native DuckDB dependency, not an
    # earlier Python-source import omission.
    relative_files.add("ipfs_accelerate_py/utils/cid_utils.py")
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
    lifecycle_token = _test_lifecycle_token(
        tmp_path,
        "plan-bound-reassignment",
    )
    profile = multi_runner_module.LifecycleProfile(
        target_id=f"supervisor-track:{donor.name}",
        run_id=f"test-plan-bound-reassignment-{lifecycle_token}",
        configuration_root=(
            f"test-plan-bound-reassignment-config-{lifecycle_token}"
        ),
        repository_root=str(repo.resolve()),
        state_root=str(state_root),
        run_root=str(
            state_root / "lifecycle-runs" / f"{donor.name}-{lifecycle_token}"
        ),
        argv=tuple(command),
        cwd=str(repo.resolve()),
    )
    process = _spawn_test_process(
        command,
        cwd=repo,
        env=profile.launch_environment(0),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    process_identity = _capture_test_process_identity(process, profile)
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


def _publish_test_no_change_disposition(
    *,
    repo: Path,
    child: multi_runner_module.PlanBoundSupervisorChild,
) -> tuple[str, execution_plan_module.PlanBoundExecutionLease]:
    """Move one synthetic reserved lease to a restartable no-change result."""

    store = PlanRevisionStore(repo / child.plan_revision_store_path)
    adapter = ProductionParallelPlanAdapter(store)
    with store._thread_lock:
        with store._guard():
            current = execution_plan_module._load_plan_bound_execution_lease_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
            )
            assert current is not None
            current_cid, reserved = current
            task_id = child.task_ids[0]
            task_cid = child.task_cids[0]
            assignment = reserved.assignment_for(task_id, task_cid)
            claimed = replace(
                reserved,
                generation=reserved.generation + 1,
                phase="claimed",
                prior_execution_lease_cid=current_cid,
                active_task_id=task_id,
                active_task_cid=task_cid,
                daemon_process_birth={"pid": os.getpid()},
                canonical_claim_path=str(repo / "test-recovery-claim.json"),
                canonical_claim_cid="sha256:" + "1" * 64,
                canonical_claim_lease_id=str(assignment["lease_id"]),
            )
            current_cid = (
                execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    claimed,
                    expected_current_cid=current_cid,
                )
            )
            workspace_path = repo / "test-recovery-worktree"
            prepared = replace(
                claimed,
                generation=claimed.generation + 1,
                phase="workspace_prepared",
                prior_execution_lease_cid=current_cid,
                workspace_lifecycle_path=str(
                    repo / "test-recovery-worktree-lifecycle.json"
                ),
                workspace_lifecycle_cid="sha256:" + "2" * 64,
                workspace_record_id="workspace-record:test-recovery",
                workspace_path=str(workspace_path),
                workspace_lease_id="workspace-lease:test-recovery",
                workspace_fence=1,
            )
            current_cid = (
                execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    prepared,
                    expected_current_cid=current_cid,
                )
            )
            provider_ready = replace(
                prepared,
                generation=prepared.generation + 1,
                phase="provider_ready",
                prior_execution_lease_cid=current_cid,
                provider_ready=True,
            )
            current_cid = (
                execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    provider_ready,
                    expected_current_cid=current_cid,
                )
            )
            baseline_ref = child.source_head
            enqueue_fields = {
                "branch_name": "implementation/test-recovery",
                "task_id": task_id,
                "priority": "normal",
                "lane_id": child.lane_id,
                "attempt": 1,
                "metadata": {
                    "baseline_ref": baseline_ref,
                    "implementation_commit": baseline_ref,
                },
                "commit_sha": baseline_ref,
                "canonical_task_id": task_cid,
                "canonical_task_key": "task/v1/test-recovery",
                "canonical_task_cid": task_cid,
                "target_repository_id": str(repo.resolve()),
                "target_branch": "main",
            }
            proposal_handoff = {
                "schema": execution_plan_module.PLAN_BOUND_PROPOSAL_HANDOFF_SCHEMA,
                "revision_cid": provider_ready.revision_cid,
                "plan_root_cid": provider_ready.plan_root_cid,
                "execution_plan_cid": provider_ready.execution_plan_cid,
                "capacity_snapshot_id": provider_ready.capacity_snapshot_id,
                "slice_manifest_cid": provider_ready.slice_manifest_cid,
                "slice_id": provider_ready.slice_id,
                "lane_id": provider_ready.lane_id,
                "reassignment_cid": provider_ready.reassignment_cid,
                "task_id": task_id,
                "task_cid": task_cid,
                "source_execution_lease_cid": current_cid,
                "process_birth_cid": provider_ready.process_birth_cid,
                "canonical_claim_cid": provider_ready.canonical_claim_cid,
                "canonical_claim_lease_id": (
                    provider_ready.canonical_claim_lease_id
                ),
                "workspace_lifecycle_cid": (
                    provider_ready.workspace_lifecycle_cid
                ),
                "workspace_record_id": provider_ready.workspace_record_id,
                "workspace_path": provider_ready.workspace_path,
                "workspace_lease_id": provider_ready.workspace_lease_id,
                "workspace_fence": provider_ready.workspace_fence,
                "attempt": 1,
                "branch_name": "implementation/test-recovery",
                "baseline_ref": baseline_ref,
                "implementation_commit": baseline_ref,
                "actual_changed_paths": [],
                "outcome": "no_change",
                "enqueue_fields": enqueue_fields,
                "enqueue_fields_cid": execution_plan_module.content_identity(
                    enqueue_fields
                ),
                "created_at_ms": int(time.time() * 1000),
            }
            proposal_handoff_cid = store.put_cas(proposal_handoff)
            proposal_ready = replace(
                provider_ready,
                generation=provider_ready.generation + 1,
                phase="proposal_ready",
                prior_execution_lease_cid=current_cid,
                proposal_handoff_cid=proposal_handoff_cid,
            )
            proposal_cid = (
                execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    proposal_ready,
                    expected_current_cid=current_cid,
                )
            )
            disposition = execution_plan_module.PlanBoundProposalDisposition(
                revision_cid=child.revision_cid,
                plan_root_cid=child.plan_root_cid,
                execution_plan_cid=child.execution_plan_cid,
                capacity_snapshot_id=child.capacity_snapshot_id,
                slice_manifest_cid=child.slice_manifest_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
                reassignment_cid=child.reassignment_cid,
                task_id=task_id,
                task_cid=task_cid,
                execution_lease_cid=proposal_cid,
                process_birth_cid=proposal_ready.process_birth_cid,
                proposal_id="",
                proposal_receipt_id="",
                outcome="no_change",
                reason_codes=(),
                actual_changed_paths=(),
                baseline_ref=baseline_ref,
                implementation_commit=baseline_ref,
            )
    adapter.publish_proposal_disposition(disposition)
    return proposal_cid, proposal_ready


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
        "plan-bound-gate-test-"
        + _test_lifecycle_token(tmp_path, "identity-capture-failure")
    )
    runtime_root = REPO_ROOT / runtime_relative
    lane_relative = runtime_relative / "lane-0"
    lane_root = REPO_ROOT / lane_relative
    supervisor_pid = lane_root / "supervisor.pid"
    store_relative = runtime_relative / "plan-revision-store"
    plan_args = (
        "--state-dir", str(lane_relative),
        "--state-prefix", "identity_capture_failure",
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
    hostile_environment = {
        "LD_PRELOAD": str(tmp_path / "hostile-preload.so"),
        "LD_LIBRARY_PATH": str(tmp_path / "hostile-library-path"),
        "LD_AUDIT": str(tmp_path / "hostile-audit.so"),
        "GLIBC_TUNABLES": "glibc.malloc.check=3",
        "PYTHONPATH": str(tmp_path / "hostile-python-path"),
        "PYTHONSTARTUP": str(tmp_path / "hostile-startup.py"),
        "ASE3_UNRELATED_AMBIENT": "must-not-cross-plan-bound-exec",
    }
    for name, value in hostile_environment.items():
        monkeypatch.setenv(name, value)
    observed_environment: dict[str, str] = {}

    def fail_identity(_self, _pid, _profile):
        # Popen may return while the child is still between fork and exec, at
        # which point procfs can transiently expose an empty environment.  Do
        # not mistake that observation race for the environment accepted by
        # the gated interpreter.
        deadline = time.monotonic() + 2.0
        while True:
            raw_environment = Path(f"/proc/{_pid}/environ").read_bytes()
            current_environment = dict(
                item.decode("utf-8").split("=", 1)
                for item in raw_environment.split(b"\0")
                if item and b"=" in item
            )
            if current_environment.get("PATH") == "/usr/bin:/bin":
                observed_environment.update(current_environment)
                break
            if time.monotonic() >= deadline:
                observed_environment.update(current_environment)
                break
            time.sleep(0.01)
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
        assert observed_environment["PATH"] == "/usr/bin:/bin"
        assert not set(hostile_environment).intersection(observed_environment)
        lifecycle_environment = {
            "IPFS_ACCELERATE_LIFECYCLE_RUN_ID",
            "IPFS_ACCELERATE_LIFECYCLE_PROFILE_ID",
            "IPFS_ACCELERATE_LIFECYCLE_TARGET_ID",
            "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT",
            "IPFS_ACCELERATE_LIFECYCLE_STATE_ROOT",
            "IPFS_ACCELERATE_LIFECYCLE_RUN_ROOT",
            "IPFS_ACCELERATE_LIFECYCLE_FENCING_EPOCH",
            "IPFS_ACCELERATE_LIFECYCLE_CONFIGURATION_ROOT",
        }
        ambient_environment = {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}
        route_environment = {
            *multi_runner_module.ORDERED_IMPLEMENTATION_PROVIDER_ROUTE,
            *multi_runner_module._ROUTE_AUTHORIZATION_ENV_NAMES,
        }
        assert lifecycle_environment.issubset(observed_environment)
        assert set(observed_environment).issubset(
            lifecycle_environment | ambient_environment | route_environment
        )
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


def test_legacy_track_in_mixed_runner_inherits_no_sealed_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "legacy-supervisor.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    track = multi_runner_module.SupervisorTrack(
        name="legacy-track",
        script_path=script,
        log_path=tmp_path / "legacy.log",
        supervisor_pid_path=tmp_path / "legacy.pid",
        daemon_pid_path=tmp_path / "legacy-daemon.pid",
        supervisor_status_path=tmp_path / "legacy-status.json",
    )
    captured: dict[str, object] = {}

    def capture_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(multi_runner_module.subprocess, "Popen", capture_popen)
    inherited_read, inherited_write = os.pipe()
    try:
        process = multi_runner_module.start_track(
            track,
            repo_root=tmp_path,
            common_args=(),
            python_executable=sys.executable,
            accepted_control_plane_descriptor=inherited_read,
            output=lambda _message: None,
        )
    finally:
        os.close(inherited_read)
        os.close(inherited_write)
    assert process.pid == os.getpid()
    assert captured["pass_fds"] == ()
    assert captured["command"] == [sys.executable, str(script)]


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ASE3-031 must require an independently accepted sealed native "
        "DuckDB launch before importing the implementation supervisor"
    ),
)
def test_sealed_bootstrap_denies_missing_native_dependency_pin(
    tmp_path: Path,
) -> None:
    """The production bootstrap must deny a source-only implementation lane."""

    source_head = _git(REPO_ROOT, "rev-parse", "HEAD").stdout.strip()
    source_tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}").stdout.strip()
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=source_head,
        source_tree=source_tree,
    )
    command = multi_runner_module.build_sealed_control_plane_module_command(
        python_executable=sys.executable,
        pin=control_plane_pin,
        descriptor=control_plane_launch.descriptor,
        module_name=(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_supervisor"
        ),
        argv=("--help",),
    )
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "")},
        pass_fds=(control_plane_launch.descriptor,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 78


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
    assert "--no-retry-budget-guardrail" not in common
    assert "--no-dependency-guardrail" not in common
    assert "--no-reconciliation-guardrail" not in common
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


def test_disabled_guardrails_project_to_existing_supervisor_flags(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "retry_budget_guardrail_enabled": False,
            "dependency_guardrail_enabled": False,
            "reconciliation_guardrail_enabled": False,
        }
    )
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    board = load_configured_board(config_path, repo_root=repo)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260804T000000Z",
    )
    common = _common_args(plan)

    assert common.count("--no-retry-budget-guardrail") == 1
    assert common.count("--no-dependency-guardrail") == 1
    assert common.count("--no-reconciliation-guardrail") == 1


def test_idle_lane_virgin_transfer_loads_and_propagates_to_supervisors(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["idle_lane_work_stealing"] = "virgin-transfer"
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    board = load_configured_board(config_path, repo_root=repo)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260812T000000Z",
    )
    common = _common_args(plan)

    assert board.strict_task_sharding is True
    assert board.idle_lane_work_stealing == "virgin-transfer"
    assert common.count("--idle-lane-work-stealing") == 1
    option = common.index("--idle-lane-work-stealing")
    assert common[option + 1] == "virgin-transfer"
    assert plan["effective_idle_lane_work_stealing"] == "virgin-transfer"


def test_idle_lane_virgin_transfer_requires_strict_sharding(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["strict_task_sharding"] = False
    payload["idle_lane_work_stealing"] = "virgin-transfer"
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        scheduler_module.ConfiguredBoardError,
        match="requires strict_task_sharding",
    ):
        load_configured_board(config_path, repo_root=repo)


def test_idle_lane_virgin_transfer_requires_multiple_lanes(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["max_lanes"] = 1
    payload["lanes"] = payload["lanes"][:1]
    payload["idle_lane_work_stealing"] = "virgin-transfer"
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        scheduler_module.ConfiguredBoardError,
        match="requires at least two lanes",
    ):
        load_configured_board(config_path, repo_root=repo)


@pytest.mark.parametrize(
    "field",
    (
        "retry_budget_guardrail_enabled",
        "dependency_guardrail_enabled",
        "reconciliation_guardrail_enabled",
    ),
)
def test_guardrail_policy_rejects_nonboolean_and_defaults_enabled(
    tmp_path: Path,
    field: str,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = "false"
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ConfiguredBoardError, match=field):
        load_configured_board(config_path, repo_root=repo)

    del payload[field]
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    board = load_configured_board(config_path, repo_root=repo)
    common = _common_args(
        configured_board_launch_plan(
            board,
            implement=True,
            detach=True,
            stamp="20260804T000000Z",
        )
    )
    assert f"--no-{field.removesuffix('_enabled').replace('_', '-')}" not in common


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


def test_v3_population_scopes_display_attempts_to_canonical_revision(
    tmp_path: Path,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    source_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    original = scheduler_module._configured_board_task_population(
        board,
        source_head=source_head,
        task_state_snapshots=(),
    )
    old_cid = str(original.all_records[0]["canonical_task_cid"])
    revised_taskboard = (
        "# Tasks\n\n"
        + _task_block("TEST-A").rstrip()
        + "\n- Semantic key: provider-effect-retry-revision@1\n"
    ).encode("utf-8")
    revised = scheduler_module._configured_board_task_population(
        board,
        source_head=source_head,
        taskboard_bytes=revised_taskboard,
        task_state_snapshots=(),
    )
    revised_cid = str(revised.all_records[0]["canonical_task_cid"])
    assert revised_cid != old_cid

    old_revision_state = {
        "implementation_attempts": {"TEST-A": 3},
        "implementation_attempts_by_cid": {old_cid: 3},
        "task_identities": {
            "TEST-A": {
                "display_task_id": "TEST-A",
                "canonical_task_cid": old_cid,
            }
        },
    }
    same_revision = scheduler_module._configured_board_task_population(
        board,
        source_head=source_head,
        task_state_snapshots=(old_revision_state,),
    )
    assert same_revision.attempt_limited_task_ids == ("TEST-A",)
    assert same_revision.ready_records == ()

    fresh_revision = scheduler_module._configured_board_task_population(
        board,
        source_head=source_head,
        taskboard_bytes=revised_taskboard,
        task_state_snapshots=(old_revision_state,),
    )
    assert fresh_revision.attempt_limited_task_ids == ()
    assert tuple(item["task_id"] for item in fresh_revision.ready_records) == (
        "TEST-A",
    )
    assert fresh_revision.ready_records[0]["canonical_task_cid"] == revised_cid

    legacy_state = {
        "implementation_attempts": {"TEST-A": 3},
        "implementation_attempts_by_cid": {old_cid: 3},
    }
    legacy_revision = scheduler_module._configured_board_task_population(
        board,
        source_head=source_head,
        taskboard_bytes=revised_taskboard,
        task_state_snapshots=(legacy_state,),
    )
    assert legacy_revision.attempt_limited_task_ids == ("TEST-A",)
    assert legacy_revision.ready_records == ()


def test_v3_population_rejects_unbacked_mismatched_task_identity(
    tmp_path: Path,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    source_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    original = scheduler_module._configured_board_task_population(
        board,
        source_head=source_head,
        task_state_snapshots=(),
    )
    old_cid = str(original.all_records[0]["canonical_task_cid"])
    revised_taskboard = (
        "# Tasks\n\n"
        + _task_block("TEST-A").rstrip()
        + "\n- Semantic key: provider-effect-retry-revision@1\n"
    ).encode("utf-8")

    with pytest.raises(
        ConfiguredBoardError,
        match="mismatched task identity.*canonical attempt ledger",
    ):
        scheduler_module._configured_board_task_population(
            board,
            source_head=source_head,
            taskboard_bytes=revised_taskboard,
            task_state_snapshots=(
                {
                    "implementation_attempts": {"TEST-A": 3},
                    "implementation_attempts_by_cid": {old_cid: 2},
                    "task_identities": {
                        "TEST-A": {
                            "display_task_id": "TEST-A",
                            "canonical_task_cid": old_cid,
                        }
                    },
                },
                # A different lane's ledger cannot authenticate this lane's
                # display-ID-to-revision association.
                {"implementation_attempts_by_cid": {old_cid: 3}},
            ),
        )


@pytest.mark.parametrize(
    "task_identities",
    (
        [],
        {"TEST-A": []},
        {"TEST-A": {"canonical_task_cid": 7}},
        {
            "TEST-A": {
                "display_task_id": "TEST-B",
                "canonical_task_cid": "task-cid:test-a",
            }
        },
    ),
)
def test_v3_population_rejects_malformed_task_identity_projection(
    tmp_path: Path,
    task_identities: object,
) -> None:
    repo, _config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"),),
    )
    with pytest.raises(ConfiguredBoardError, match=r"task[_ ]identit"):
        scheduler_module._configured_board_task_population(
            board,
            source_head=_git(repo, "rev-parse", "HEAD").stdout.strip(),
            task_state_snapshots=(
                {
                    "implementation_attempts": {"TEST-A": 1},
                    "implementation_attempts_by_cid": {},
                    "task_identities": task_identities,
                },
            ),
        )


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
    monkeypatch: pytest.MonkeyPatch,
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

    trusted_now = int(time.time() * 1000)
    monkeypatch.setattr(
        scheduler_module.time,
        "time",
        lambda: trusted_now / 1000,
    )
    future_observations = _provider_capacity(
        lanes=1,
        observed_at_ms=trusted_now + 1,
    )
    observed_host, observed_providers, observed_now = (
        scheduler_module.configured_board_capacity_observation(
            board,
            host_capacity_snapshot={
                **_host_capacity(lanes=1),
                "observed_at_ms": trusted_now,
            },
            provider_capacity_snapshots=future_observations,
        )
    )
    assert observed_host["observed_at_ms"] == trusted_now
    assert observed_providers == future_observations
    assert observed_now == trusted_now
    future_projection, _route = (
        scheduler_module.configured_board_route_capacity_projection(
            board,
            provider_capacity_snapshots=observed_providers,
            now_ms=observed_now,
        )
    )
    assert future_projection["schedulable"] is False
    with pytest.raises(ExecutionPlanError):
        materialize_configured_board_execution_plan(
            board,
            host_capacity_snapshot={
                **_host_capacity(lanes=1),
                "observed_at_ms": trusted_now,
            },
            provider_capacity_snapshots=future_observations,
            task_state_snapshots=(),
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
    "bridge_scenario",
    (
        "normal",
        "scope_drift",
        "no_change",
        "no_change_guard_denied",
        "tamper",
        "mode_tamper",
        "effect_submodule_symlink",
        "effect_submodule_non_git",
        "effect_submodule_escape",
    ),
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
    if bridge_scenario in {"no_change", "no_change_guard_denied"}:
        _write(repo / "src/test-a.py", "VALUE = 'already-present'\n")
        _git(repo, "add", "src/test-a.py")
        _git(repo, "commit", "-m", "seed already-satisfied declared output")
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
        worktree_submodule_paths=board.worktree_submodule_paths,
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
        workspace.parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "worktree", "add", "--detach", str(workspace), "HEAD")
        if bridge_scenario in {"no_change", "no_change_guard_denied"}:
            _git(
                workspace,
                "checkout",
                "-b",
                "implementation/execution-lease-bridge",
            )
        _git(
            workspace,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
        )
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
        if bridge_scenario.startswith("effect_submodule_"):
            submodule = workspace / "dependency"
            if bridge_scenario in {
                "effect_submodule_symlink",
                "effect_submodule_non_git",
            }:
                if submodule.is_symlink():
                    submodule.unlink()
                elif submodule.exists():
                    shutil.rmtree(submodule)
                if bridge_scenario == "effect_submodule_symlink":
                    outside = tmp_path / "outside-submodule"
                    outside.mkdir()
                    submodule.symlink_to(outside, target_is_directory=True)
                else:
                    submodule.mkdir()
                    _write(submodule / "not-a-worktree.txt", "unsafe\n")
            else:
                self.worktree_submodule_paths = ("../outside-submodule",)
            try:
                self._full_plan_bound_effect_paths(
                    workspace,
                    baseline_ref=_git(
                        repo,
                        "rev-parse",
                        "HEAD",
                    ).stdout.strip(),
                )
                raise AssertionError(
                    "unsafe configured submodule was accepted"
                )
            finally:
                self._release_implementation_task_claim(
                    claim_path,
                    captured["claim"],
                )
        if bridge_scenario == "scope_drift":
            settling = self._mark_worktree_lifecycle_settling(workspace)
            assert settling is not None
            assert settling.state.value == "settling"
            _write(workspace / "src/test-a.py", "VALUE = 'declared'\n")
            _write(workspace / "src/undeclared-shared.py", "VALUE = 'drift'\n")
            try:
                self._validate_implementation_patch(
                    workspace,
                    tasks[0],
                    baseline_ref=_git(repo, "rev-parse", "HEAD").stdout.strip(),
                    allow_scope_adjudication=False,
                )
            finally:
                self._release_implementation_task_claim(
                    claim_path,
                    captured["claim"],
                )
        if bridge_scenario in {"no_change", "no_change_guard_denied"}:
            settling = self._mark_worktree_lifecycle_settling(workspace)
            assert settling is not None
            baseline = _git(repo, "rev-parse", "HEAD").stdout.strip()
            commit_result = self._commit_worktree_changes(
                workspace,
                tasks[0],
                1,
                baseline_ref=baseline,
            )
            assert commit_result["reason"] == "no_changes", commit_result
            branch = self._git_current_branch(workspace)
            captured["no_change_guard"] = (
                self._validated_no_change_completion_guard(
                    baseline_ref=baseline,
                    current_head=(
                        "0" * 40
                        if bridge_scenario == "no_change_guard_denied"
                        else baseline
                    ),
                    expected_branch=branch,
                    current_branch=branch,
                    validation_result={
                        "attempted": True,
                        "passed": True,
                        "returncode": 0,
                        "results": [],
                        "selection": {
                            "scope": "pre_merge",
                            "changed_files": [],
                        },
                    },
                )
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
        lambda _self, _workspace, *, task=None, **_kwargs: [
            "provider-test-command"
        ],
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
    lifecycle_token = _test_lifecycle_token(
        tmp_path,
        f"execution-lease-bridge-{bridge_scenario}",
    )
    launch_profile = multi_runner_module.LifecycleProfile(
        target_id="supervisor-track:execution-lease-bridge",
        run_id=f"test-execution-lease-bridge-{lifecycle_token}",
        configuration_root=(
            f"test-execution-lease-bridge-config-{lifecycle_token}"
        ),
        repository_root=str(repo),
        state_root=str(state_root),
        run_root=str(
            state_root
            / "lifecycle-runs"
            / f"execution-lease-bridge-{lifecycle_token}"
        ),
        argv=launch_argv,
        cwd=str(repo),
    )
    launch_process = _spawn_test_process(
        launch_argv,
        cwd=repo,
        env=launch_profile.launch_environment(0),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    launch_identity = _capture_test_process_identity(
        launch_process,
        launch_profile,
    )
    multi_runner_module._persist_plan_bound_process_birth(
        profile=launch_profile,
        process_identity=launch_identity,
        repo_root=repo,
    )
    helper_argv = command[command.index(marker) + 1 :]
    try:
        if bridge_scenario == "scope_drift":
            assert supervisor_module._run_plan_bound_daemon_child(
                helper_argv
            ) == supervisor_module.PLAN_BOUND_REPLAN_RETURN_CODE
        elif bridge_scenario in {
            "tamper",
            "mode_tamper",
            "effect_submodule_symlink",
            "effect_submodule_non_git",
            "effect_submodule_escape",
        }:
            with pytest.raises(
                supervisor_module.PlanBoundDispatchError,
                match=(
                    "lease or fence changed before provider"
                    if bridge_scenario == "tamper"
                    else (
                        "exactly mode 0600"
                        if bridge_scenario == "mode_tamper"
                        else {
                            "effect_submodule_symlink": "symlink",
                            "effect_submodule_non_git": "exact Git worktree",
                            "effect_submodule_escape": "effect path is unsafe",
                        }[bridge_scenario]
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
        "no_change": "merge_completed",
        "no_change_guard_denied": "provider_ready",
        "tamper": "workspace_prepared",
        "mode_tamper": "workspace_prepared",
        "effect_submodule_symlink": "provider_ready",
        "effect_submodule_non_git": "provider_ready",
        "effect_submodule_escape": "provider_ready",
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

    retry_process = _spawn_test_process(
        launch_argv,
        cwd=repo,
        env=launch_profile.launch_environment(1),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    retry_identity = _capture_test_process_identity(
        retry_process,
        launch_profile,
    )
    try:
        with pytest.raises(
            ValueError,
            match=(
                "provider boundary"
                if bridge_scenario
                in {
                    "normal",
                    "scope_drift",
                    "no_change",
                    "no_change_guard_denied",
                    "effect_submodule_symlink",
                    "effect_submodule_non_git",
                    "effect_submodule_escape",
                }
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
    if bridge_scenario == "no_change":
        # The no-change path now crosses the canonical queue/completion
        # boundary.  The exact task is complete, so there is no downstream
        # ready work to compile into a replacement wave.
        assert replacement is None
    else:
        assert replacement is not None
        assert replacement.binding.revision_cid != receipt.binding.revision_cid
    if bridge_scenario == "scope_drift":
        assert replacement is not None
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
            match="revision|fence|process birth",
        ):
            supervisor._build_daemon_command()


@pytest.mark.parametrize(
    "wave_scenario",
    (
        "mixed",
        "disjoint",
        "changed_no_change",
        "compact_hidden_drift",
        "crash_proposal_ready",
        "crash_before_enqueue",
        "crash_after_enqueue",
        "crash_confirmed",
        "crash_confirmed_retry",
        "crash_completed_before_finalize",
        "crash_serialized_merge_confirmed",
        "crash_no_change",
        "crash_after_enqueue_mismatch",
    ),
)
def test_genuine_two_lane_diff_barrier_precedes_every_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wave_scenario: str,
) -> None:
    """Every genuine daemon lane publishes before whole-wave enqueue release."""

    repo, config_path, board = _seed_v3_task_repo(
        tmp_path,
        (_task_block("TEST-A"), _task_block("TEST-B")),
    )
    if wave_scenario == "crash_serialized_merge_confirmed":
        _write(repo / ".gitignore", "*.json\n*.log\n*.duckdb\n")
        _git(repo, "add", ".gitignore")
        _git(repo, "commit", "-m", "seed ignored runtime artifact patterns")
    plan_common_args = scheduler_module.configured_board_common_args(
        board,
        implement=True,
    )
    if wave_scenario in {
        "changed_no_change",
        "crash_proposal_ready",
        "crash_before_enqueue",
        "crash_after_enqueue",
        "crash_confirmed",
        "crash_confirmed_retry",
        "crash_completed_before_finalize",
        "crash_no_change",
        "crash_after_enqueue_mismatch",
    }:
        _write(repo / "src/test-b.py", "VALUE = 'already-present'\n")
        _git(repo, "add", "src/test-b.py")
        _git(repo, "commit", "-m", "seed no-change sibling output")
    receipt = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(lanes=2),
        provider_capacity_snapshots=_provider_capacity(lanes=2),
        task_state_snapshots=(),
    )
    assert receipt is not None
    assert len(receipt.slice_manifest.nonempty) == 2
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=receipt.slice_manifest.source_head,
        source_tree=receipt.slice_manifest.repository_tree_id,
    )
    launch_plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        stamp="20260808T-two-lane-mixed",
        parallelism_receipt=receipt,
        accepted_control_plane_pin=control_plane_pin,
        accepted_control_plane_descriptor=control_plane_launch.descriptor,
    )
    children = tuple(
        multi_runner_module.PlanBoundSupervisorChild.from_cli_record(
            launch_plan["argv"][index + 1]
        )
        for index, token in enumerate(launch_plan["argv"][:-1])
        if token == "--implementation-plan-bound-track"
    )
    assert len(children) == 2
    crash_task_id = (
        "TEST-B"
        if wave_scenario
        in {"crash_no_change", "crash_serialized_merge_confirmed"}
        else "TEST-A"
    )

    helpers: list[tuple[multi_runner_module.PlanBoundSupervisorChild, list[str]]] = []
    supervisors: list[subprocess.Popen[bytes]] = []
    for child in children:
        state_dir = repo / child.state_dir
        config = supervisor_module.PortalSupervisorConfig(
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
            plan_revision_store_path=repo / child.plan_revision_store_path,
            plan_bound_revision_cid=child.revision_cid,
            plan_bound_plan_root_cid=child.plan_root_cid,
            plan_bound_execution_plan_cid=child.execution_plan_cid,
            plan_bound_capacity_snapshot_id=child.capacity_snapshot_id,
            plan_bound_slice_manifest_cid=child.slice_manifest_cid,
            plan_bound_slice_id=child.slice_id,
            plan_bound_lane_id=child.lane_id,
            plan_bound_source_head=child.source_head,
            plan_bound_source_tree=child.source_tree,
            plan_bound_task_source_revision=child.task_source_revision,
            plan_bound_configuration_root=child.configuration_root,
            plan_bound_accepted_tree_root=repo,
            accepted_control_plane_pin=control_plane_pin,
            accepted_control_plane_descriptor=control_plane_launch.descriptor,
        )
        supervisor = supervisor_module.PortalImplementationSupervisor(config)
        with monkeypatch.context() as build_context:
            # The genuine parent supervisor performs this check in its own
            # persisted process.  The test builds the exact nested argv in the
            # pytest process, after separately persisting the launch birth.
            build_context.setattr(
                supervisor_module.PortalImplementationSupervisor,
                "_validated_plan_bound_slice",
                lambda _self: None,
            )
            command = supervisor._build_daemon_command()
        marker = supervisor_module.PLAN_BOUND_DAEMON_CHILD_MARKER
        helper_argv = command[command.index(marker) + 1 :]

        track = child.track(stamp="20260808T-two-lane-mixed").resolve(repo)
        launch_argv = (
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            *track.extra_args,
        )
        state_root = track.supervisor_pid_path.parent.resolve()
        lifecycle_token = _test_lifecycle_token(
            tmp_path,
            f"two-lane-{wave_scenario}-{child.lane_id}",
        )
        profile = multi_runner_module.LifecycleProfile(
            target_id=f"supervisor-track:{child.name}",
            run_id=f"test-two-lane-{wave_scenario}-{lifecycle_token}",
            configuration_root=(
                f"test-two-lane-{wave_scenario}-{lifecycle_token}"
            ),
            repository_root=str(repo.resolve()),
            state_root=str(state_root),
            run_root=str(
                state_root
                / "lifecycle-runs"
                / f"{child.name}-{lifecycle_token}"
            ),
            argv=launch_argv,
            cwd=str(repo.resolve()),
        )
        process = _spawn_test_process(
            launch_argv,
            cwd=repo,
            env=profile.launch_environment(0),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        process_identity = _capture_test_process_identity(process, profile)
        multi_runner_module._persist_plan_bound_process_birth(
            profile=profile,
            process_identity=process_identity,
            repo_root=repo,
        )
        supervisors.append(process)
        helpers.append((child, helper_argv))

    original_capacity_observation = (
        scheduler_module.configured_board_capacity_observation
    )

    def capacity_observation(current_board, **kwargs):
        if kwargs:
            return original_capacity_observation(current_board, **kwargs)
        return _host_capacity(lanes=2), _provider_capacity(lanes=2), PLAN_NOW

    monkeypatch.setattr(
        scheduler_module,
        "configured_board_capacity_observation",
        capacity_observation,
    )
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

    def provider_gate(_self, *, task, attempt, worktree_path):
        del worktree_path
        return {
            "skip_provider": False,
            "provider_authorized": True,
            "disposition": "residual_llm_authorized",
            "reason_code": "test_bound_provider",
            "receipt_cid": f"receipt:test:{task.task_id}:{attempt}",
            "event": {},
        }

    def implementation_command(_self, _workspace, *, task=None, **_kwargs):
        assert task is not None
        provider_invocation_path = tmp_path / "provider-command-invocations.jsonl"
        descriptor = os.open(
            provider_invocation_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(
                descriptor,
                _canonical_json_bytes({"task_id": task.task_id}) + b"\n",
            )
        finally:
            os.close(descriptor)
        if wave_scenario in {
            "changed_no_change",
            "crash_proposal_ready",
            "crash_before_enqueue",
            "crash_after_enqueue",
            "crash_confirmed",
            "crash_confirmed_retry",
            "crash_completed_before_finalize",
            "crash_no_change",
            "crash_after_enqueue_mismatch",
        } and task.task_id == "TEST-B":
            command = "pass"
            if wave_scenario == "crash_proposal_ready":
                # Keep the no-change sibling behind TEST-A long enough for
                # TEST-A to die after publishing its immutable disposition but
                # before any process can publish the whole-wave decision.
                command = "import time; time.sleep(0.5)"
            return [sys.executable, "-c", command]
        target = (
            "src/test-b.py"
            if wave_scenario == "mixed"
            else f"src/{task.task_id.lower()}.py"
        )
        # In the mixed case TEST-A drifts into TEST-B's exact valid path while
        # TEST-B changes its own path.  Other cases exercise disjoint release
        # and changed+no-change release through the same production boundary.
        payload = f"VALUE = {task.task_id!r}\n"
        hidden_effect = (
            "q=Path('src/hidden.py'); q.write_text('HIDDEN = True\\n', "
            "encoding='utf-8'); "
            if wave_scenario == "compact_hidden_drift"
            and task.task_id == "TEST-A"
            else ""
        )
        return [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                f"p=Path({target!r}); p.parent.mkdir(parents=True, exist_ok=True); "
                f"p.write_text({payload!r}, encoding='utf-8'); "
                f"{hidden_effect}"
            ),
        ]

    def production_proposal_validation(
        self,
        workspace_path,
        task,
        _log_path,
        *,
        baseline_ref,
        **_kwargs,
    ):
        if not self._run_git(
            ["status", "--porcelain"],
            cwd=workspace_path,
        ).stdout.strip():
            return {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "results": [],
                "selection": {"scope": "pre_merge", "changed_files": []},
                "proposal_gate": {
                    "attempted": False,
                    "accepted": True,
                    "reason": "no_candidate_changes",
                    "changed_paths": [],
                },
            }
        if wave_scenario == "compact_hidden_drift" and task.task_id == "TEST-A":
            return {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "results": [],
                "selection": {
                    "scope": "pre_merge",
                    "changed_files": ["src/test-a.py"],
                },
                "proposal_gate": {
                    "attempted": True,
                    "accepted": True,
                    "reason_codes": [],
                    "proposal_id": "proposal:test:compact-hidden",
                    "policy_id": "policy:test:compact-hidden",
                    "receipt_id": "receipt:test:compact-hidden",
                    "repository_tree_id": baseline_ref,
                    "changed_paths": ["src/test-a.py"],
                    "proof_authoritative": False,
                    "completion_authoritative": False,
                },
            }
        proposal = self._validate_implementation_patch(
            workspace_path,
            task,
            baseline_ref=baseline_ref,
            allow_scope_adjudication=False,
        )
        compact = self._compact_proposal_validation(proposal)
        return {
            "attempted": True,
            "passed": bool(proposal.accepted),
            "returncode": 0 if proposal.accepted else 2,
            "results": [],
            "selection": {
                "scope": "pre_merge",
                "changed_files": list(compact["changed_paths"]),
            },
            "proposal_gate": compact,
        }

    enqueue_receipt_path = tmp_path / "canonical-enqueue-reached.jsonl"
    crash_receipt_path = tmp_path / "canonical-enqueue-crash.json"
    store = PlanRevisionStore(repo / children[0].plan_revision_store_path)
    original_queue_enqueue = daemon_module.MergeQueue.enqueue
    original_queue_get = daemon_module.MergeQueue.get
    original_consume_merge = (
        daemon_module.PortalImplementationDaemon._consume_one_merge_candidate
    )
    original_await_barrier = (
        execution_plan_module.ProductionParallelPlanAdapter.await_wave_diff_barrier
    )

    def record_canonical_enqueue(self, **kwargs):
        barrier = ProductionParallelPlanAdapter(store).load_wave_diff_barrier(
            revision_cid=receipt.binding.revision_cid,
            slice_manifest_cid=receipt.slice_manifest_cid,
        )
        row = {
            "task_id": kwargs["task_id"],
            "barrier_decision": "" if barrier is None else barrier[1].decision,
            "disposition_count": (
                0 if barrier is None else len(barrier[1].dispositions)
            ),
        }
        if (
            wave_scenario == "crash_before_enqueue"
            and kwargs["task_id"] == "TEST-A"
            and not crash_receipt_path.exists()
        ):
            crash_receipt_path.write_text("before\n", encoding="utf-8")
            os._exit(86)
        request = original_queue_enqueue(self, **kwargs)
        descriptor = os.open(
            enqueue_receipt_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, _canonical_json_bytes(row) + b"\n")
        finally:
            os.close(descriptor)
        if (
            wave_scenario
            in {"crash_after_enqueue", "crash_after_enqueue_mismatch"}
            and kwargs["task_id"] == "TEST-A"
            and not crash_receipt_path.exists()
        ):
            crash_receipt_path.write_text("after\n", encoding="utf-8")
            os._exit(86)
        return request

    def current_plan_attempt() -> tuple[str, str]:
        with store._thread_lock:
            with store._guard():
                for child in children:
                    current = execution_plan_module._load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=receipt.binding.revision_cid,
                        slice_id=child.slice_id,
                        lane_id=child.lane_id,
                    )
                    if (
                        current is not None
                        and current[1].daemon_process_birth.get("pid")
                        == os.getpid()
                    ):
                        return current[1].active_task_id, current[1].phase
        return "", ""

    def crash_after_proposal_barrier(self, **kwargs):
        if wave_scenario in {"crash_proposal_ready", "crash_no_change"}:
            task_id, phase = current_plan_attempt()
            if (
                task_id == crash_task_id
                and phase == "proposal_ready"
                and not crash_receipt_path.exists()
            ):
                crash_receipt_path.write_text(
                    "proposal_ready\n",
                    encoding="utf-8",
                )
                os._exit(86)
        return original_await_barrier(self, **kwargs)

    def crash_after_queue_confirmation(self):
        if wave_scenario in {
            "crash_confirmed",
            "crash_confirmed_retry",
            "crash_serialized_merge_confirmed",
        }:
            task_id, phase = current_plan_attempt()
            if (
                task_id == crash_task_id
                and phase == "merge_enqueue_confirmed"
                and not crash_receipt_path.exists()
            ):
                crash_receipt_path.write_text(
                    "merge_enqueue_confirmed\n",
                    encoding="utf-8",
                )
                os._exit(86)
        return original_consume_merge(self)

    def crash_after_queue_completion(self, request_id):
        request = original_queue_get(self, request_id)
        if (
            wave_scenario == "crash_completed_before_finalize"
            and request is not None
            and request.task_id == crash_task_id
            and request.status == "completed"
            and not crash_receipt_path.exists()
        ):
            task_id, phase = current_plan_attempt()
            if task_id == crash_task_id and phase == "merge_enqueue_confirmed":
                crash_receipt_path.write_text(
                    "queue_completed_before_task_finalize\n",
                    encoding="utf-8",
                )
                os._exit(86)
        return request

    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_evaluate_pre_implementation_provider_gate",
        provider_gate,
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_build_implementation_command",
        implementation_command,
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_run_validation_with_candidate_binding",
        production_proposal_validation,
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_apply_implementation_failure_review",
        lambda _self, **kwargs: dict(kwargs["validation_result"]),
    )
    canonical_board_completion = (
        daemon_module.PortalImplementationDaemon._board_completion_decision
    )

    def keep_crash_probe_no_change_pending(
        *,
        returncode: int,
        merge_result: dict[str, Any],
        no_change_completion: bool,
    ) -> dict[str, Any]:
        if wave_scenario.startswith("crash_") and no_change_completion:
            return {
                "complete": False,
                "pending_merge": False,
                "reason": "test_crash_probe_no_change_pending",
            }
        return canonical_board_completion(
            returncode=returncode,
            merge_result=merge_result,
            no_change_completion=no_change_completion,
        )

    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_board_completion_decision",
        staticmethod(keep_crash_probe_no_change_pending),
    )
    monkeypatch.setattr(
        daemon_module.MergeQueue,
        "enqueue",
        record_canonical_enqueue,
    )
    monkeypatch.setattr(
        daemon_module.MergeQueue,
        "get",
        crash_after_queue_completion,
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_consume_one_merge_candidate",
        crash_after_queue_confirmation,
    )
    monkeypatch.setattr(
        execution_plan_module.ProductionParallelPlanAdapter,
        "await_wave_diff_barrier",
        crash_after_proposal_barrier,
    )

    child_pids: dict[int, str] = {}
    error_paths: dict[int, Path] = {}
    try:
        for child, helper_argv in helpers:
            pid = os.fork()
            if pid == 0:  # pragma: no branch - isolated production boundary
                try:
                    child_rc = supervisor_module._run_plan_bound_daemon_child(
                        helper_argv
                    )
                except BaseException as exc:  # noqa: BLE001
                    error_path = tmp_path / f"child-{child.lane_id}.error"
                    error_path.write_text(
                        f"{type(exc).__name__}: {exc}\n",
                        encoding="utf-8",
                    )
                    os._exit(97)
                os._exit(int(child_rc))
            child_pids[pid] = child.lane_id
            error_paths[pid] = tmp_path / f"child-{child.lane_id}.error"

        statuses: dict[str, int] = {}
        deadline = time.monotonic() + 45.0
        while child_pids and time.monotonic() < deadline:
            for pid, lane_id in tuple(child_pids.items()):
                observed, status = os.waitpid(pid, os.WNOHANG)
                if observed:
                    statuses[lane_id] = os.waitstatus_to_exitcode(status)
                    child_pids.pop(pid)
            if wave_scenario.startswith("crash_") and any(
                returncode == 86 for returncode in statuses.values()
            ):
                # Keep the surviving lane blocked on the production manifest-
                # order merge turn while the accepted recovery supervisor is
                # started below.  Waiting for every original child here would
                # reproduce the old false topology in which no runner existed
                # to recover the crashed predecessor.
                break
            if child_pids:
                time.sleep(0.02)
        if wave_scenario.startswith("crash_"):
            crashed_lane = next(
                child.lane_id
                for child, _helper in helpers
                if child.task_ids == (crash_task_id,)
            )
            assert statuses.get(crashed_lane) == 86
            assert all(
                returncode in {0, 86} for returncode in statuses.values()
            )
        else:
            assert not child_pids, (
                "plan-bound children did not reach a barrier decision"
            )
            expected_returncode = (
                supervisor_module.PLAN_BOUND_REPLAN_RETURN_CODE
                if wave_scenario in {"mixed", "compact_hidden_drift"}
                else 0
            )
            expected_statuses = {
                child.lane_id: expected_returncode
                for child, _helper in helpers
            }
            assert statuses == expected_statuses, {
                "statuses": statuses,
                "errors": {
                    str(path): path.read_text(encoding="utf-8")
                    for path in error_paths.values()
                    if path.exists()
                },
            }
            if wave_scenario == "mixed":
                # Feed one genuine rejected child outcome through the outer
                # production runner.  A terminal rejected barrier is STEER
                # evidence, never authority to relaunch provider-bearing
                # work under the same revision.
                rejected_child = children[0]
                rejected_track = rejected_child.track(
                    stamp="20260808T-rejected-no-restart"
                ).resolve(repo)

                class CompletedRejectedTrack:
                    pid = supervisors[0].pid

                    @staticmethod
                    def poll() -> int:
                        return supervisor_module.PLAN_BOUND_REPLAN_RETURN_CODE

                    @staticmethod
                    def wait(*, timeout: float | None = None) -> int:
                        del timeout
                        return supervisor_module.PLAN_BOUND_REPLAN_RETURN_CODE

                rejected_start_calls = 0

                def rejected_start(*_args, **_kwargs):
                    nonlocal rejected_start_calls
                    rejected_start_calls += 1
                    assert rejected_start_calls == 1, (
                        "runner restarted a rejected provider-bearing slice"
                    )
                    return CompletedRejectedTrack()

                canonical_terminate = (
                    multi_runner_module._terminate_managed_process
                )

                def rejected_terminate(process, **kwargs):
                    if isinstance(process, CompletedRejectedTrack):
                        return True, []
                    return canonical_terminate(process, **kwargs)

                with monkeypatch.context() as runner_patch:
                    runner_patch.setattr(
                        multi_runner_module,
                        "start_track",
                        rejected_start,
                    )
                    runner_patch.setattr(
                        multi_runner_module,
                        "_terminate_managed_process",
                        rejected_terminate,
                    )
                    rejected_result = (
                        multi_runner_module.run_supervisor_tracks(
                            (rejected_track,),
                            repo_root=repo,
                            common_args=(),
                            duration_seconds=0.25,
                            heartbeat_interval_seconds=0.01,
                            stop_grace_seconds=0.01,
                            exit_when_all_tracks_terminal=True,
                            plan_bound_children=(rejected_child,),
                            accepted_control_plane_pin=control_plane_pin,
                            accepted_control_plane_descriptor=(
                                control_plane_launch.descriptor
                            ),
                            output=lambda _message: None,
                        )
                    )
                assert rejected_start_calls == 1
                assert rejected_result["replan_required"] is True, rejected_result
                assert rejected_result["terminal_quiescent"] is False

        if wave_scenario.startswith("crash_"):
            crashed_index = next(
                index
                for index, (child, _helper) in enumerate(helpers)
                if child.task_ids == (crash_task_id,)
            )
            crashed_child, crashed_helper = helpers[crashed_index]
            with store._thread_lock:
                with store._guard():
                    prepared = execution_plan_module._load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=receipt.binding.revision_cid,
                        slice_id=crashed_child.slice_id,
                        lane_id=crashed_child.lane_id,
                    )
            assert prepared is not None
            expected_crash_phase = {
                "crash_proposal_ready": "proposal_ready",
                "crash_before_enqueue": "merge_enqueue_prepared",
                "crash_after_enqueue": "merge_enqueue_prepared",
                "crash_confirmed": "merge_enqueue_confirmed",
                "crash_confirmed_retry": "merge_enqueue_confirmed",
                "crash_completed_before_finalize": (
                    "merge_enqueue_confirmed"
                ),
                "crash_serialized_merge_confirmed": (
                    "merge_enqueue_confirmed"
                ),
                "crash_no_change": "proposal_ready",
                "crash_after_enqueue_mismatch": "merge_enqueue_prepared",
            }[wave_scenario]
            assert prepared[1].phase == expected_crash_phase
            assert prepared[1].merge_enqueue_reached is (
                expected_crash_phase != "proposal_ready"
            )
            assert bool(prepared[1].merge_request_id) is (
                expected_crash_phase == "merge_enqueue_confirmed"
            )

            if wave_scenario == "crash_after_enqueue_mismatch":
                queue = daemon_module.MergeQueue(
                    board.path(board.runtime_paths["merge_queue"])
                )
                with queue._connect() as connection:
                    row = connection.execute(
                        "SELECT request_id, metadata_json FROM merge_requests "
                        "WHERE task_id='TEST-A'"
                    ).fetchone()
                    assert row is not None
                    mismatched_metadata = json.loads(row["metadata_json"])
                    mismatched_metadata["baseline_ref"] = "0" * 40
                    connection.execute(
                        "UPDATE merge_requests SET metadata_json=? "
                        "WHERE request_id=?",
                        (
                            json.dumps(
                                mismatched_metadata,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            row["request_id"],
                        ),
                    )

            retry_transition: dict[str, Any] = {}
            if wave_scenario == "crash_confirmed_retry":
                queue = daemon_module.MergeQueue(
                    board.path(board.runtime_paths["merge_queue"])
                )
                pending = queue.get(prepared[1].merge_request_id)
                assert pending is not None
                assert pending.status == "pending"
                claimed = queue.dequeue(consumer_id="test-plan-bound-retry")
                assert claimed is not None
                assert claimed.request_id == pending.request_id
                assert claimed.status == "processing"
                assert claimed.claim_token
                assert claimed.consumer_id == "test-plan-bound-retry"
                assert claimed.claim_generation == pending.claim_generation + 1
                assert claimed.claimed_at >= pending.enqueued_at
                retried = queue.requeue(
                    claimed,
                    reason="test canonical retry transition",
                    metadata={"kind": "test_plan_bound_retry"},
                )
                assert retried is not None and not isinstance(retried, Path)
                assert retried.status == "pending"
                assert retried.attempt == pending.attempt + 1
                assert retried.failure_count == pending.failure_count + 1
                assert retried.claim_generation == claimed.claim_generation + 1
                assert retried.consumer_id == ""
                assert retried.claim_token == ""
                assert retried.claimed_at == 0
                retry_transition = {
                    "enqueued_at": pending.enqueued_at,
                    "claim_generation": retried.claim_generation,
                    "attempt": retried.attempt,
                    "failure_count": retried.failure_count,
                }

            # The accepted recovery supervisor replaces the dead original
            # process birth, then forks the merge-only daemon as its direct
            # child.  The provider method is a tripwire: recovery must never
            # select or dispatch the task again.
            crashed_supervisor = supervisors[crashed_index]
            if crashed_supervisor.poll() is None:
                crashed_supervisor.terminate()
                crashed_supervisor.wait(timeout=5)
            recovery_track = crashed_child.track(
                stamp="20260808T-two-lane-recovery"
            ).resolve(repo)
            if wave_scenario == "crash_serialized_merge_confirmed":
                # TEST-A has already crossed the canonical merge train and
                # advanced the shared checkout.  The later TEST-B child may
                # therefore restart only through the exact persisted
                # confirmed handoff; a normal source-generation launch must
                # remain fenced.
                advanced_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
                advanced_tree = _git(
                    repo,
                    "rev-parse",
                    "HEAD^{tree}",
                ).stdout.strip()
                assert advanced_head != receipt.slice_manifest.source_head
                _git(
                    repo,
                    "merge-base",
                    "--is-ancestor",
                    receipt.slice_manifest.source_head,
                    advanced_head,
                )
                fsmonitor_sentinel = tmp_path / "hostile-fsmonitor-ran"
                fsmonitor_script = tmp_path / "hostile-fsmonitor.sh"
                fsmonitor_script.write_text(
                    "#!/bin/sh\n"
                    f"printf bad > {str(fsmonitor_sentinel)!r}\n"
                    "exit 0\n",
                    encoding="utf-8",
                )
                fsmonitor_script.chmod(0o700)
                _git(
                    repo,
                    "config",
                    "core.fsmonitor",
                    str(fsmonitor_script),
                )
                hostile_recovery_import = tmp_path / "recovery-shadow-imported"
                probe_root = recovery_track.supervisor_pid_path.parent
                sealed_probe_track = replace(
                    recovery_track,
                    name=f"{recovery_track.name}-sealed-recovery-probe",
                    log_path=probe_root / "sealed-recovery-probe.log",
                    supervisor_pid_path=(
                        probe_root / "sealed-recovery-probe.pid"
                    ),
                    daemon_pid_path=probe_root / "sealed-recovery-daemon.pid",
                    supervisor_status_path=(
                        probe_root / "sealed-recovery-status.json"
                    ),
                    extra_args=(*recovery_track.extra_args, "--help"),
                )
                sealed_probe = multi_runner_module.start_track(
                    sealed_probe_track,
                    repo_root=repo,
                    common_args=plan_common_args,
                    python_executable=sys.executable,
                    accepted_control_plane_pin=control_plane_pin,
                    accepted_control_plane_descriptor=(
                        control_plane_launch.descriptor
                    ),
                    output=lambda _message: None,
                )
                assert sealed_probe.wait(timeout=30) == 0
                assert not hostile_recovery_import.exists()
                assert not fsmonitor_sentinel.exists()
                assert multi_runner_module._remove_owned_pid_projection(
                    sealed_probe_track.supervisor_pid_path,
                    sealed_probe.pid,
                )
                sealed_profile = getattr(
                    sealed_probe,
                    "_agent_supervisor_lifecycle_profile",
                )
                sealed_argv = list(sealed_profile.argv)
                marker_index = sealed_argv.index(
                    multi_runner_module.PLAN_BOUND_LAUNCH_GATE_MARKER
                )
                recovery_authorization_cid = sealed_argv[marker_index + 5]
                assert recovery_authorization_cid not in {"", "-"}
                store_adapter = ProductionParallelPlanAdapter(store)
                recovery_authorization = store_adapter.load_recovery_launch(
                    revision_cid=receipt.binding.revision_cid,
                    slice_id=crashed_child.slice_id,
                    lane_id=crashed_child.lane_id,
                    authorization_cid=recovery_authorization_cid,
                )
                assert recovery_authorization.execution_phase == (
                    "merge_enqueue_confirmed"
                )
                assert recovery_authorization.repository_head == advanced_head
                assert recovery_authorization.repository_tree == advanced_tree

                separator_index = sealed_argv.index("--", marker_index)
                sealed_child_command = sealed_argv[separator_index + 1 :]
                sealed_probe_track.log_path.unlink()
                sealed_probe_track.supervisor_pid_path.with_name(
                    f".{sealed_probe_track.supervisor_pid_path.name}.update.lock"
                ).unlink()

                def denied_gate_returncode(recovery_token: str) -> int:
                    gate_read_fd, gate_write_fd = os.pipe()
                    gate_argv = (
                        multi_runner_module.PLAN_BOUND_LAUNCH_GATE_MARKER,
                        str(gate_read_fd),
                        str(repo.resolve()),
                        multi_runner_module.accepted_control_plane_pin_json(
                            control_plane_pin
                        ),
                        str(control_plane_launch.descriptor),
                        recovery_token,
                        "--",
                        *sealed_child_command,
                    )
                    gate_command = (
                        multi_runner_module.build_sealed_control_plane_module_command(
                            python_executable=sys.executable,
                            pin=control_plane_pin,
                            descriptor=control_plane_launch.descriptor,
                            module_name=(
                                multi_runner_module.PLAN_BOUND_LAUNCH_GATE_MODULE
                            ),
                            argv=gate_argv,
                        )
                    )
                    gate_process = _spawn_test_process(
                        gate_command,
                        cwd=repo,
                        env={"PATH": "/usr/bin:/bin"},
                        pass_fds=(
                            gate_read_fd,
                            control_plane_launch.descriptor,
                        ),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                    os.close(gate_read_fd)
                    try:
                        os.write(
                            gate_write_fd,
                            multi_runner_module.PLAN_BOUND_LAUNCH_GATE_SUCCESS,
                        )
                    finally:
                        os.close(gate_write_fd)
                    _stdout, _stderr = gate_process.communicate(timeout=30)
                    return int(gate_process.returncode)

                # Local repository configuration is untrusted input.  A
                # foreign core.worktree must not redirect the gate's HEAD or
                # cleanliness probes away from the accepted checkout.
                foreign_core_worktree = tmp_path / "foreign-core-worktree"
                foreign_worktree_add = multi_runner_module._plan_bound_git(
                    repo,
                    "worktree",
                    "add",
                    "--detach",
                    str(foreign_core_worktree),
                    advanced_head,
                )
                assert foreign_worktree_add.returncode == 0, (
                    foreign_worktree_add.stderr
                )
                readme_path = repo / "README.md"
                readme_payload = readme_path.read_bytes()
                _git(
                    repo,
                    "config",
                    "core.worktree",
                    str(foreign_core_worktree),
                )
                readme_path.write_bytes(readme_payload + b"hostile drift\n")
                try:
                    bound_status = multi_runner_module._plan_bound_git(
                        repo,
                        "status",
                        "--porcelain=v1",
                        "-z",
                        "--untracked-files=all",
                        "--ignored=matching",
                        "--ignore-submodules=none",
                    )
                    assert bound_status.returncode == 0
                    assert " M README.md\0" in str(bound_status.stdout)
                finally:
                    readme_path.write_bytes(readme_payload)
                    _git(repo, "config", "--unset", "core.worktree")
                    foreign_worktree_remove = (
                        multi_runner_module._plan_bound_git(
                            repo,
                            "worktree",
                            "remove",
                            "--force",
                            str(foreign_core_worktree),
                        )
                    )
                    assert foreign_worktree_remove.returncode == 0, (
                        foreign_worktree_remove.stderr
                    )

                # A local submodule ignore policy likewise cannot hide a
                # modified tracked dependency from the recovery gate.
                dependency_payload_path = repo / "dependency" / "dependency.txt"
                dependency_payload = dependency_payload_path.read_bytes()
                _git(repo, "config", "submodule.dependency.ignore", "all")
                dependency_payload_path.write_bytes(
                    dependency_payload + b"hostile submodule drift\n"
                )
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    dependency_payload_path.write_bytes(dependency_payload)
                    _git(
                        repo,
                        "config",
                        "--unset",
                        "submodule.dependency.ignore",
                    )

                assert denied_gate_returncode("-") == 78
                assert denied_gate_returncode(
                    recovery_authorization_cid + "-tampered"
                ) == 78
                (repo / "sitecustomize.py").write_text(
                    "from pathlib import Path\n"
                    f"Path({str(hostile_recovery_import)!r}).write_text('bad')\n",
                    encoding="utf-8",
                )
                (repo / "duckdb.py").write_text(
                    "from pathlib import Path\n"
                    f"Path({str(hostile_recovery_import)!r}).write_text('bad')\n",
                    encoding="utf-8",
                )
                assert denied_gate_returncode(recovery_authorization_cid) == 78
                assert not hostile_recovery_import.exists()
                (repo / "sitecustomize.py").unlink()
                (repo / "duckdb.py").unlink()
                unbound_runtime = (
                    board.path(board.runtime_paths["state"])
                    / crashed_child.lane_id
                    / "unbound-runtime.json"
                )
                unbound_runtime.write_text("{}\n", encoding="utf-8")
                assert denied_gate_returncode(recovery_authorization_cid) == 78
                unbound_runtime.unlink()
                foreign_log = unbound_runtime.with_name("foreign-runtime.log")
                foreign_log.write_text("foreign\n", encoding="utf-8")
                assert denied_gate_returncode(recovery_authorization_cid) == 78
                foreign_log.unlink()
                executable_json = unbound_runtime.with_name(
                    "foreign-executable.json"
                )
                executable_json.write_text("{}\n", encoding="utf-8")
                executable_json.chmod(0o700)
                assert denied_gate_returncode(recovery_authorization_cid) == 78
                executable_json.unlink()
                symlink_target = tmp_path / "foreign-runtime-target.json"
                symlink_target.write_text("{}\n", encoding="utf-8")
                hostile_symlink = unbound_runtime.with_name(
                    "foreign-runtime-symlink.json"
                )
                hostile_symlink.symlink_to(symlink_target)
                assert denied_gate_returncode(recovery_authorization_cid) == 78
                hostile_symlink.unlink()
                symlink_target.unlink()
                hardlink_source = tmp_path / "foreign-runtime-hardlink.json"
                hardlink_source.write_text("{}\n", encoding="utf-8")
                hostile_hardlink = unbound_runtime.with_name(
                    "foreign-runtime-hardlink.json"
                )
                os.link(hardlink_source, hostile_hardlink)
                assert denied_gate_returncode(recovery_authorization_cid) == 78
                hostile_hardlink.unlink()
                hardlink_source.unlink()
                unbound_candidate = (
                    board.path(board.runtime_paths["worktrees"])
                    / "unbound-candidate"
                    / "payload.json"
                )
                unbound_candidate.parent.mkdir(parents=True)
                unbound_candidate.write_text("{}\n", encoding="utf-8")
                assert denied_gate_returncode(recovery_authorization_cid) == 78
                unbound_candidate.unlink()
                unbound_candidate.parent.rmdir()

                # Exact owner-derived artifacts receive no executable or
                # writable-marker exemption.  Tamper the active recovery
                # lane's own lock/workspace identities, not merely foreign
                # lookalike names, and require the sealed gate to fail closed.
                implementation_lock = probe_root / "implementation.lock"
                lock_mode = stat.S_IMODE(os.lstat(implementation_lock).st_mode)
                implementation_lock.chmod(lock_mode | 0o100)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    implementation_lock.chmod(lock_mode)
                implementation_lock_backup = (
                    tmp_path / "active-implementation-lock"
                )
                os.replace(implementation_lock, implementation_lock_backup)
                implementation_lock.mkdir()
                (implementation_lock / "payload.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                )
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    shutil.rmtree(implementation_lock)
                    os.replace(
                        implementation_lock_backup,
                        implementation_lock,
                    )

                crashed_execution = store_adapter.load_execution_lease(
                    revision_cid=receipt.binding.revision_cid,
                    slice_id=crashed_child.slice_id,
                    lane_id=crashed_child.lane_id,
                )
                assert crashed_execution is not None
                active_workspace = Path(crashed_execution[1].workspace_path)
                workspace_mode = stat.S_IMODE(
                    os.lstat(active_workspace).st_mode
                )
                assert workspace_mode == 0o700
                active_workspace.chmod(0o770)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    active_workspace.chmod(workspace_mode)
                active_workspace_backup = tmp_path / "active-workspace"
                os.replace(active_workspace, active_workspace_backup)
                active_workspace.write_text(
                    "not a worktree\n",
                    encoding="utf-8",
                )
                active_workspace.chmod(0o600)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    active_workspace.unlink()
                    os.replace(active_workspace_backup, active_workspace)

                git_marker = active_workspace / ".git"
                marker_bytes = git_marker.read_bytes()
                marker_mode = stat.S_IMODE(os.lstat(git_marker).st_mode)
                assert marker_mode == 0o600

                # The exact launch-owned PID/log names are file projections.
                # An embedded-worktree directory at either name must not be
                # reinterpreted as an authenticated workspace after the
                # recovery snapshot was published.
                for launch_owned_path in (
                    sealed_probe_track.log_path,
                    sealed_probe_track.supervisor_pid_path,
                ):
                    assert not launch_owned_path.exists()
                    launch_owned_path.mkdir(mode=0o700)
                    launch_marker = launch_owned_path / ".git"
                    launch_marker.write_bytes(marker_bytes)
                    launch_marker.chmod(marker_mode)
                    try:
                        assert denied_gate_returncode(
                            recovery_authorization_cid
                        ) == 78
                    finally:
                        shutil.rmtree(launch_owned_path)

                git_marker.chmod(0o666)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    git_marker.chmod(marker_mode)

                marker_backup = tmp_path / "active-worktree-git-marker"
                os.replace(git_marker, marker_backup)
                git_marker.symlink_to(marker_backup)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    git_marker.unlink()
                    os.replace(marker_backup, git_marker)

                os.replace(git_marker, marker_backup)
                os.link(marker_backup, git_marker)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    git_marker.unlink()
                    os.replace(marker_backup, git_marker)

                outside_git_dir = tmp_path / "outside-worktree-git-dir"
                outside_git_dir.mkdir()
                git_marker.write_text(
                    f"gitdir: {outside_git_dir}\n",
                    encoding="utf-8",
                )
                git_marker.chmod(marker_mode)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    git_marker.write_bytes(marker_bytes)
                    git_marker.chmod(marker_mode)

                git_dir = Path(
                    marker_bytes.decode("utf-8").strip()[len("gitdir: ") :]
                )
                git_dir_mode = stat.S_IMODE(os.lstat(git_dir).st_mode)
                assert git_dir_mode == 0o700
                git_dir.chmod(0o770)
                try:
                    assert denied_gate_returncode(
                        recovery_authorization_cid
                    ) == 78
                finally:
                    git_dir.chmod(git_dir_mode)
                assert not fsmonitor_sentinel.exists()
                _git(repo, "config", "--unset", "core.fsmonitor")
            gate_read, gate_write = os.pipe()
            recovery_error_path = tmp_path / "merge-recovery.error"
            replay_path = tmp_path / "merge-recovery-provider-replay"
            recovery_code = "\n".join(
                (
                    "import json, os, sys",
                    "from pathlib import Path",
                    "sys.path.insert(0, os.environ['_ASE3_TEST_SOURCE_ROOT'])",
                    "from ipfs_accelerate_py.agent_supervisor.entrypoints "
                    "import local_profile as local_profile_module",
                    "from ipfs_accelerate_py.agent_supervisor.runtime import "
                    "configured_board_scheduler as scheduler_module",
                    "from ipfs_accelerate_py.agent_supervisor.todo_daemon import "
                    "implementation_daemon as daemon_module",
                    "from ipfs_accelerate_py.agent_supervisor.todo_daemon import "
                    "implementation_supervisor as supervisor_module",
                    "repo = Path(os.environ['_ASE3_TEST_REPO'])",
                    "daemon_module.REPO_ROOT = repo",
                    "supervisor_module.__file__ = str(repo / "
                    "'ipfs_accelerate_py/agent_supervisor/todo_daemon/' "
                    "'implementation_supervisor.py')",
                    "local_profile_module._LIFECYCLE_REGISTRY_ROOT_OVERRIDE = "
                    "Path(os.environ['_ASE3_TEST_LIFECYCLE_ROOT'])",
                    "host = json.loads(os.environ['_ASE3_TEST_HOST'])",
                    "providers = tuple(json.loads("
                    "os.environ['_ASE3_TEST_PROVIDERS']))",
                    "now_ms = int(os.environ['_ASE3_TEST_NOW_MS'])",
                    "scheduler_module.configured_board_capacity_observation = "
                    "lambda _board, **_kwargs: (host, providers, now_ms)",
                    "replay_path = Path(os.environ['_ASE3_TEST_REPLAY_PATH'])",
                    "def replay_tripwire(_self, _workspace, **_kwargs):",
                    "    replay_path.write_text('provider replayed\\n', "
                    "encoding='utf-8')",
                    "    return [sys.executable, '-c', 'raise SystemExit(99)']",
                    "daemon_module.PortalImplementationDaemon."
                    "_build_implementation_command = replay_tripwire",
                    "gate_fd = int(os.environ['_ASE3_TEST_GATE_FD'])",
                    "os.read(gate_fd, 1)",
                    "os.close(gate_fd)",
                    "pid = os.fork()",
                    "if pid == 0:",
                    "    try:",
                    "        rc = supervisor_module."
                    "_run_plan_bound_daemon_child(json.loads("
                    "os.environ['_ASE3_TEST_HELPER_ARGV']))",
                    "    except BaseException as exc:",
                    "        Path(os.environ['_ASE3_TEST_RECOVERY_ERROR'])."
                    "write_text(f'{type(exc).__name__}: {exc}\\n', "
                    "encoding='utf-8')",
                    "        os._exit(97)",
                    "    os._exit(int(rc))",
                    "_pid, status = os.waitpid(pid, 0)",
                    "raise SystemExit(os.waitstatus_to_exitcode(status))",
                )
            )
            recovery_argv = (
                sys.executable,
                "-P",
                "-c",
                recovery_code,
                *plan_common_args,
                *recovery_track.extra_args,
            )
            recovery_state_root = (
                recovery_track.supervisor_pid_path.parent.resolve()
            )
            recovery_lifecycle_token = _test_lifecycle_token(
                tmp_path,
                f"merge-recovery-{wave_scenario}-{crashed_child.lane_id}",
            )
            recovery_profile = multi_runner_module.LifecycleProfile(
                target_id=f"supervisor-track:{crashed_child.name}",
                run_id=(
                    f"test-merge-recovery-{wave_scenario}-"
                    f"{recovery_lifecycle_token}"
                ),
                configuration_root=(
                    f"test-merge-recovery-{wave_scenario}-"
                    f"{recovery_lifecycle_token}"
                ),
                repository_root=str(repo.resolve()),
                state_root=str(recovery_state_root),
                run_root=str(
                    recovery_state_root
                    / "lifecycle-runs"
                    / f"{crashed_child.name}-{recovery_lifecycle_token}"
                ),
                argv=recovery_argv,
                cwd=str(repo.resolve()),
            )
            recovery_env = recovery_profile.launch_environment(0)
            recovery_env.update(
                {
                    "_ASE3_TEST_SOURCE_ROOT": str(REPO_ROOT),
                    "_ASE3_TEST_REPO": str(repo),
                    "_ASE3_TEST_LIFECYCLE_ROOT": str(
                        tmp_path / "local-profile-root-registry"
                    ),
                    "_ASE3_TEST_HOST": json.dumps(_host_capacity(lanes=2)),
                    "_ASE3_TEST_PROVIDERS": json.dumps(
                        _provider_capacity(lanes=2)
                    ),
                    "_ASE3_TEST_NOW_MS": str(PLAN_NOW),
                    "_ASE3_TEST_GATE_FD": str(gate_read),
                    "_ASE3_TEST_HELPER_ARGV": json.dumps(crashed_helper),
                    "_ASE3_TEST_RECOVERY_ERROR": str(recovery_error_path),
                    "_ASE3_TEST_REPLAY_PATH": str(replay_path),
                }
            )
            class CompletedOriginalTrack:
                """Expose the already-reaped original to the real runner."""

                pid = supervisors[crashed_index].pid

                @staticmethod
                def poll() -> int:
                    return 86

                @staticmethod
                def wait(*, timeout: float | None = None) -> int:
                    del timeout
                    return 86

            completed_original = CompletedOriginalTrack()
            recovery_processes: list[subprocess.Popen[bytes]] = []
            start_calls = 0
            original_terminate_managed = (
                multi_runner_module._terminate_managed_process
            )

            def runner_driven_start(_track, **_kwargs):
                nonlocal start_calls
                start_calls += 1
                if start_calls == 1:
                    return completed_original
                assert start_calls == 2, "runner restarted provider-bearing work"
                recovery_process = _spawn_test_process(
                    recovery_argv,
                    cwd=repo,
                    env=recovery_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=(
                        gate_read,
                        control_plane_launch.descriptor,
                    ),
                    start_new_session=True,
                )
                recovery_processes.append(recovery_process)
                recovery_identity = _capture_test_process_identity(
                    recovery_process,
                    recovery_profile,
                )
                multi_runner_module._persist_plan_bound_process_birth(
                    profile=recovery_profile,
                    process_identity=recovery_identity,
                    repo_root=repo,
                )
                os.write(gate_write, b"1")
                return recovery_process

            def runner_terminate(process, **kwargs):
                if process is completed_original or process.poll() is not None:
                    return True, []
                return original_terminate_managed(process, **kwargs)

            with monkeypatch.context() as runner_patch:
                runner_patch.setattr(
                    multi_runner_module,
                    "start_track",
                    runner_driven_start,
                )
                runner_patch.setattr(
                    multi_runner_module,
                    "_terminate_managed_process",
                    runner_terminate,
                )
                runner_result = multi_runner_module.run_supervisor_tracks(
                    (recovery_track,),
                    repo_root=repo,
                    common_args=plan_common_args,
                    duration_seconds=45.0,
                    heartbeat_interval_seconds=0.02,
                    stop_grace_seconds=0.2,
                    exit_when_all_tracks_terminal=True,
                    plan_bound_children=(crashed_child,),
                    accepted_control_plane_pin=control_plane_pin,
                    accepted_control_plane_descriptor=(
                        control_plane_launch.descriptor
                    ),
                    output=lambda _message: None,
                )
            os.close(gate_read)
            os.close(gate_write)
            assert start_calls == 2, json.dumps(
                runner_result,
                sort_keys=True,
                default=str,
            )
            assert len(recovery_processes) == 1
            recovery_process = recovery_processes[0]
            recovery_stdout, recovery_stderr = recovery_process.communicate(
                timeout=5
            )
            remaining_deadline = time.monotonic() + 45.0
            while child_pids and time.monotonic() < remaining_deadline:
                for pid, lane_id in tuple(child_pids.items()):
                    observed, status = os.waitpid(pid, os.WNOHANG)
                    if observed:
                        statuses[lane_id] = os.waitstatus_to_exitcode(status)
                        child_pids.pop(pid)
                if child_pids:
                    time.sleep(0.02)
            expected_recovery_returncode = (
                supervisor_module.PLAN_BOUND_REPLAN_RETURN_CODE
                if wave_scenario == "crash_after_enqueue_mismatch"
                else 0
            )
            assert recovery_process.returncode == expected_recovery_returncode, {
                "stdout": recovery_stdout.decode(errors="replace"),
                "stderr": recovery_stderr.decode(errors="replace"),
                "error": (
                    recovery_error_path.read_text(encoding="utf-8")
                    if recovery_error_path.exists()
                    else ""
                ),
            }
            assert runner_result["replan_required"] is (
                wave_scenario == "crash_after_enqueue_mismatch"
            )
            assert runner_result["terminal_quiescent"] is (
                wave_scenario != "crash_after_enqueue_mismatch"
            ), json.dumps(runner_result, sort_keys=True, default=str)
            assert not child_pids, (
                "surviving lane did not observe recovered predecessor terminality"
            )
            assert statuses == {
                child.lane_id: (
                    86
                    if child.task_ids == (crash_task_id,)
                    else (
                        supervisor_module.PLAN_BOUND_REPLAN_RETURN_CODE
                        if wave_scenario == "crash_after_enqueue_mismatch"
                        else 0
                    )
                )
                for child, _helper in helpers
            }
            assert not replay_path.exists()
            with store._thread_lock:
                with store._guard():
                    recovered_lease = execution_plan_module._load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=receipt.binding.revision_cid,
                        slice_id=crashed_child.slice_id,
                        lane_id=crashed_child.lane_id,
                    )
            assert recovered_lease is not None
            queue = daemon_module.MergeQueue(
                board.path(board.runtime_paths["merge_queue"])
            )
            with queue._connect() as connection:
                queue_row = connection.execute(
                    "SELECT * FROM merge_requests "
                    "WHERE task_id=?",
                    (crash_task_id,),
                ).fetchone()
            assert queue_row is not None
            with queue._connect() as connection:
                assert connection.execute(
                    "SELECT COUNT(*) AS count FROM merge_requests "
                    "WHERE task_id=?",
                    (crash_task_id,),
                ).fetchone()["count"] == 1
            if wave_scenario == "crash_after_enqueue_mismatch":
                assert recovered_lease[1].phase == "merge_enqueue_prepared"
                assert recovered_lease[1].merge_request_id == ""
                assert queue_row["status"] == "quarantined"
                assert queue_row["failure_count"] == 1
                assert queue_row["failure_reason"] == (
                    "plan_bound_merge_enqueue_mismatch"
                )
                assert not (repo / "src/test-a.py").exists()
                with store._thread_lock:
                    with store._guard():
                        terminal_failure = execution_plan_module._load_plan_bound_merge_terminal_failure_locked(
                            store,
                            revision_cid=receipt.binding.revision_cid,
                            slice_id=crashed_child.slice_id,
                        )
                assert terminal_failure is not None
                assert terminal_failure[1]["queue_status"] == "quarantined"
                assert terminal_failure[1]["request_id"] == queue_row[
                    "request_id"
                ]
                assert "merge_queue_intent_mismatch" in terminal_failure[1][
                    "reason_codes"
                ]
            else:
                assert recovered_lease[1].phase == "merge_completed"
                assert recovered_lease[1].merge_request_id
                assert recovered_lease[1].merge_queue_receipt_cid
                durable_request = queue.get(
                    recovered_lease[1].merge_request_id
                )
                assert durable_request is not None
                assert durable_request.task_id == crash_task_id
                assert durable_request.status == "completed"
                if wave_scenario == "crash_confirmed_retry":
                    assert durable_request.attempt == retry_transition["attempt"]
                    assert durable_request.failure_count == retry_transition[
                        "failure_count"
                    ]
                    assert durable_request.claim_generation >= (
                        retry_transition["claim_generation"] + 2
                    )
                    assert durable_request.enqueued_at == retry_transition[
                        "enqueued_at"
                    ]
                    assert queue_row["status"] == "completed"
                    assert queue_row["claimed_at"] == 0
                    assert queue_row["consumer_id"] == ""
                    assert queue_row["claim_token"] == ""
                    assert queue_row["retry_not_before"] == 0
                    assert queue_row["finished_at"] >= retry_transition[
                        "enqueued_at"
                    ]
                    assert queue_row["updated_at"] >= queue_row["finished_at"]
                else:
                    assert durable_request.failure_count == 0
                if crash_task_id == "TEST-A":
                    assert _git(repo, "show", "main:src/test-a.py").stdout == (
                        "VALUE = 'TEST-A'\n"
                    )
                elif wave_scenario == "crash_serialized_merge_confirmed":
                    assert _git(repo, "show", "main:src/test-b.py").stdout == (
                        "VALUE = 'TEST-B'\n"
                    )
                    assert _git(repo, "show", "main:src/test-a.py").stdout == (
                        "VALUE = 'TEST-A'\n"
                    )
                else:
                    assert _git(repo, "show", "main:src/test-b.py").stdout == (
                        "VALUE = 'already-present'\n"
                    )
                    assert _git(repo, "show", "main:src/test-a.py").stdout == (
                        "VALUE = 'TEST-A'\n"
                    )
                if wave_scenario == "crash_serialized_merge_confirmed":
                    train_receipt = (
                        board.path(board.runtime_paths["merge_queue"])
                        / "train"
                        / "receipts"
                        / f"{durable_request.dedupe_key}.json"
                    )
                    assert train_receipt.is_file()
                    receipt_payload = json.loads(
                        train_receipt.read_text(encoding="utf-8")
                    )
                    merge_result = receipt_payload.get("merge_result")
                    assert isinstance(merge_result, dict)
                    assert receipt_payload.get("commit_sha") == (
                        durable_request.commit_sha
                    )
                    integration_proof = merge_result.get(
                        "integration_commit_proof"
                    )
                    assert isinstance(integration_proof, dict)
                    assert integration_proof.get("passed") is True
                    assert integration_proof.get("implementation_commit") == (
                        durable_request.commit_sha
                    )
                    assert integration_proof.get("integration_commit") == (
                        merge_result.get("merge_commit")
                    )
                    final_head = _git(
                        repo,
                        "rev-parse",
                        "main",
                    ).stdout.strip()
                    assert receipt_payload.get("target_commit") == final_head
                    _git(
                        repo,
                        "merge-base",
                        "--is-ancestor",
                        str(integration_proof["integration_commit"]),
                        final_head,
                    )
            provider_invocations = [
                json.loads(line)["task_id"]
                for line in (
                    tmp_path / "provider-command-invocations.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line
            ]
            assert provider_invocations.count(crash_task_id) == 1
    finally:
        for pid in child_pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
        for process in supervisors:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        for workspace in board.path(board.runtime_paths["worktrees"]).glob(
            "workspace-*"
        ):
            if workspace.is_dir():
                _git(repo, "worktree", "remove", "--force", str(workspace))

    enqueue_rows = (
        [
            json.loads(line)
            for line in enqueue_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        if enqueue_receipt_path.exists()
        else []
    )
    expected_enqueues = {
        "mixed": 0,
        "disjoint": 2,
        "changed_no_change": 2,
        "compact_hidden_drift": 0,
        # The recovery process is a fresh interpreter, so this parent-local
        # enqueue probe records the surviving sibling plus only the original
        # pre-crash call.  Canonical queue cardinality is asserted above.
        "crash_proposal_ready": 1,
        "crash_before_enqueue": 1,
        "crash_after_enqueue": 2,
        "crash_confirmed": 2,
        "crash_confirmed_retry": 2,
        "crash_completed_before_finalize": 2,
        "crash_serialized_merge_confirmed": 2,
        "crash_no_change": 1,
        "crash_after_enqueue_mismatch": 1,
    }[wave_scenario]
    assert len(enqueue_rows) == expected_enqueues
    assert all(
        row == {
            "task_id": row["task_id"],
            "barrier_decision": "released",
            "disposition_count": 2,
        }
        for row in enqueue_rows
    )
    barrier = ProductionParallelPlanAdapter(
        PlanRevisionStore(repo / children[0].plan_revision_store_path)
    ).load_wave_diff_barrier(
        revision_cid=receipt.binding.revision_cid,
        slice_manifest_cid=receipt.slice_manifest_cid,
    )
    assert barrier is not None
    assert barrier[1].decision == (
        "rejected"
        if wave_scenario in {"mixed", "compact_hidden_drift"}
        else "released"
    )
    assert len(barrier[1].dispositions) == len(children) == 2
    store = PlanRevisionStore(repo / children[0].plan_revision_store_path)
    with store._thread_lock:
        with store._guard():
            dispositions = tuple(
                execution_plan_module._load_plan_bound_proposal_disposition_locked(
                    store,
                    revision_cid=receipt.binding.revision_cid,
                    slice_id=child.slice_id,
                )
                for child in children
            )
    assert all(item is not None for item in dispositions)
    records = {item[1].task_id: item[1] for item in dispositions if item}
    if wave_scenario == "mixed":
        assert records["TEST-A"].outcome == "rejected"
        assert "path_outside_scope" in records["TEST-A"].reason_codes
        assert records["TEST-B"].outcome == "changed"
        assert records["TEST-A"].actual_changed_paths == ("src/test-b.py",)
        assert records["TEST-B"].actual_changed_paths == ("src/test-b.py",)
    elif wave_scenario == "compact_hidden_drift":
        assert records["TEST-A"].outcome == "rejected"
        assert "path_outside_scope" in records["TEST-A"].reason_codes
        assert records["TEST-A"].actual_changed_paths == (
            "src/hidden.py",
            "src/test-a.py",
        )
        assert records["TEST-B"].outcome == "changed"
    elif wave_scenario in {"disjoint", "crash_serialized_merge_confirmed"}:
        assert {record.outcome for record in records.values()} == {"changed"}
        assert {
            record.actual_changed_paths for record in records.values()
        } == {("src/test-a.py",), ("src/test-b.py",)}
        queue = daemon_module.MergeQueue(
            board.path(board.runtime_paths["merge_queue"])
        )
        with queue._connect() as connection:
            completed_rows = connection.execute(
                "SELECT task_id, status, failure_count FROM merge_requests "
                "ORDER BY task_id"
            ).fetchall()
        assert [dict(row) for row in completed_rows] == [
            {"task_id": "TEST-A", "status": "completed", "failure_count": 0},
            {"task_id": "TEST-B", "status": "completed", "failure_count": 0},
        ]
        assert _git(repo, "show", "main:src/test-a.py").stdout == (
            "VALUE = 'TEST-A'\n"
        )
        assert _git(repo, "show", "main:src/test-b.py").stdout == (
            "VALUE = 'TEST-B'\n"
        )
    else:
        assert records["TEST-A"].outcome == "changed"
        assert records["TEST-A"].actual_changed_paths == ("src/test-a.py",)
        assert records["TEST-B"].outcome == "no_change"
        assert records["TEST-B"].actual_changed_paths == ()


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
    profile = process._agent_supervisor_lifecycle_profile
    process_identity = process._agent_supervisor_process_identity
    original_stat = multi_runner_module.LinuxProcessAdapter._stat
    original_environ = multi_runner_module.LinuxProcessAdapter._environ

    def attempt_reassignment():
        return multi_runner_module.reassign_fenced_plan_bound_child(
            donor=donor,
            recipient=recipient,
            donor_process=process,
            repo_root=repo,
        )

    with monkeypatch.context() as inspection_failure:
        inspection_failure.setattr(
            multi_runner_module.LinuxProcessAdapter,
            "_stat",
            staticmethod(
                lambda pid: (
                    (
                        os.getppid(),
                        process_identity.process_group_id,
                        process_identity.session_id,
                        1,
                    )
                    if pid == os.getpid()
                    else original_stat(pid)
                )
            ),
        )

        def unreadable_potential_member(pid):
            if pid == os.getpid():
                raise PermissionError("simulated /proc permission denial")
            return original_environ(pid)

        inspection_failure.setattr(
            multi_runner_module.LinuxProcessAdapter,
            "_environ",
            staticmethod(unreadable_potential_member),
        )
        process_state, evidence = (
            multi_runner_module._strict_plan_bound_process_fence_observation(
                profile,
                process_identity,
            )
        )
        assert process_state == "unknown"
        assert evidence is None
        with pytest.raises(
            ExecutionClaimConflictError,
            match="death is not provable",
        ):
            attempt_reassignment()
        assert ProductionParallelPlanAdapter(
            PlanRevisionStore(repo / donor.plan_revision_store_path)
        ).load_slice_reassignment(
            revision_cid=donor.revision_cid,
            slice_id=donor.slice_id,
        ) is None

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
        lambda _board, **_kwargs: (
            _host_capacity(),
            _provider_capacity(),
            PLAN_NOW,
        ),
    )
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "run_once",
        bounded_run_once,
    )
    marker = supervisor_module.PLAN_BOUND_DAEMON_CHILD_MARKER
    helper_argv = command[command.index(marker) + 1 :]
    adopted_track = adopted_child.track(
        stamp="20260808T-reassigned-recipient"
    ).resolve(repo)
    adopted_launch_argv = (
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        *adopted_track.extra_args,
    )
    adopted_state_root = adopted_track.supervisor_pid_path.parent.resolve()
    lifecycle_token = _test_lifecycle_token(
        tmp_path,
        "reassigned-recipient-adoption",
    )
    adopted_profile = multi_runner_module.LifecycleProfile(
        target_id=f"supervisor-track:{adopted_child.name}",
        run_id=f"test-reassigned-recipient-adoption-{lifecycle_token}",
        configuration_root=(
            f"test-reassigned-recipient-adoption-{lifecycle_token}"
        ),
        repository_root=str(repo.resolve()),
        state_root=str(adopted_state_root),
        run_root=str(
            adopted_state_root
            / "lifecycle-runs"
            / f"{adopted_child.name}-{lifecycle_token}"
        ),
        argv=adopted_launch_argv,
        cwd=str(repo.resolve()),
    )
    adopted_process = _spawn_test_process(
        adopted_launch_argv,
        cwd=repo,
        env=adopted_profile.launch_environment(0),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    adopted_identity = _capture_test_process_identity(
        adopted_process,
        adopted_profile,
    )
    multi_runner_module._persist_plan_bound_process_birth(
        profile=adopted_profile,
        process_identity=adopted_identity,
        repo_root=repo,
    )
    try:
        assert supervisor_module._run_plan_bound_daemon_child(helper_argv) == 0
    finally:
        adopted_process.terminate()
        adopted_process.wait(timeout=5)
    assert captured["decision"] is None
    claim = captured["claim"]
    assert isinstance(claim, dict)
    assert claim["plan_revision_cid"] == receipt.binding.revision_cid
    assert claim["compiled_claim_acquired_before_publish"] is True

    # Prove scheduler adoption follows the authoritative reassignment owner,
    # not the immutable slice's original lane.  The real drift publication is
    # covered above at the daemon boundary; this chain isolates the owner-read
    # join by advancing the recipient's canonical execution lease generations.
    current_execution = adapter.load_execution_lease(
        revision_cid=adopted_child.revision_cid,
        slice_id=adopted_child.slice_id,
        lane_id=adopted_child.lane_id,
    )
    assert current_execution is not None
    assert current_execution[1].phase == "reserved"
    task_id = adopted_child.task_ids[0]
    task_cid = adopted_child.task_cids[0]
    claim_path = repo / "recipient-authority-claim.json"
    workspace_path = repo / "recipient-authority-worktree"
    lifecycle_path = repo / "recipient-authority-lifecycle.json"
    claimed = replace(
        current_execution[1],
        generation=current_execution[1].generation + 1,
        phase="claimed",
        prior_execution_lease_cid=current_execution[0],
        active_task_id=task_id,
        active_task_cid=task_cid,
        daemon_process_birth={"pid": os.getpid()},
        canonical_claim_path=str(claim_path),
        canonical_claim_cid="sha256:" + "1" * 64,
        canonical_claim_lease_id=str(
            current_execution[1].assignment_for(task_id, task_cid)["lease_id"]
        ),
    )
    with store._thread_lock:
        with store._guard():
            claimed_cid = (
                execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    claimed,
                    expected_current_cid=current_execution[0],
                )
            )
            prepared = replace(
                claimed,
                generation=claimed.generation + 1,
                phase="workspace_prepared",
                prior_execution_lease_cid=claimed_cid,
                workspace_lifecycle_path=str(lifecycle_path),
                workspace_lifecycle_cid="sha256:" + "2" * 64,
                workspace_record_id="workspace-record:recipient",
                workspace_path=str(workspace_path),
                workspace_lease_id="workspace-lease:recipient",
                workspace_fence=1,
            )
            prepared_cid = (
                execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    prepared,
                    expected_current_cid=claimed_cid,
                )
            )
            provider_ready = replace(
                prepared,
                generation=prepared.generation + 1,
                phase="provider_ready",
                prior_execution_lease_cid=prepared_cid,
                provider_ready=True,
            )
            provider_cid = (
                execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    provider_ready,
                    expected_current_cid=prepared_cid,
                )
            )
            recipient_drift = replace(
                provider_ready,
                generation=provider_ready.generation + 1,
                phase="scope_drift",
                prior_execution_lease_cid=provider_cid,
                proposal_id="proposal:recipient-owner-drift",
                proposal_receipt_id="receipt:recipient-owner-drift",
                proposal_reason_codes=("path_outside_scope",),
                actual_changed_paths=("src/recipient-owner-drift.py",),
            )
            execution_plan_module._publish_plan_bound_execution_lease_locked(
                store,
                recipient_drift,
                expected_current_cid=provider_cid,
            )

    replacement = materialize_configured_board_execution_plan(
        board,
        now_ms=PLAN_NOW,
        host_capacity_snapshot=_host_capacity(),
        provider_capacity_snapshots=_provider_capacity(),
        task_state_snapshots=(),
    )
    assert replacement is not None
    assert replacement.binding.revision_cid != receipt.binding.revision_cid
    replacement_revision = store.load_revision(replacement.binding.revision_cid)
    assert replacement_revision.origin.value == "steer"
    assert "src/recipient-owner-drift.py" in (
        replacement_revision.conflict_contract.predicted_files
    )


def test_unknown_process_fence_denies_transfer_and_deadline_has_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UNKNOWN is never DEAD; bounded contenders adopt one missing decision."""

    repo, _board, receipt, donor, recipient, process = _fenced_plan_children(
        tmp_path
    )
    profile = process._agent_supervisor_lifecycle_profile
    process_identity = process._agent_supervisor_process_identity
    original_stat = multi_runner_module.LinuxProcessAdapter._stat
    original_environ = multi_runner_module.LinuxProcessAdapter._environ

    with monkeypatch.context() as inspection_failure:
        inspection_failure.setattr(
            multi_runner_module.LinuxProcessAdapter,
            "_stat",
            staticmethod(
                lambda pid: (
                    (
                        os.getppid(),
                        process_identity.process_group_id,
                        process_identity.session_id,
                        1,
                    )
                    if pid == os.getpid()
                    else original_stat(pid)
                )
            ),
        )

        def unreadable_potential_member(pid):
            if pid == os.getpid():
                raise PermissionError("simulated /proc permission denial")
            return original_environ(pid)

        inspection_failure.setattr(
            multi_runner_module.LinuxProcessAdapter,
            "_environ",
            staticmethod(unreadable_potential_member),
        )
        state, evidence = (
            multi_runner_module._strict_plan_bound_process_fence_observation(
                profile,
                process_identity,
            )
        )
        assert state == "unknown"
        assert evidence is None
        with pytest.raises(
            ExecutionClaimConflictError,
            match="death is not provable",
        ):
            multi_runner_module.reassign_fenced_plan_bound_child(
                donor=donor,
                recipient=recipient,
                donor_process=process,
                repo_root=repo,
            )

    store = PlanRevisionStore(repo / donor.plan_revision_store_path)
    adapter = ProductionParallelPlanAdapter(store)
    assert adapter.load_slice_reassignment(
        revision_cid=donor.revision_cid,
        slice_id=donor.slice_id,
    ) is None

    def await_missing():
        return adapter.await_wave_diff_barrier(
            revision_cid=donor.revision_cid,
            slice_manifest_cid=donor.slice_manifest_cid,
            timeout_ms=50,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        decisions = tuple(
            future.result()
            for future in (executor.submit(await_missing) for _ in range(2))
        )
    assert decisions[0] == decisions[1]
    barrier_cid, barrier = decisions[0]
    assert barrier.decision == "missing"
    assert barrier.terminal_missing == ()
    assert barrier.dispositions == ()
    assert set(barrier.missing_slice_ids) == {
        item.slice_id for item in receipt.slice_manifest.nonempty
    }
    assert adapter.load_wave_diff_barrier(
        revision_cid=donor.revision_cid,
        slice_manifest_cid=donor.slice_manifest_cid,
    ) == (barrier_cid, barrier)


@pytest.mark.parametrize(
    ("race_winner", "publisher_delay", "deadline_delay"),
    (
        ("complete", 0.0, 0.02),
        ("deadline", 0.02, 0.0),
    ),
)
def test_complete_disposition_and_deadline_contenders_have_one_decision(
    tmp_path: Path,
    race_winner: str,
    publisher_delay: float,
    deadline_delay: float,
) -> None:
    """A completion/deadline race cannot replace its first durable outcome."""

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
    launch = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        stamp="20260808T-deadline-race",
        parallelism_receipt=receipt,
    )
    argv = list(launch["argv"])
    record = next(
        argv[index + 1]
        for index, token in enumerate(argv[:-1])
        if token == "--implementation-plan-bound-track"
    )
    child = multi_runner_module.PlanBoundSupervisorChild.from_cli_record(
        record
    )
    track = child.track().resolve(repo)
    command = (
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        *track.extra_args,
    )
    state_root = track.supervisor_pid_path.parent.resolve()
    lifecycle_token = _test_lifecycle_token(
        tmp_path,
        f"wave-deadline-race-{race_winner}",
    )
    profile = multi_runner_module.LifecycleProfile(
        target_id=f"supervisor-track:{child.name}",
        run_id=f"test-wave-deadline-race-{race_winner}-{lifecycle_token}",
        configuration_root=(
            f"test-wave-deadline-race-{race_winner}-{lifecycle_token}"
        ),
        repository_root=str(repo.resolve()),
        state_root=str(state_root),
        run_root=str(
            state_root / "lifecycle-runs" / f"{child.name}-{lifecycle_token}"
        ),
        argv=command,
        cwd=str(repo.resolve()),
    )
    process = _spawn_test_process(
        command,
        cwd=repo,
        env=profile.launch_environment(0),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    identity = _capture_test_process_identity(process, profile)
    multi_runner_module._persist_plan_bound_process_birth(
        profile=profile,
        process_identity=identity,
        repo_root=repo,
    )
    store = PlanRevisionStore(repo / child.plan_revision_store_path)
    adapter = ProductionParallelPlanAdapter(store)
    try:
        with store._thread_lock:
            with store._guard():
                reserved = execution_plan_module._load_plan_bound_execution_lease_locked(
                    store,
                    revision_cid=child.revision_cid,
                    slice_id=child.slice_id,
                    lane_id=child.lane_id,
                )
                assert reserved is not None
                current_cid, current = reserved
                claimed = replace(
                    current,
                    generation=current.generation + 1,
                    phase="claimed",
                    prior_execution_lease_cid=current_cid,
                    active_task_id=child.task_ids[0],
                    active_task_cid=child.task_cids[0],
                    daemon_process_birth={"pid": process.pid},
                    canonical_claim_path=str(repo / "deadline-race-claim.json"),
                    canonical_claim_cid="sha256:" + "1" * 64,
                    canonical_claim_lease_id=str(
                        current.assignment_for(
                            child.task_ids[0], child.task_cids[0]
                        )["lease_id"]
                    ),
                )
                current_cid = execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    claimed,
                    expected_current_cid=current_cid,
                )
                prepared = replace(
                    claimed,
                    generation=claimed.generation + 1,
                    phase="workspace_prepared",
                    prior_execution_lease_cid=current_cid,
                    workspace_lifecycle_path=str(
                        repo / "deadline-race-lifecycle.json"
                    ),
                    workspace_lifecycle_cid="sha256:" + "2" * 64,
                    workspace_record_id="workspace-record:deadline-race",
                    workspace_path=str(repo / "deadline-race-worktree"),
                    workspace_lease_id="workspace-lease:deadline-race",
                    workspace_fence=1,
                )
                current_cid = execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    prepared,
                    expected_current_cid=current_cid,
                )
                provider_ready = replace(
                    prepared,
                    generation=prepared.generation + 1,
                    phase="provider_ready",
                    prior_execution_lease_cid=current_cid,
                    provider_ready=True,
                )
                current_cid = execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    provider_ready,
                    expected_current_cid=current_cid,
                )
                baseline_ref = receipt.slice_manifest.source_head
                branch_name = "implementation/deadline-race"
                enqueue_fields = {
                    "branch_name": branch_name,
                    "task_id": child.task_ids[0],
                    "priority": "normal",
                    "lane_id": child.lane_id,
                    "attempt": 1,
                    "metadata": {
                        "baseline_ref": baseline_ref,
                        "implementation_commit": baseline_ref,
                    },
                    "commit_sha": baseline_ref,
                    "canonical_task_id": child.task_cids[0],
                    "canonical_task_key": "task/v1/deadline-race",
                    "canonical_task_cid": child.task_cids[0],
                    "target_repository_id": str(repo.resolve()),
                    "target_branch": board.merge_target_branch,
                }
                proposal_handoff = {
                    "schema": execution_plan_module.PLAN_BOUND_PROPOSAL_HANDOFF_SCHEMA,
                    "revision_cid": provider_ready.revision_cid,
                    "plan_root_cid": provider_ready.plan_root_cid,
                    "execution_plan_cid": provider_ready.execution_plan_cid,
                    "capacity_snapshot_id": provider_ready.capacity_snapshot_id,
                    "slice_manifest_cid": provider_ready.slice_manifest_cid,
                    "slice_id": provider_ready.slice_id,
                    "lane_id": provider_ready.lane_id,
                    "reassignment_cid": provider_ready.reassignment_cid,
                    "task_id": provider_ready.active_task_id,
                    "task_cid": provider_ready.active_task_cid,
                    "source_execution_lease_cid": current_cid,
                    "process_birth_cid": provider_ready.process_birth_cid,
                    "canonical_claim_cid": provider_ready.canonical_claim_cid,
                    "canonical_claim_lease_id": (
                        provider_ready.canonical_claim_lease_id
                    ),
                    "workspace_lifecycle_cid": (
                        provider_ready.workspace_lifecycle_cid
                    ),
                    "workspace_record_id": provider_ready.workspace_record_id,
                    "workspace_path": provider_ready.workspace_path,
                    "workspace_lease_id": provider_ready.workspace_lease_id,
                    "workspace_fence": provider_ready.workspace_fence,
                    "attempt": 1,
                    "branch_name": branch_name,
                    "baseline_ref": baseline_ref,
                    "implementation_commit": baseline_ref,
                    "actual_changed_paths": [],
                    "outcome": "no_change",
                    "enqueue_fields": enqueue_fields,
                    "enqueue_fields_cid": execution_plan_module.content_identity(
                        enqueue_fields
                    ),
                    "created_at_ms": int(time.time() * 1000),
                }
                proposal_handoff_cid = store.put_cas(proposal_handoff)
                proposal_ready = replace(
                    provider_ready,
                    generation=provider_ready.generation + 1,
                    phase="proposal_ready",
                    prior_execution_lease_cid=current_cid,
                    proposal_handoff_cid=proposal_handoff_cid,
                )
                proposal_cid = execution_plan_module._publish_plan_bound_execution_lease_locked(
                    store,
                    proposal_ready,
                    expected_current_cid=current_cid,
                )
                disposition = execution_plan_module.PlanBoundProposalDisposition(
                    revision_cid=child.revision_cid,
                    plan_root_cid=child.plan_root_cid,
                    execution_plan_cid=child.execution_plan_cid,
                    capacity_snapshot_id=child.capacity_snapshot_id,
                    slice_manifest_cid=child.slice_manifest_cid,
                    slice_id=child.slice_id,
                    lane_id=child.lane_id,
                    reassignment_cid="",
                    task_id=child.task_ids[0],
                    task_cid=child.task_cids[0],
                    execution_lease_cid=proposal_cid,
                    process_birth_cid=proposal_ready.process_birth_cid,
                    proposal_id="",
                    proposal_receipt_id="",
                    outcome="no_change",
                    reason_codes=(),
                    actual_changed_paths=(),
                    baseline_ref=baseline_ref,
                    implementation_commit=baseline_ref,
                )
                started_at_ms = int(time.time() * 1000)
                assert adapter._evaluate_wave_diff_barrier_locked(
                    revision_cid=child.revision_cid,
                    slice_manifest_cid=child.slice_manifest_cid,
                    timeout_ms=50,
                    now_ms=started_at_ms,
                ) is None

        gate = threading.Barrier(2)

        def publish_completion():
            gate.wait(timeout=2)
            time.sleep(publisher_delay)
            try:
                return ("published", adapter.publish_proposal_disposition(disposition))
            except ExecutionPlanError as exc:
                return ("rejected", str(exc))

        def publish_deadline():
            gate.wait(timeout=2)
            time.sleep(deadline_delay)
            with store._thread_lock:
                with store._guard():
                    return adapter._evaluate_wave_diff_barrier_locked(
                        revision_cid=child.revision_cid,
                        slice_manifest_cid=child.slice_manifest_cid,
                        timeout_ms=50,
                        now_ms=started_at_ms + 51,
                    )

        with ThreadPoolExecutor(max_workers=2) as executor:
            completion_future = executor.submit(publish_completion)
            deadline_future = executor.submit(publish_deadline)
            completion_result = completion_future.result()
            deadline_result = deadline_future.result()
        assert deadline_result is not None
        barrier = adapter.load_wave_diff_barrier(
            revision_cid=child.revision_cid,
            slice_manifest_cid=child.slice_manifest_cid,
        )
        assert barrier == deadline_result
        if race_winner == "complete":
            assert completion_result[0] == "published"
            assert barrier[1].decision == "released"
            assert len(barrier[1].dispositions) == 1
        else:
            assert completion_result[0] == "rejected"
            assert "terminal wave barrier" in completion_result[1]
            assert barrier[1].decision == "missing"
            assert barrier[1].dispositions == ()
            assert adapter.load_execution_lease(
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
            ) == (proposal_cid, proposal_ready)
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize("exit_code", (0, 9))
def test_runner_terminalizes_every_exited_slice_without_a_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    """A bounded child cannot exit successfully or fail without wave evidence."""

    repo, _board, receipt, donor, _recipient, seeded_process = (
        _fenced_plan_children(tmp_path)
    )
    assert seeded_process.poll() is not None
    track = donor.track(stamp="20260808T-terminal-missing").resolve(repo)
    read_gate, write_gate = os.pipe()
    command = (
        sys.executable,
        "-c",
        (
            "import os,sys; os.read(int(sys.argv[1]), 1); "
            "raise SystemExit(int(sys.argv[2]))"
        ),
        str(read_gate),
        str(exit_code),
        *track.extra_args,
    )
    state_root = track.supervisor_pid_path.parent.resolve()
    lifecycle_token = _test_lifecycle_token(
        tmp_path,
        f"terminal-missing-{exit_code}",
    )
    profile = multi_runner_module.LifecycleProfile(
        target_id=f"supervisor-track:{donor.name}",
        run_id=f"test-terminal-missing-{exit_code}-{lifecycle_token}",
        configuration_root=(
            f"test-terminal-missing-{exit_code}-{lifecycle_token}"
        ),
        repository_root=str(repo.resolve()),
        state_root=str(state_root),
        run_root=str(
            state_root / "lifecycle-runs" / f"{donor.name}-{lifecycle_token}"
        ),
        argv=command,
        cwd=str(repo.resolve()),
    )
    process = _spawn_test_process(
        command,
        cwd=repo,
        env=profile.launch_environment(0),
        pass_fds=(read_gate,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    os.close(read_gate)
    try:
        identity = _capture_test_process_identity(process, profile)
        process._agent_supervisor_lifecycle_profile = profile
        process._agent_supervisor_process_identity = identity
        process._agent_supervisor_process_birth_cid = (
            multi_runner_module._persist_plan_bound_process_birth(
                profile=profile,
                process_identity=identity,
                repo_root=repo,
            )
        )
        os.write(write_gate, b"x")
    finally:
        os.close(write_gate)
    assert process.wait(timeout=5) == exit_code

    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=receipt.slice_manifest.source_head,
        source_tree=receipt.slice_manifest.repository_tree_id,
    )
    start_calls = 0

    def start_missing_track(*_args, **_kwargs):
        nonlocal start_calls
        start_calls += 1
        assert start_calls == 1, (
            "runner restarted a terminal-missing provider-bearing slice"
        )
        return process

    monkeypatch.setattr(
        multi_runner_module,
        "start_track",
        start_missing_track,
    )
    if exit_code:
        monkeypatch.setattr(
            multi_runner_module,
            "reassign_fenced_plan_bound_child",
            lambda **_kwargs: (_ for _ in ()).throw(
                ExecutionClaimConflictError("test exhausted recovery")
            ),
        )
    result = multi_runner_module.run_supervisor_tracks(
        (track,),
        repo_root=repo,
        common_args=(),
        duration_seconds=0.25,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.01,
        exit_when_all_tracks_terminal=True,
        plan_bound_children=(donor,),
        accepted_control_plane_pin=control_plane_pin,
        accepted_control_plane_descriptor=control_plane_launch.descriptor,
        output=lambda _message: None,
    )
    assert result["completed"] is False
    assert start_calls == 1
    assert result["replan_required"] is True
    assert result["all_trees_fenced"] is True
    assert len(result["scope_drift_receipts"]) == 1
    assert result["scope_drift_receipts"][0]["decision"] == "missing"
    store = PlanRevisionStore(repo / donor.plan_revision_store_path)
    adapter = ProductionParallelPlanAdapter(store)
    terminal = adapter.load_terminal_missing(
        revision_cid=donor.revision_cid,
        slice_id=donor.slice_id,
    )
    assert terminal is not None
    assert terminal[1].exit_code == exit_code
    barrier = adapter.load_wave_diff_barrier(
        revision_cid=donor.revision_cid,
        slice_manifest_cid=donor.slice_manifest_cid,
    )
    assert barrier is not None
    assert barrier[1].decision == "missing"


def test_reassignment_uses_the_immutable_global_wave_budget(
    tmp_path: Path,
) -> None:
    """A two-slice wave permits only two total same-revision transfers."""

    repo, _board, receipt, donor, recipient, donor_process = (
        _fenced_plan_children(tmp_path)
    )

    def exited_owner(
        child: multi_runner_module.PlanBoundSupervisorChild,
        ordinal: int,
    ) -> subprocess.Popen[bytes]:
        track = child.track(
            stamp=f"20260808T-budget-{ordinal}"
        ).resolve(repo)
        command = (
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            *track.extra_args,
        )
        state_root = track.supervisor_pid_path.parent.resolve()
        lifecycle_token = _test_lifecycle_token(
            tmp_path,
            f"wave-transfer-budget-{ordinal}-{child.lane_id}",
        )
        profile = multi_runner_module.LifecycleProfile(
            target_id=f"supervisor-track:{child.name}",
            run_id=(
                f"test-wave-transfer-budget-{ordinal}-{lifecycle_token}"
            ),
            configuration_root=(
                f"test-wave-transfer-budget-{ordinal}-{lifecycle_token}"
            ),
            repository_root=str(repo.resolve()),
            state_root=str(state_root),
            run_root=str(
                state_root
                / "lifecycle-runs"
                / f"{child.name}-{lifecycle_token}"
            ),
            argv=command,
            cwd=str(repo.resolve()),
        )
        process = _spawn_test_process(
            command,
            cwd=repo,
            env=profile.launch_environment(ordinal),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        identity = _capture_test_process_identity(process, profile)
        process._agent_supervisor_lifecycle_profile = profile
        process._agent_supervisor_process_identity = identity
        process._agent_supervisor_process_birth_cid = (
            multi_runner_module._persist_plan_bound_process_birth(
                profile=profile,
                process_identity=identity,
                repo_root=repo,
            )
        )
        fenced, _members = multi_runner_module._terminate_managed_process(
            process,
            grace_seconds=1,
        )
        assert fenced is True
        process.wait(timeout=5)
        return process

    first = multi_runner_module.reassign_fenced_plan_bound_child(
        donor=donor,
        recipient=recipient,
        donor_process=donor_process,
        repo_root=repo,
    )
    second_recipient = replace(
        first,
        name="test-recovery-lane-2",
        state_dir=str(Path(first.state_dir).parent / "test-recovery-lane-2"),
        state_prefix="test_recovery_lane_2",
        lane_id="test-recovery-lane-2",
    )
    first_process = exited_owner(first, 1)
    second = multi_runner_module.reassign_fenced_plan_bound_child(
        donor=first,
        recipient=second_recipient,
        donor_process=first_process,
        repo_root=repo,
    )
    third_recipient = replace(
        second,
        name="test-recovery-lane-3",
        state_dir=str(Path(second.state_dir).parent / "test-recovery-lane-3"),
        state_prefix="test_recovery_lane_3",
        lane_id="test-recovery-lane-3",
    )
    second_process = exited_owner(second, 2)
    with pytest.raises(
        ExecutionClaimConflictError,
        match="wave reassignment budget is exhausted",
    ):
        multi_runner_module.reassign_fenced_plan_bound_child(
            donor=second,
            recipient=third_recipient,
            donor_process=second_process,
            repo_root=repo,
        )
    current = ProductionParallelPlanAdapter(
        PlanRevisionStore(repo / donor.plan_revision_store_path)
    ).load_slice_reassignment(
        revision_cid=receipt.binding.revision_cid,
        slice_id=donor.slice_id,
    )
    assert current is not None
    assert current[1].generation == 2
    assert current[1].recipient_lane_id == second.lane_id


def test_runner_bounds_process_birth_chain_and_concurrent_append_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated gated recovery deaths consume one global finite birth budget."""

    repo, _board, receipt, donor, _peer, seeded_process = (
        _fenced_plan_children(tmp_path)
    )
    proposal_cid, proposal_ready = _publish_test_no_change_disposition(
        repo=repo,
        child=donor,
    )
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=receipt.slice_manifest.source_head,
        source_tree=receipt.slice_manifest.repository_tree_id,
    )
    track = donor.track(stamp="20260808T-birth-budget").resolve(repo)
    start_count = 0
    concurrent_results: tuple[str, str] | None = None
    spawned: list[subprocess.Popen[bytes]] = []
    provider_calls = 0

    def provider_tripwire(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("recovery birth must not replay a provider")

    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_evaluate_pre_implementation_provider_gate",
        provider_tripwire,
    )

    def controlled_start(
        current_track: multi_runner_module.SupervisorTrack,
        **_kwargs,
    ) -> subprocess.Popen[bytes]:
        nonlocal start_count, concurrent_results
        start_count += 1
        if start_count == 1:
            return seeded_process
        generation = start_count - 1
        assert 1 <= generation <= (
            execution_plan_module.MAX_PLAN_BOUND_WAVE_TRANSFERS
        )
        command = (
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            *current_track.extra_args,
        )
        state_root = current_track.supervisor_pid_path.parent.resolve()
        token = hashlib.sha256(
            f"{tmp_path.resolve()}:{generation}".encode("utf-8")
        ).hexdigest()[:16]
        profile = multi_runner_module.LifecycleProfile(
            target_id=f"supervisor-track:{current_track.name}",
            run_id=f"test-birth-budget-{token}",
            configuration_root=f"test-birth-budget-{token}",
            repository_root=str(repo.resolve()),
            state_root=str(state_root),
            run_root=str(
                state_root
                / "lifecycle-runs"
                / f"{current_track.name}-{token}"
            ),
            argv=command,
            cwd=str(repo.resolve()),
        )
        process = _spawn_test_process(
            command,
            cwd=repo,
            env=profile.launch_environment(generation),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        spawned.append(process)
        identity = _capture_test_process_identity(process, profile)

        def persist() -> str:
            return multi_runner_module._persist_plan_bound_process_birth(
                profile=profile,
                process_identity=identity,
                repo_root=repo,
            )

        if generation == 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = (executor.submit(persist), executor.submit(persist))
                concurrent_results = tuple(
                    future.result() for future in futures
                )
            assert concurrent_results[0] == concurrent_results[1]
            birth_cid = concurrent_results[0]
        else:
            birth_cid = persist()
        process._agent_supervisor_lifecycle_profile = profile
        process._agent_supervisor_process_identity = identity
        process._agent_supervisor_process_birth_cid = birth_cid
        fenced, _members = multi_runner_module._terminate_managed_process(
            process,
            grace_seconds=1,
        )
        assert fenced is True
        process.wait(timeout=5)
        return process

    monkeypatch.setattr(multi_runner_module, "start_track", controlled_start)
    try:
        result = multi_runner_module.run_supervisor_tracks(
            (track,),
            repo_root=repo,
            common_args=(),
            duration_seconds=10,
            heartbeat_interval_seconds=0.001,
            stop_grace_seconds=0.01,
            exit_when_all_tracks_terminal=True,
            plan_bound_children=(donor,),
            accepted_control_plane_pin=control_plane_pin,
            accepted_control_plane_descriptor=(
                control_plane_launch.descriptor
            ),
            output=lambda _message: None,
        )
    finally:
        for process in spawned:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

    assert concurrent_results is not None
    assert start_count == (
        execution_plan_module.MAX_PLAN_BOUND_WAVE_TRANSFERS + 1
    )
    assert provider_calls == 0
    assert result["completed"] is False
    assert result["replan_required"] is True
    assert result["all_trees_fenced"] is True
    assert len(result["scope_drift_receipts"]) == 1
    assert result["scope_drift_receipts"][0]["kind"] == (
        "process_birth_budget_exhausted"
    )
    assert result["scope_drift_receipts"][0]["decision"] == "missing"

    store = PlanRevisionStore(repo / donor.plan_revision_store_path)
    with store._thread_lock:
        with store._guard():
            chain = (
                execution_plan_module._load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                    lane_id=donor.lane_id,
                )
            )
            assert chain is not None
            assert chain[1].generation == (
                execution_plan_module.MAX_PLAN_BOUND_WAVE_TRANSFERS
            )
            assert len(chain[2]) == (
                execution_plan_module.MAX_PLAN_BOUND_WAVE_TRANSFERS + 1
            )
            exhausted = (
                execution_plan_module._load_plan_bound_process_birth_exhausted_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                )
            )
            assert exhausted is not None
            assert exhausted[1].generation == (
                execution_plan_module.MAX_PLAN_BOUND_WAVE_TRANSFERS
            )
    assert ProductionParallelPlanAdapter(store).load_execution_lease(
        revision_cid=donor.revision_cid,
        slice_id=donor.slice_id,
        lane_id=donor.lane_id,
    ) == (proposal_cid, proposal_ready)


def test_process_birth_chain_rejects_rollback_missing_cycle_and_identity_drift(
    tmp_path: Path,
) -> None:
    """The guarded birth loader rejects every hidden or malformed predecessor."""

    repo, _board, _receipt, donor, _peer, seeded_process = (
        _fenced_plan_children(tmp_path)
    )
    track = donor.track(stamp="20260808T-birth-chain-tamper").resolve(repo)
    command = (
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        *track.extra_args,
    )
    token = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:16]
    state_root = track.supervisor_pid_path.parent.resolve()
    profile = multi_runner_module.LifecycleProfile(
        target_id=f"supervisor-track:{donor.name}",
        run_id=f"test-birth-chain-{token}",
        configuration_root=f"test-birth-chain-{token}",
        repository_root=str(repo.resolve()),
        state_root=str(state_root),
        run_root=str(state_root / "lifecycle-runs" / f"{donor.name}-{token}"),
        argv=command,
        cwd=str(repo.resolve()),
    )
    process = _spawn_test_process(
        command,
        cwd=repo,
        env=profile.launch_environment(1),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = _capture_test_process_identity(process, profile)
        head_cid = multi_runner_module._persist_plan_bound_process_birth(
            profile=profile,
            process_identity=identity,
            repo_root=repo,
        )
        process._agent_supervisor_lifecycle_profile = profile
        process._agent_supervisor_process_identity = identity
        process._agent_supervisor_process_birth_cid = head_cid
        fenced, _members = multi_runner_module._terminate_managed_process(
            process,
            grace_seconds=1,
        )
        assert fenced is True
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    store = PlanRevisionStore(repo / donor.plan_revision_store_path)
    key = execution_plan_module.plan_bound_process_birth_key(
        donor.revision_cid,
        donor.slice_id,
        donor.lane_id,
    )
    with store._thread_lock:
        with store._guard():
            chain = (
                execution_plan_module._load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                    lane_id=donor.lane_id,
                )
            )
            assert chain is not None
            assert chain[0] == head_cid
            assert chain[1].generation == 1
            root_cid, root = chain[2][-1]
            head = chain[1]
            head_pointer = {
                "phase": "committed",
                "operation": "plan_bound_process_birth",
                "revision_cid": donor.revision_cid,
                "slice_id": donor.slice_id,
                "lane_id": donor.lane_id,
                "process_birth_cid": head_cid,
                "generation": 1,
                "global_budget": (
                    execution_plan_module.MAX_PLAN_BOUND_WAVE_TRANSFERS
                ),
            }

            # An older valid pointer cannot hide the later immutable CAS row.
            store.put_continuation(
                key,
                {
                    **head_pointer,
                    "process_birth_cid": root_cid,
                    "generation": 0,
                },
            )
            with pytest.raises(ExecutionPlanError, match="rolled back|forked"):
                execution_plan_module._load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                    lane_id=donor.lane_id,
                )
            store.put_continuation(key, head_pointer)

            # A linked generation whose predecessor is absent fails closed.
            missing = replace(
                head,
                generation=2,
                prior_process_birth_cid="sha256:" + "f" * 64,
            )
            missing_cid = store.put_cas(missing.to_dict())
            store.put_continuation(
                key,
                {
                    **head_pointer,
                    "process_birth_cid": missing_cid,
                    "generation": 2,
                },
            )
            with pytest.raises(ExecutionPlanError):
                execution_plan_module._load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                    lane_id=donor.lane_id,
                )
            (store.cas_dir / missing_cid).unlink()
            store.put_continuation(key, head_pointer)

            # A syntactically valid successor cannot change static authority.
            drifted = replace(
                head,
                configuration_root=head.configuration_root + "-drift",
                generation=2,
                prior_process_birth_cid=head_cid,
            )
            drifted_cid = store.put_cas(drifted.to_dict())
            store.put_continuation(
                key,
                {
                    **head_pointer,
                    "process_birth_cid": drifted_cid,
                    "generation": 2,
                },
            )
            with pytest.raises(ExecutionPlanError, match="identity.*drift"):
                execution_plan_module._load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                    lane_id=donor.lane_id,
                )
            (store.cas_dir / drifted_cid).unlink()
            store.put_continuation(key, head_pointer)

            # A self-cycle requires changing immutable CAS bytes and is denied
            # by content identity before it can be followed as authority.
            head_path = store.cas_dir / head_cid
            original_bytes = head_path.read_bytes()
            envelope = json.loads(original_bytes)
            envelope["payload"]["prior_process_birth_cid"] = head_cid
            head_path.write_text(
                json.dumps(envelope, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            head_path.chmod(0o600)
            with pytest.raises(ExecutionPlanError):
                execution_plan_module._load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                    lane_id=donor.lane_id,
                )
            head_path.write_bytes(original_bytes)
            head_path.chmod(0o600)
            restored = (
                execution_plan_module._load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                    lane_id=donor.lane_id,
                )
            )
            assert restored is not None and restored[0] == head_cid
            assert restored[2][-1] == (root_cid, root)


def test_runner_reassigns_preclaim_crash_in_freed_slot_while_peer_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery starts immediately without exceeding immutable wave width."""

    repo, _board, receipt, donor, peer, crashed = _fenced_plan_children(
        tmp_path
    )
    control_plane_pin, control_plane_launch = _test_sealed_control_plane(
        tmp_path,
        source_head=receipt.slice_manifest.source_head,
        source_tree=receipt.slice_manifest.repository_tree_id,
    )
    tracks = (donor.track().resolve(repo), peer.track().resolve(repo))
    spawned: list[subprocess.Popen[bytes]] = []
    started_names: list[str] = []
    live_widths: list[int] = []
    peer_waiting_when_recovery_started = False
    peer_waiting_path: Path | None = None
    peer_completed_path: Path | None = None

    def spawn_bound_process(
        track: multi_runner_module.SupervisorTrack,
        *,
        barrier_waiter: bool,
    ) -> subprocess.Popen[bytes]:
        nonlocal peer_completed_path
        nonlocal peer_waiting_path
        nonlocal peer_waiting_when_recovery_started
        ready_path = tmp_path / f"{track.name}-{len(spawned)}.ready"
        waiting_path = tmp_path / f"{track.name}-{len(spawned)}.waiting"
        completed_path = tmp_path / f"{track.name}-{len(spawned)}.completed"
        if barrier_waiter:
            peer_waiting_path = waiting_path
            peer_completed_path = completed_path
            code = (
                "import os,time; from pathlib import Path; "
                "from ipfs_accelerate_py.agent_supervisor.entrypoints."
                "execution_plan import ProductionParallelPlanAdapter; "
                "from ipfs_accelerate_py.agent_supervisor.task_sources."
                "plan_revision_store import PlanRevisionStore; "
                "ready=Path(os.environ['_ASE3_READY']); "
                "waiting=Path(os.environ['_ASE3_WAITING']); "
                "completed=Path(os.environ['_ASE3_COMPLETED'])\n"
                "while not ready.exists():\n"
                "    time.sleep(0.01)\n"
                "adapter=ProductionParallelPlanAdapter(PlanRevisionStore("
                "os.environ['_ASE3_STORE'])); "
                "waiting.write_text('waiting'); "
                "adapter.await_wave_diff_barrier("
                "revision_cid=os.environ['_ASE3_REVISION'], "
                "slice_manifest_cid=os.environ['_ASE3_MANIFEST'], "
                "timeout_ms=60000); "
                "completed.write_text('completed')"
            )
        else:
            code = "import time; time.sleep(60)"
        command = (sys.executable, "-c", code, *track.extra_args)
        state_root = track.supervisor_pid_path.parent.resolve()
        lifecycle_token = _test_lifecycle_token(
            tmp_path,
            f"immediate-recovery-{track.name}-{len(spawned)}",
        )
        profile = multi_runner_module.LifecycleProfile(
            target_id=f"supervisor-track:{track.name}",
            run_id=f"test-immediate-recovery-{track.name}-{lifecycle_token}",
            configuration_root=(
                f"test-immediate-recovery-{track.name}-{lifecycle_token}"
            ),
            repository_root=str(repo.resolve()),
            state_root=str(state_root),
            run_root=str(
                state_root
                / "lifecycle-runs"
                / f"{track.name}-{lifecycle_token}"
            ),
            argv=command,
            cwd=str(repo.resolve()),
        )
        environment = profile.launch_environment(len(spawned))
        environment["PYTHONPATH"] = str(REPO_ROOT)
        environment["_ASE3_STORE"] = str(
            repo / donor.plan_revision_store_path
        )
        environment["_ASE3_REVISION"] = donor.revision_cid
        environment["_ASE3_MANIFEST"] = donor.slice_manifest_cid
        environment["_ASE3_READY"] = str(ready_path)
        environment["_ASE3_WAITING"] = str(waiting_path)
        environment["_ASE3_COMPLETED"] = str(completed_path)
        process = _spawn_test_process(
            command,
            cwd=repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        identity = _capture_test_process_identity(process, profile)
        process._agent_supervisor_lifecycle_profile = profile
        process._agent_supervisor_process_identity = identity
        process._agent_supervisor_process_birth_cid = (
            multi_runner_module._persist_plan_bound_process_birth(
                profile=profile,
                process_identity=identity,
                repo_root=repo,
            )
        )
        spawned.append(process)
        ready_path.write_text("ready", encoding="utf-8")
        if barrier_waiter:
            deadline = time.monotonic() + 5.0
            while not waiting_path.exists() and process.poll() is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            assert waiting_path.exists()
            assert not completed_path.exists()
            assert process.poll() is None
        started_names.append(track.name)
        live_widths.append(
            sum(item.poll() is None for item in (crashed, *spawned))
        )
        if track.name.startswith("recovery-"):
            peer_waiting_when_recovery_started = (
                peer_waiting_path is not None
                and peer_completed_path is not None
                and spawned[0].poll() is None
                and peer_waiting_path.exists()
                and not peer_completed_path.exists()
            )
        return process

    def controlled_start(track, **_kwargs):
        if track.name == donor.name:
            started_names.append(track.name)
            live_widths.append(
                sum(item.poll() is None for item in (crashed, *spawned))
            )
            return crashed
        return spawn_bound_process(
            track,
            barrier_waiter=track.name == peer.name,
        )

    monkeypatch.setattr(
        multi_runner_module,
        "start_track",
        controlled_start,
    )
    result = multi_runner_module.run_supervisor_tracks(
        tracks,
        repo_root=repo,
        common_args=(),
        # Leave enough time after the CAS-bound transfer for the peer's real
        # barrier waiter to persist its reassignment-window extension before
        # the bounded runner fences the process.  Killing it during that
        # atomic store publication would test SIGKILL residue, not recovery
        # width or sibling-wait semantics.
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        plan_bound_children=(donor, peer),
        accepted_control_plane_pin=control_plane_pin,
        accepted_control_plane_descriptor=control_plane_launch.descriptor,
        output=lambda _message: None,
    )
    assert result["reassignment_count"] == 1
    assert result["reassignment_blockers"] == []
    assert result["track_count"] == 3
    assert result["all_trees_fenced"] is True
    assert peer_waiting_when_recovery_started is True
    assert max(live_widths) <= len(receipt.slice_manifest.nonempty) == 2
    recovery_names = [
        name for name in started_names if name.startswith("recovery-")
    ]
    assert len(recovery_names) == 1
    reassignment = ProductionParallelPlanAdapter(
        PlanRevisionStore(repo / donor.plan_revision_store_path)
    ).load_slice_reassignment(
        revision_cid=donor.revision_cid,
        slice_id=donor.slice_id,
    )
    assert reassignment is not None
    assert reassignment[1].generation == 1
    assert reassignment[1].recipient_lane_id.startswith("recovery-1-")
    assert reassignment[1].recipient_lane_id != donor.lane_id


def test_recovery_artifact_binding_uses_exact_reassigned_lane_name(
    tmp_path: Path,
) -> None:
    """An ancestor lane token cannot select another lane's log authority."""

    root = tmp_path / "ancestor-lane-0-token" / "repo"
    state_root = root / "state"
    state_dir = state_root / "recovery-1-token"
    runtime_bindings = (
        {
            "lane_index": 0,
            "lane_id": "lane-0",
            "active_task_id": "WRONG-TASK",
            "task_ids": ("WRONG-TASK",),
            "attempt": 1,
        },
        {
            "lane_index": 1,
            "lane_id": state_dir.name,
            "active_task_id": "RIGHT-TASK",
            "task_ids": ("RIGHT-TASK",),
            "attempt": 1,
        },
    )
    common = {
        "directory_projection": False,
        "runtime_roots": (
            state_root,
            root / "worktrees",
            root / "merge-queue",
        ),
        "owner_bound_artifacts": (),
        "runtime_bindings": runtime_bindings,
        "state_dir": state_dir,
        "state_prefix": "recovery_1_token",
    }
    assert not multi_runner_module._plan_bound_recovery_runtime_kind(
        state_dir / "implementation_logs/wrong-task-attempt-1.log",
        **common,
    )
    assert multi_runner_module._plan_bound_recovery_runtime_kind(
        state_dir / "implementation_logs/right-task-attempt-1.log",
        **common,
    ) == "file"


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
