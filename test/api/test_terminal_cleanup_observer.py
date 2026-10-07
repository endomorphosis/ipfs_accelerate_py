"""Terminal cleanup observation never creates, locks, or repairs CAS state.

These cases use owned on-disk CAS directories, actual signed current routes,
and the current command builder. Docker absence and dispatch observations are
explicit doubles; no Docker or provider command is launched.
"""

from pathlib import Path

import pytest

from ipfs_accelerate_py import agent_implementation_route as authority
from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
    DurableProviderAttemptCAS,
)


def _snapshot(root: Path) -> dict:
    return {
        str(path.relative_to(root)): (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_mode,
            path.read_bytes() if path.is_file() and not path.is_symlink() else None,
        )
        for path in root.rglob("*")
    }


def _observe(path: Path, identity: str, logical: str, repo: Path):
    return authority.observe_agent_implementation_terminal_cleanup(
        store_path=path,
        expected_store_identity=identity,
        logical_attempt_id=logical,
        repo_root=repo,
        max_age_ms=60_000,
    )


def test_missing_terminal_store_does_not_create_any_paths(tmp_path):
    path = tmp_path / "missing" / "attempts"
    assert _observe(path, "sha256:" + "a" * 64, "attempt:one", tmp_path) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "state", ["missing", "reserved", "corrupt", "foreign", "symlink"]
)
def test_nonterminal_or_untrusted_native_cas_stays_unchanged(tmp_path, state):
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    logical = "attempt:observer-test"
    if state in {"reserved", "corrupt"}:
        store.reserve_or_adopt(
            logical_attempt_id=logical,
            route_id="route:observer-test",
            decision_id="decision:observer-test",
            task_id="task:observer-test",
            worktree_id="worktree:observer-test",
            authorized=True,
            now_ms=1_000,
        )
    if state == "corrupt":
        record = next(store.directory.glob("*.json"))
        record.write_bytes(b'{"schema":"foreign-record"}\n')
    path, identity = store.directory, store.directory_identity
    if state == "foreign":
        identity = "sha256:" + "f" * 64
    if state == "symlink":
        path = tmp_path / "aliased-attempts"
        path.symlink_to(store.directory, target_is_directory=True)
    before = _snapshot(tmp_path)
    assert _observe(path, identity, logical, tmp_path) is None
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "identity,logical",
    [
        ("", "attempt:one"),
        (False, "attempt:one"),
        ("sha256:" + "a" * 64, ""),
        ("sha256:" + "a" * 64, False),
    ],
)
def test_invalid_observation_identifiers_cannot_initialize_store(
    tmp_path, identity, logical
):
    assert _observe(tmp_path / "attempts", identity, logical, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_capacity_companion_parser_keeps_only_last_four_records():
    prefix = authority.AGENT_IMPLEMENTATION_CODEX_CAPACITY_RECEIPT_PREFIX
    text = "unrelated log\n" + "\n".join(
        prefix + '{"sequence":' + str(index) + '}' for index in range(7)
    )
    assert authority.extract_agent_implementation_codex_capacity_receipts(text) == tuple(
        {"sequence": index} for index in range(3, 7)
    )


@pytest.mark.parametrize(
    "raw",
    [
        "not-json", "null", "[]", '{"key":1,"key":2}',
        '{"nested":{"key":1,"key":2}}', '{}\r',
        '{"text":"' + "x" * (16 * 1024) + '"}',
    ],
    ids=["invalid-json", "null", "array", "duplicate", "nested-duplicate", "cr", "oversize"],
)
def test_capacity_companion_parser_rejects_noncanonical_or_oversize_records(raw):
    prefix = authority.AGENT_IMPLEMENTATION_CODEX_CAPACITY_RECEIPT_PREFIX
    assert authority.extract_agent_implementation_codex_capacity_receipts(prefix + raw) == ()

# Signed fixtures adapted from the existing route/closure tests. Current command
# builders and signature validators run unchanged; no Docker or provider starts.
import base64
import hashlib
import json
import os
import tempfile
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.control import provider_attempt_store as cas
from ipfs_accelerate_py.agent_supervisor.entrypoints import local_profile as local_profile_module
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key, export_local_profile_lifecycle_witness,
    initialize_local_profile, lifecycle_root_identity_did,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import candidate_rejection_closure as closure
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl
from test.api import test_llm_router_agent_supervisor_fallback_route as routes
from test.api.test_protected_watchdog_cleanup_handoff import _publish_prepared


@pytest.fixture(autouse=True)
def isolated_lifecycle_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(local_profile_module, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "root-registry")


_git = routes._git
_sign = routes._sign
_canonical = routes._canonical
_test_control_plane_capsule = routes._test_control_plane_capsule
_BOARD = routes._BOARD
_ARTIFACT = routes._ARTIFACT
_ROOT_PIN = routes._ROOT_PIN
_WITNESS = routes._WITNESS
_ROUTE_ID = authority._V3_AGENT_IMPLEMENTATION_ROUTE_ID


def _current_launch_context(tmp_path, workspace):
    """Build the current protected command using inert owned credential files."""
    lease_root = Path(tempfile.mkdtemp(prefix="asref-codex-container-"))
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(mode=0o700)
    cidfile = lease_root / "container.cid"
    cidfile.write_text("6" * 64, encoding="ascii")
    cidfile.chmod(0o600)
    provider_home = Path(tempfile.mkdtemp(prefix="asref-codex-home-"))
    descriptor, prompt_name = tempfile.mkstemp(prefix="asref-grok-prompt-", suffix=".txt")
    os.close(descriptor)
    prompt_path = Path(prompt_name)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="ascii")
    source_auth.chmod(0o600)
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 126\n", encoding="ascii")
    codex.chmod(0o500)
    provider_argv = [
        str(codex), "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
        "-s", "workspace-write", "-C", str(workspace), "-m",
        authority.AGENT_IMPLEMENTATION_CANONICAL_FALLBACK_MODEL_ID,
        "-c", 'model_reasoning_effort="high"', "-",
    ]
    environment = runner._codex_task_container_environment()
    container_name = "ipfs-accelerate-codex-123-" + "a" * 32
    create = runner._docker_codex_fallback_command(
        codex_command=provider_argv, workspace=workspace, source_auth=source_auth,
        child_env=environment, docker_config=docker_config,
        container_name=container_name, cidfile=cidfile, docker_bin="/usr/bin/docker",
        isolation_image=authority.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID, base_env={},
    )
    start = [
        "/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config",
        str(docker_config), "start", "--attach", "--interactive", "6" * 64,
    ]
    cleanup = {
        "schema": "ipfs_accelerate_py.agent_supervisor.provider-effect-cleanup@1",
        "lease_root": str(lease_root), "docker_config": str(docker_config),
        "cidfile": str(cidfile), "provider_home": str(provider_home),
        "prompt_path": str(prompt_path), "watchdog_pid": os.getpid(),
        "watchdog_start_ticks": runner._runner_process_start_ticks(os.getpid()),
    }
    cleanup["receipt_id"] = runner._effect_receipt_identity(cleanup)
    context = {
        "provider_id": "codex", "container_name": container_name,
        "container_id": "sha256:" + "6" * 64,
        "image_id": authority.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID,
        "image_receipt": {
            "image_id": authority.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID,
            "image_label": authority.AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL,
        },
        "runtime_receipt": runner._docker_runtime_receipt("/usr/bin/docker"),
        "command_receipt": {"create_argv": create, "start_argv": start, "provider_argv": provider_argv},
        "mount_receipt": [create[index + 1] for index, arg in enumerate(create[:-1]) if arg == "--mount"],
        "environment_receipt": {"docker_cli": runner._docker_control_env(environment), "container": environment},
        "cleanup_receipt": cleanup, "cleanup_id": cleanup["receipt_id"],
    }
    routes._refresh_effect_detail_identities(context)
    assert authority._agent_effect_launch_details_valid(context, workspace_path=str(workspace))
    return context, {
        "lease_root": lease_root, "docker_config": docker_config, "cidfile": cidfile,
        "provider_home": provider_home, "prompt_path": prompt_path,
    }


def _reviewed_route(
    tmp_path: Path,
    *,
    reviewer_provider: str = "local_operator",
    corrupt_static_signature: bool = False,
) -> tuple[
    Path,
    Ed25519PrivateKey,
    llm_router.AgentImplementationRoutePlan,
    llm_router.AgentImplementationInvocationBinding,
]:
    repository = tmp_path / "candidate"
    repository.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Route Test")
    (repository / "README").write_text("accepted source\n", encoding="utf-8")
    _git(repository, "add", "README")
    _git(repository, "-c", "commit.gpgsign=false", "commit", "-qm", "source")
    source_head = _git(repository, "rev-parse", "HEAD^{commit}")
    source_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_identity = ed25519_did_key(reviewer_key.public_key())
    route_fields: dict[str, Any] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.7",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-6.1-sol",
        "fallback_reasoning_effort": "high",
        "route_id": _ROUTE_ID,
        "allowed_trigger_classes": [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ],
    }
    profile_dir = tmp_path / "reviewer-profile"
    lifecycle_dir = tmp_path / "reviewer-lifecycle"
    profile = initialize_local_profile(
        repository_cid="repository:one",
        baseline_commit=source_head,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        signing_key=reviewer_key.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        ),
        effect_bounds=("edit", "isolated_worktree", "test"),
        budget_cid="budget:one",
        resource_cid="resource:one",
        route_id=_ROUTE_ID,
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        fallback_provider_id="codex",
        fallback_model_id="gpt-6.1-sol",
        fallback_reasoning_effort="high",
    )
    root_identity_did = lifecycle_root_identity_did()
    pinned_at_ms = int(time.time()) * 1000
    root_pin: dict[str, Any] = {
        "schema": llm_router._AGENT_LIFECYCLE_ROOT_PIN_SCHEMA,
        "board_namespace": _BOARD,
        "base_head": source_head,
        "base_tree": source_tree,
        "root_identity_did": root_identity_did,
        "pinned_at_ms": pinned_at_ms,
    }
    root_pin["pin_id"] = llm_router._content_addressed_mapping(
        root_pin,
        identity_field="pin_id",
    )
    root_pin_path = repository / _ROOT_PIN
    root_pin_path.parent.mkdir(parents=True, exist_ok=True)
    root_pin_path.write_bytes(_canonical(root_pin))
    root_pin_path.chmod(0o644)
    _git(repository, "add", _ROOT_PIN)
    _git(
        repository,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "pin lifecycle root",
    )
    root_pin_sha256 = "sha256:" + hashlib.sha256(
        root_pin_path.read_bytes()
    ).hexdigest()
    witness_nonce = "witness:" + hashlib.sha256(
        str(repository).encode("utf-8")
    ).hexdigest()
    authorized_at_ms = int(time.time()) * 1000
    witness = export_local_profile_lifecycle_witness(
        repository_cid="repository:one",
        board_namespace=_BOARD,
        base_head=source_head,
        base_tree=source_tree,
        nonce=witness_nonce,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        observed_at_ms=authorized_at_ms,
        expires_at_ms=authorized_at_ms + 10 * 60 * 1000,
    )
    witness_path = repository / _WITNESS
    witness_path.write_bytes(_canonical(witness))
    witness_path.chmod(0o644)
    witness_sha256 = "sha256:" + hashlib.sha256(
        witness_path.read_bytes()
    ).hexdigest()
    authority_bounds: dict[str, Any] = {
        "repository_cid": "repository:one",
        "baseline_commit": source_head,
        "effects": ["edit", "isolated_worktree", "test"],
        "budget_cid": "budget:one",
        "resource_cid": "resource:one",
        "authority_cid": profile.content_id,
    }
    review_payload = llm_router.agent_implementation_route_review_payload(
        board_namespace=_BOARD,
        authorization_kind="explicit_operator_override",
        source_head=source_head,
        source_tree=source_tree,
        route=route_fields,
        authority_bounds=authority_bounds,
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        reviewer_profile_id=profile.profile_id,
        reviewer_profile_content_id=profile.content_id,
        reviewer_lifecycle_anchor_id=profile.lifecycle_anchor_id,
        reviewer_lifecycle_generation=profile.lifecycle_generation,
        reviewer_witness_path=_WITNESS,
        reviewer_witness_sha256=witness_sha256,
        lifecycle_root_identity_did=root_identity_did,
        lifecycle_witness_nonce=witness_nonce,
        lifecycle_root_pin_path=_ROOT_PIN,
        lifecycle_root_pin_sha256=root_pin_sha256,
        authorized_at_ms=authorized_at_ms,
        fallback_implementer_identity="codex",
    )
    signature = _sign(reviewer_key, review_payload)
    if corrupt_static_signature:
        signature = signature[:-2] + "AA"
    artifact = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-policy-authorization@2"
        ),
        "board_namespace": _BOARD,
        "authorization_source": {
            "kind": "explicit_operator_override",
            "source_head": source_head,
            "source_tree": source_tree,
            "prospective_only": True,
            "requires_descendant_tree": True,
        },
        "route": route_fields,
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
            "provider": reviewer_provider,
            "profile_id": profile.profile_id,
            "profile_content_id": profile.content_id,
            "lifecycle_anchor_id": profile.lifecycle_anchor_id,
            "generation": profile.lifecycle_generation,
            "witness_path": _WITNESS,
            "witness_sha256": witness_sha256,
            "signature": signature,
        },
        "authority_bounds": authority_bounds,
        "fallback_implementer_identity": "codex",
        "lifecycle_root_identity_did": root_identity_did,
        "lifecycle_witness_nonce": witness_nonce,
        "lifecycle_root_pin_path": _ROOT_PIN,
        "lifecycle_root_pin_sha256": root_pin_sha256,
        "authorized_at_ms": authorized_at_ms,
    }
    artifact_path = repository / _ARTIFACT
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(_canonical(artifact))
    artifact_path.chmod(0o644)
    _git(repository, "add", _ARTIFACT, _WITNESS)
    _git(
        repository,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "reviewed route",
    )
    digest = "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    authorization = llm_router.load_agent_implementation_route_authorization(
        repo_root=repository,
        artifact_path=_ARTIFACT,
        board_namespace=_BOARD,
        expected_sha256=digest,
    )
    route = llm_router.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.7",
        fallback_provider_id="codex",
        fallback_model_id="gpt-6.1-sol",
        fallback_trigger="primary_quota_or_auth_unavailable",
        fallback_reasoning_effort="high",
        authorization=authorization,
    )

    control_plane = _test_control_plane_capsule(
        tmp_path,
        source_head=source_head,
        source_tree=source_tree,
    )
    baseline = _git(repository, "rev-parse", "HEAD^{commit}")
    attempt = 1
    task_id = "task:one"
    task_revision_cid = "task-revision:one"
    prompt_cid = "prompt:one"
    worktree_id = content_identity(
        {
            "workspace_path": str(repository.resolve()),
            "baseline_commit": baseline,
        }
    )
    logical_body = {
        "task_id": task_id,
        "task_revision_cid": task_revision_cid,
        "attempt": attempt,
        "prompt_cid": prompt_cid,
        "worktree_id": worktree_id,
        "route_id": route.route_id,
    }
    logical_attempt_id = content_identity(logical_body)
    invocation_id = content_identity(
        {**logical_body, "logical_attempt_id": logical_attempt_id}
    )
    issued_at_ms = int(time.time() * 1000)
    attempt_store, attempt_store_identity = (
        llm_router.bind_agent_implementation_attempt_store(
            tmp_path / "attempt-state",
            create=True,
        )
    )
    unsigned = llm_router.AgentImplementationInvocationBinding(
        schema=(
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-invocation@2"
        ),
        invocation_id=invocation_id,
        logical_attempt_id=logical_attempt_id,
        task_id=task_id,
        attempt=attempt,
        task_revision_cid=task_revision_cid,
        prompt_cid=prompt_cid,
        worktree_id=worktree_id,
        workspace_path=str(repository.resolve()),
        repository_cid="repository:one",
        baseline_commit=baseline,
        effects=("edit", "isolated_worktree", "test"),
        scope_cid="scope:one",
        budget_cid="budget:one",
        resource_cid="resource:one",
        authority_cid=profile.content_id,
        route_id=route.route_id,
        primary_provider_id=route.primary_provider_id,
        primary_model_id=route.primary_model_id,
        fallback_provider_id=route.fallback_provider_id,
        fallback_model_id=route.fallback_model_id,
        fallback_reasoning_effort=route.fallback_reasoning_effort,
        fallback_implementer_identity=route.fallback_implementer_identity,
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        profile_id=profile.profile_id,
        profile_identity_did=profile.identity_did,
        profile_lifecycle_anchor_id=profile.lifecycle_anchor_id,
        profile_lifecycle_generation=profile.lifecycle_generation,
        profile_dir=str(profile_dir.resolve()),
        profile_lifecycle_dir=str(lifecycle_dir.resolve()),
        issued_at_ms=issued_at_ms,
        expires_at_ms=issued_at_ms + 60_000,
        provider_attempt_store=str(attempt_store),
        provider_attempt_store_identity=attempt_store_identity,
        control_plane=control_plane,
        reviewer_signature="pending",
    )
    invocation = replace(
        unsigned,
        reviewer_signature=_sign(reviewer_key, unsigned.signed_payload()),
    )
    bound = llm_router.bind_agent_implementation_route_invocation(
        route,
        invocation,
        repo_root=repository,
        workspace=repository,
        expected_binding=invocation.signed_payload(),
        now_ms=issued_at_ms,
        max_age_ms=60_000,
    )
    return repository, reviewer_key, bound, invocation


def _native_cleanup(
    tmp_path, monkeypatch, *, repository, route, invocation, complete=True
):
    now = int(time.time() * 1000)
    nonce = "c" * 64
    primary = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="not signed in",
        nonce=nonce,
        model="grok-4.7",
        probe_returncode=1,
        observed_at_ms=now,
    )
    decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=primary,
        expected_nonce=nonce,
        expected_model="grok-4.7",
        expected_probe_returncode=1,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=now,
        max_age_ms=60000,
    )
    assert decision.authorized
    authorization = llm_router.build_agent_implementation_effect_authorization_context(
        route=route,
        repo_root=repository,
        failure_receipt=primary,
        decision=decision,
        expected_nonce=nonce,
        expected_model="grok-4.7",
        expected_probe_returncode=1,
    )
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    state = tmp_path / "runner-state"
    (state / "run").mkdir(parents=True)
    for name in runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        monkeypatch.setenv(name, "fixture")
    for name, value in {
        runner.STATE_ROOT_ENV: str(state),
        runner.RUN_ROOT_ENV: str(state / "run"),
        runner.REPOSITORY_ROOT_ENV: str(repository),
        runner.FENCING_EPOCH_ENV: "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    context, paths = _current_launch_context(tmp_path, Path(invocation.workspace_path))
    cleanup = context["cleanup_receipt"]
    cleanup["watchdog_pid"] = os.getpid()
    cleanup["watchdog_start_ticks"] = runner._runner_process_start_ticks(os.getpid())
    cleanup.pop("receipt_id")
    cleanup["receipt_id"] = routes._effect_detail_identity(cleanup)
    context["cleanup_id"] = cleanup["receipt_id"]
    routes._refresh_effect_detail_identities(context)
    store = cas.DurableProviderAttemptCAS(
        invocation.provider_attempt_store,
        expected_directory_identity=invocation.provider_attempt_store_identity,
    )
    started = store.reserve_or_adopt(
        logical_attempt_id=invocation.logical_attempt_id,
        route_id=route.route_id,
        decision_id=decision.content_id,
        task_id=invocation.task_id,
        worktree_id=invocation.worktree_id,
        authorized=True,
        authorization_context=authorization,
        launch_context=context,
    )
    case = SimpleNamespace(
        store=store,
        started=started,
        context=context,
        paths=paths,
        binding_path=runner._docker_cleanup_binding_path(context["container_name"]),
        observation={
            "logical_attempt_id": invocation.logical_attempt_id,
            "provider_attempt_store": str(store.directory),
            "provider_attempt_store_identity": store.directory_identity,
        },
    )
    binding = _publish_prepared(case)
    fence = _created_docker_termination_fence(
        container_name=context["container_name"],
        container_id=str(context["container_id"]).removeprefix("sha256:"),
        image_id=context["image_id"],
    )
    binding.update(
        binding_state="command_bound",
        create_command_id="sha256:" + "a" * 64,
        create_cwd=str(root),
        create_environment_id="sha256:" + "b" * 64,
        termination_fence=fence,
    )
    binding.pop("record_id")
    binding["record_id"] = runner._effect_receipt_identity(binding)
    runner._write_private_control_record(
        case.binding_path.parent, case.binding_path.name, binding, replace_existing=True
    )
    identity = runner._cleanup_path_identity(case.binding_path, directory=False)
    outcome = llm_router.build_agent_implementation_route_outcome(
        receipt=primary,
        route=route,
        decision="fallback_succeeded",
        verifier_status=decision.verifier_status,
        fallback_dispatched=True,
        fallback_returncode=0,
        decision_id=decision.content_id,
        reservation_id=started.reservation.reservation_id,
        effect_launch_receipt=started.reservation.effect_launch_receipt,
    )
    terminal = store.complete(
        started.reservation,
        returncode=0,
        outcome=outcome,
        completion_capability=started.completion_capability,
        terminal_cleanup_evidence={
            "binding_path": str(case.binding_path),
            "binding_record_id": binding["record_id"],
            "termination_fence_id": fence["fence_id"],
        },
    )
    if complete:
        birth = runner.read_process_birth(os.getpid())
        dispatch = runner._docker_removal_dispatch_value(
            binding_path=case.binding_path,
            binding_record=binding,
            termination_fence=fence,
            issuer_process_birth={
                "pid": os.getpid(),
                "start_time_ticks": birth.start_time_ticks,
                "boot_id": birth.boot_id,
                "parent_pid": os.getppid(),
            },
            state="request_completed",
            generation=1,
            previous_dispatch_id="sha256:" + "c" * 64,
            docker_returncode=0,
            failure_kind="",
        )
        dispatch_path = runner._docker_removal_dispatch_path(case.binding_path)
        runner._write_private_control_record(
            dispatch_path.parent, dispatch_path.name, dispatch, replace_existing=False
        )
        assert runner._finalize_verified_cleanup_completion(
            binding_path=case.binding_path,
            binding_identity=identity,
            binding_record=binding,
            terminal_cleanup_store=store,
            terminal_cleanup_reservation=terminal,
        )
    command = [
        "python",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
        "--agent-implementation-route-json",
        json.dumps(route.as_binding_dict()),
    ]
    text = (
        routes.render_grok_failure_receipt(primary)
        + "\n"
        + llm_router.render_agent_implementation_route_outcome(outcome)
        + "\n"
    )
    case.command, case.receipt_text = command, text
    return case


@pytest.mark.parametrize("complete", [False, True])
def test_signed_native_terminal_requires_actual_cleanup_completion(
    tmp_path, monkeypatch, complete
):
    repository, key, route, original = _reviewed_route(tmp_path)
    workspace = tmp_path / "worktree"
    _git(repository, "worktree", "add", "--detach", str(workspace), "HEAD")
    route, invocation = _rebind_route(
        repository, route, original, key,
        task_id=original.task_id, task_cid=original.task_revision_cid,
        workspace=workspace,
    )
    case = _native_cleanup(
        tmp_path,
        monkeypatch,
        repository=repository,
        route=route,
        invocation=invocation,
        complete=complete,
    )
    try:
        observed = impl.PortalImplementationDaemon._protected_provider_effect_audit(
            repo_root=repository,
            command_items=case.command,
            receipt_text=case.receipt_text,
            returncode=0,
        )
        assert bool(observed.get("candidate_provider_cleanup")) is complete
        if complete:
            proof = observed["candidate_provider_cleanup"]
            assert closure.exact_seal(proof, "proof_id")
            assert (
                proof["cleanup_progress_id"]
                == case.store.observe(
                    invocation.logical_attempt_id
                ).terminal_cleanup_progress["progress_id"]
            )
            wrong = impl.PortalImplementationDaemon._protected_provider_effect_audit(
                repo_root=repository,
                command_items=case.command,
                receipt_text=case.receipt_text,
                returncode=78,
            )
            assert not wrong.get("candidate_provider_cleanup")
    finally:
        routes._discard_live_cleanup_inputs(case.paths)


@pytest.mark.parametrize(
    "mode",
    ["removed", "incomplete", "missing", "foreign_store", "symlink", "changed_cas"],
)
def test_terminal_observer_is_not_live_effect_authority(tmp_path, monkeypatch, mode):
    from ipfs_accelerate_py import agent_implementation_route as authority

    repository, key, route, original = _reviewed_route(tmp_path)
    parent = tmp_path / "candidate-parent"
    parent.mkdir()
    workspace = parent / "candidat-é"
    routes._git(repository, "worktree", "add", "--detach", str(workspace), "HEAD")
    route, invocation = _rebind_route(
        repository,
        route,
        original,
        key,
        task_id=original.task_id,
        task_cid=original.task_revision_cid,
        workspace=workspace,
    )
    case = _native_cleanup(
        tmp_path,
        monkeypatch,
        repository=repository,
        route=route,
        invocation=invocation,
        complete=mode != "incomplete",
    )
    try:
        produced = impl.PortalImplementationDaemon._protected_provider_effect_audit(
            repo_root=repository,
            command_items=case.command,
            receipt_text=case.receipt_text,
            returncode=0,
        ).get("candidate_provider_cleanup")
        routes._git(repository, "worktree", "remove", str(workspace))
        assert not workspace.exists()
        terminal = case.store.observe(invocation.logical_attempt_id)
        assert (
            llm_router.parse_agent_implementation_effect_authorization_context(
                terminal.authorization_context,
                repo_root=repository,
                effect_started_at_ms=terminal.effect_started_at_ms,
                expected_signer_parent_pid=terminal.effect_launch_receipt[
                    "effect_owner_pid"
                ],
                max_age_ms=60000,
            )
            is None
        )
        with pytest.raises(ValueError):
            llm_router.bind_agent_implementation_route_invocation(
                route,
                invocation,
                repo_root=repository,
                workspace=workspace,
                now_ms=invocation.issued_at_ms,
                max_age_ms=60000,
            )
        store_identity = invocation.provider_attempt_store_identity
        logical = invocation.logical_attempt_id
        if mode == "foreign_store":
            store_identity = "sha256:" + "f" * 64
        elif mode == "missing":
            logical = "sha256:" + "e" * 64
        elif mode == "symlink":
            moved = tmp_path / "candidate-parent-moved"
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        elif mode == "changed_cas":
            observe = cas.DurableProviderAttemptCAS.observe
            calls = []

            def changed(self, logical):
                calls.append(logical)
                return observe(self, logical) if len(calls) == 1 else None

            monkeypatch.setattr(cas.DurableProviderAttemptCAS, "observe", changed)
        evidence = authority.observe_agent_implementation_terminal_cleanup(
            store_path=invocation.provider_attempt_store,
            expected_store_identity=store_identity,
            logical_attempt_id=logical,
            repo_root=repository,
            max_age_ms=60000,
        )
        if mode == "removed":
            assert (
                type(evidence) is authority.AgentImplementationTerminalCleanupEvidence
            )
            value = json.loads(evidence.evidence_json)
            assert value["provider_cleanup"] == produced
            assert (
                value["provider_cleanup"]["cleanup_progress_id"]
                == terminal.terminal_cleanup_progress["progress_id"]
            )
            assert value["new_effect_authority"] is False
            assert not hasattr(evidence, "route") and not hasattr(
                evidence, "authorized"
            )
            # Historical timestamps remain strict on every public effect API.
            common = {
                "repo_root": repository,
                "now_ms": terminal.effect_started_at_ms,
                "max_age_ms": 60000,
                "historical_effect_started_at_ms": terminal.effect_started_at_ms,
            }
            for fn, args, kwargs in (
                (
                    authority.verify_agent_implementation_invocation_binding,
                    (invocation,),
                    {**common, "route": route, "workspace": workspace},
                ),
                (
                    authority.bind_agent_implementation_route_invocation,
                    (route, invocation),
                    {**common, "workspace": workspace},
                ),
                (
                    authority.decide_agent_implementation_fallback,
                    (route,),
                    {
                        **common,
                        "failure_receipt": terminal.authorization_context[
                            "failure_receipt"
                        ],
                        "expected_nonce": terminal.authorization_context[
                            "expected_nonce"
                        ],
                        "expected_model": terminal.authorization_context[
                            "expected_model"
                        ],
                        "expected_probe_returncode": terminal.authorization_context[
                            "expected_probe_returncode"
                        ],
                        "expected_invocation_binding": invocation.signed_payload(),
                    },
                ),
            ):
                with pytest.raises(ValueError):
                    fn(*args, **kwargs)
                with pytest.raises(TypeError, match="_terminal_workspace"):
                    fn(*args, **kwargs, _terminal_workspace=str(workspace))
            with pytest.raises(TypeError, match="_terminal_workspace"):
                authority.parse_agent_implementation_effect_authorization_context(
                    terminal.authorization_context,
                    repo_root=repository,
                    effect_started_at_ms=terminal.effect_started_at_ms,
                    expected_signer_parent_pid=terminal.effect_launch_receipt[
                        "effect_owner_pid"
                    ],
                    max_age_ms=60000,
                    _terminal_workspace=str(workspace),
                )
            assert (
                closure.observe_provider_cleanup(
                    repo_root=repository,
                    command_items=case.command,
                    receipt_text=case.receipt_text,
                )
                == value["provider_cleanup"]
            )
            for companion in ({}, {"schema": "foreign-capacity-record"}):
                assert closure.observe_provider_cleanup(
                    repo_root=repository,
                    command_items=case.command,
                    receipt_text=case.receipt_text + "\n"
                    + authority.AGENT_IMPLEMENTATION_CODEX_CAPACITY_RECEIPT_PREFIX
                    + json.dumps(companion),
                ) is None
        else:
            assert evidence is None
    finally:
        routes._discard_live_cleanup_inputs(case.paths)


def _created_docker_termination_fence(
    *,
    container_name: str,
    container_id: str,
    image_id: str,
) -> dict[str, object]:
    """Build one validated no-init fence for an exact created container."""

    body: dict[str, object] = {
        "schema": runner._DOCKER_TERMINATION_FENCE_SCHEMA,
        "provider": "codex",
        "container_id": container_id,
        "container_name": container_name,
        "image_id": image_id,
        "isolation_label": "ipfs_accelerate.codex_fallback_isolation",
        "docker_state": "created",
        "init_pid": 0,
        "kernel_scope": {},
    }
    body["fence_id"] = runner._effect_receipt_identity(body)
    return runner._validated_docker_termination_fence(
        body,
        provider="codex",
        container_name=container_name,
        expected_container_id=container_id,
        expected_image_id=image_id,
    )


def _rebind_route(repository, route, invocation, key, *, task_id, task_cid, workspace):
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    baseline = routes._git(repository, "rev-parse", "HEAD")
    worktree_id = content_identity(
        {"workspace_path": str(workspace.resolve()), "baseline_commit": baseline}
    )
    logical = {
        "task_id": task_id,
        "task_revision_cid": task_cid,
        "attempt": 1,
        "prompt_cid": invocation.prompt_cid,
        "worktree_id": worktree_id,
        "route_id": route.route_id,
    }
    logical_id = content_identity(logical)
    unsigned = replace(
        invocation,
        task_id=task_id,
        task_revision_cid=task_cid,
        worktree_id=worktree_id,
        workspace_path=str(workspace.resolve()),
        baseline_commit=baseline,
        logical_attempt_id=logical_id,
        invocation_id=content_identity({**logical, "logical_attempt_id": logical_id}),
        reviewer_signature="pending",
    )
    signed = replace(
        unsigned, reviewer_signature=routes._sign(key, unsigned.signed_payload())
    )
    bound = llm_router.bind_agent_implementation_route_invocation(
        replace(route, invocation_binding=None),
        signed,
        repo_root=repository,
        workspace=workspace,
        expected_binding=signed.signed_payload(),
        now_ms=signed.issued_at_ms,
        max_age_ms=60000,
    )
    return bound, signed
