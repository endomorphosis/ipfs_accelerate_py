from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, replace
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
    local_profile as local_profile_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
    export_local_profile_lifecycle_witness,
    initialize_local_profile,
    lifecycle_root_identity_did,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner import (
    _validate_quota_evidence_in_accepted_child,
)

AUTHORIZATION_PATH = Path(
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "provider_fallback_policy_authorization_20260808.json"
)
BOARD_NAMESPACE = "agent-supervisor-prompt-only-self-improvement-v3"
ROUTE_ID = (
    "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
)
LIFECYCLE_ROOT_PIN_PATH = Path(llm_router._V3_AGENT_LIFECYCLE_ROOT_PIN_PATH)
LIFECYCLE_WITNESS_PATH = Path(
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "local_profile_lifecycle_witness_20260808.json"
)
SPENDING_LIMIT_MESSAGE = (
    "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
    "You have run out of credits or need a Grok subscription. Add credits at "
    "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
)


@dataclass(frozen=True)
class _AuthorizedRepository:
    repo: Path
    artifact: Path
    reviewer_key: Ed25519PrivateKey
    profile: Any
    profile_dir: Path
    lifecycle_dir: Path
    source_head: str
    source_tree: str


@pytest.fixture(autouse=True)
def _isolated_lifecycle_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_profile_module,
        "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE",
        tmp_path / "root-registry",
    )


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sign(key: Ed25519PrivateKey, payload: dict[str, Any]) -> str:
    return base64.b64encode(key.sign(_canonical(payload))).decode("ascii")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _authorized_repo(tmp_path: Path) -> _AuthorizedRepository:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Route Test")
    _git(repo, "config", "user.email", "route@example.invalid")
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")
    source_head = _git(repo, "rev-parse", "HEAD")
    source_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_identity = ed25519_did_key(reviewer_key.public_key())
    route: dict[str, Any] = {
        "route_id": ROUTE_ID,
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
        route_id=ROUTE_ID,
        reviewer_identity=reviewer_identity,
        reviewer_provider="local_operator",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_reasoning_effort="high",
    )
    root_identity_did = lifecycle_root_identity_did()
    pinned_at_ms = int(time.time()) * 1000
    root_pin: dict[str, Any] = {
        "schema": llm_router._AGENT_LIFECYCLE_ROOT_PIN_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "base_head": source_head,
        "base_tree": source_tree,
        "root_identity_did": root_identity_did,
        "pinned_at_ms": pinned_at_ms,
    }
    root_pin["pin_id"] = llm_router._content_addressed_mapping(
        root_pin,
        identity_field="pin_id",
    )
    root_pin_path = repo / LIFECYCLE_ROOT_PIN_PATH
    root_pin_path.parent.mkdir(parents=True)
    root_pin_path.write_bytes(_canonical(root_pin))
    _git(repo, "add", LIFECYCLE_ROOT_PIN_PATH.as_posix())
    _git(repo, "commit", "-m", "pin lifecycle root")
    root_pin_path.chmod(0o400)
    root_pin_sha256 = "sha256:" + hashlib.sha256(
        root_pin_path.read_bytes()
    ).hexdigest()

    witness_nonce = "witness:" + hashlib.sha256(
        str(repo).encode("utf-8")
    ).hexdigest()
    authorized_at_ms = int(time.time()) * 1000
    witness = export_local_profile_lifecycle_witness(
        repository_cid="repository:one",
        board_namespace=BOARD_NAMESPACE,
        base_head=source_head,
        base_tree=source_tree,
        nonce=witness_nonce,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        observed_at_ms=authorized_at_ms,
        expires_at_ms=authorized_at_ms + 10 * 60 * 1000,
    )
    witness_path = repo / LIFECYCLE_WITNESS_PATH
    witness_path.write_bytes(_canonical(witness))
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
        board_namespace=BOARD_NAMESPACE,
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
        reviewer_witness_path=LIFECYCLE_WITNESS_PATH.as_posix(),
        reviewer_witness_sha256=witness_sha256,
        lifecycle_root_identity_did=root_identity_did,
        lifecycle_witness_nonce=witness_nonce,
        lifecycle_root_pin_path=LIFECYCLE_ROOT_PIN_PATH.as_posix(),
        lifecycle_root_pin_sha256=root_pin_sha256,
        authorized_at_ms=authorized_at_ms,
        fallback_implementer_identity="codex",
    )
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-policy-authorization@2"
        ),
        "board_namespace": BOARD_NAMESPACE,
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
            "witness_path": LIFECYCLE_WITNESS_PATH.as_posix(),
            "witness_sha256": witness_sha256,
            "signature": _sign(reviewer_key, review_payload),
        },
        "authority_bounds": authority_bounds,
        "fallback_implementer_identity": "codex",
        "lifecycle_root_identity_did": root_identity_did,
        "lifecycle_witness_nonce": witness_nonce,
        "lifecycle_root_pin_path": LIFECYCLE_ROOT_PIN_PATH.as_posix(),
        "lifecycle_root_pin_sha256": root_pin_sha256,
        "authorized_at_ms": authorized_at_ms,
    }
    artifact = repo / AUTHORIZATION_PATH
    artifact.write_bytes(_canonical(payload))
    _git(
        repo,
        "add",
        AUTHORIZATION_PATH.as_posix(),
        LIFECYCLE_WITNESS_PATH.as_posix(),
    )
    _git(repo, "commit", "-m", "authorize route")
    for accepted_path in (root_pin_path, witness_path, artifact):
        accepted_path.chmod(0o400)
        assert accepted_path.stat().st_mode & 0o022 == 0
    return _AuthorizedRepository(
        repo=repo,
        artifact=artifact,
        reviewer_key=reviewer_key,
        profile=profile,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        source_head=source_head,
        source_tree=source_tree,
    )


def _test_control_plane_capsule(
    tmp_path: Path,
    *,
    source_head: str,
    source_tree: str,
) -> llm_router.AgentImplementationControlPlanePin:
    root = tmp_path / "accepted-control-plane"
    payloads = {
        relative: (f"# inert accepted test source: {relative}\n").encode()
        for relative in llm_router._AGENT_CONTROL_PLANE_RELATIVE_FILES
    }
    digests: dict[str, str] = {}
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o400)
        digests[relative] = "sha256:" + hashlib.sha256(payload).hexdigest()
    manifest: dict[str, Any] = {
        "schema": llm_router._AGENT_CONTROL_PLANE_MANIFEST_SCHEMA,
        "source_head": source_head,
        "source_tree": source_tree,
        "files": dict(sorted(digests.items())),
    }
    manifest["capsule_id"] = llm_router._content_addressed_mapping(
        manifest,
        identity_field="capsule_id",
    )
    manifest_path = root / llm_router._AGENT_CONTROL_PLANE_MANIFEST_FILENAME
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    manifest_path.chmod(0o400)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o500)
    root.chmod(0o500)
    return llm_router.build_agent_implementation_control_plane_pin(
        runner_path=(
            root
            / "ipfs_accelerate_py"
            / "agent_supervisor"
            / "runtime"
            / "grok_cli_runner.py"
        ),
        capsule_root=root,
    )


def _high_plan(repo: Path):
    authorization = llm_router.load_agent_implementation_route_authorization(
        repo_root=repo,
        artifact_path=AUTHORIZATION_PATH.as_posix(),
        board_namespace=BOARD_NAMESPACE,
    )
    return llm_router.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.5",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_or_auth_unavailable",
        fallback_reasoning_effort="high",
        authorization=authorization,
    )


def _signed_high_plan(
    tmp_path: Path,
) -> tuple[
    Path,
    llm_router.AgentImplementationRoutePlan,
    llm_router.AgentImplementationInvocationBinding,
]:
    fixture = _authorized_repo(tmp_path)
    route = _high_plan(fixture.repo)
    control_plane = _test_control_plane_capsule(
        tmp_path,
        source_head=fixture.source_head,
        source_tree=fixture.source_tree,
    )
    baseline = _git(fixture.repo, "rev-parse", "HEAD^{commit}")
    attempt = 1
    task_id = "task:one"
    task_revision_cid = "task-revision:one"
    prompt_cid = "prompt:one"
    worktree_id = content_identity(
        {
            "workspace_path": str(fixture.repo.resolve()),
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
        workspace_path=str(fixture.repo.resolve()),
        repository_cid="repository:one",
        baseline_commit=baseline,
        effects=("edit", "isolated_worktree", "test"),
        scope_cid="scope:one",
        budget_cid="budget:one",
        resource_cid="resource:one",
        authority_cid=fixture.profile.content_id,
        route_id=route.route_id,
        primary_provider_id=route.primary_provider_id,
        primary_model_id=route.primary_model_id,
        fallback_provider_id=route.fallback_provider_id,
        fallback_model_id=route.fallback_model_id,
        fallback_reasoning_effort=route.fallback_reasoning_effort,
        fallback_implementer_identity=route.fallback_implementer_identity,
        reviewer_identity=fixture.profile.reviewer_identity,
        reviewer_provider=fixture.profile.reviewer_provider,
        profile_id=fixture.profile.profile_id,
        profile_identity_did=fixture.profile.identity_did,
        profile_lifecycle_anchor_id=fixture.profile.lifecycle_anchor_id,
        profile_lifecycle_generation=fixture.profile.lifecycle_generation,
        profile_dir=str(fixture.profile_dir.resolve()),
        profile_lifecycle_dir=str(fixture.lifecycle_dir.resolve()),
        issued_at_ms=issued_at_ms,
        expires_at_ms=issued_at_ms + 60_000,
        provider_attempt_store=str(attempt_store),
        provider_attempt_store_identity=attempt_store_identity,
        control_plane=control_plane,
        reviewer_signature="pending",
    )
    invocation = replace(
        unsigned,
        reviewer_signature=_sign(
            fixture.reviewer_key,
            unsigned.signed_payload(),
        ),
    )
    bound = llm_router.bind_agent_implementation_route_invocation(
        route,
        invocation,
        repo_root=fixture.repo,
        workspace=fixture.repo,
        expected_binding=invocation.signed_payload(),
        now_ms=issued_at_ms,
        max_age_ms=60_000,
    )
    return fixture.repo, bound, invocation


def _receipt(stderr: str, *, overflow: bool = False):
    size = 128 * 1024 + 1 if overflow else len(stderr.encode())
    return llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text=stderr,
        nonce="a" * 64,
        model="grok-4.5",
        probe_returncode=41,
        evidence_size=size,
        evidence_overflow=overflow,
    )


def _native_quota_home(
    repo: Path,
    *,
    receipt: dict[str, object],
) -> tuple[Path, str]:
    session_id = "f159e13e-462f-43bc-9da2-01bd0c1f5761"
    home = repo / "native-verifier-home"
    session = home / "sessions" / session_id
    session.mkdir(parents=True)

    def update(value: dict[str, object]) -> dict[str, object]:
        return {
            "method": "session/update",
            "params": {"sessionId": session_id, "update": value},
        }

    events = [
        update(
            {
                "sessionUpdate": "retry_state",
                "type": "failed",
                "error_type": "api",
                "message": SPENDING_LIMIT_MESSAGE,
            }
        ),
        update(
            {
                "sessionUpdate": "turn_completed",
                "stop_reason": "error",
                "agent_result": SPENDING_LIMIT_MESSAGE,
            }
        ),
    ]
    transcript = session / "updates.jsonl"
    transcript.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    transcript.chmod(0o600)
    summary = session / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "info": {"id": session_id},
                "current_model_id": "grok-4.5",
                "grok_home": str(home),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary.chmod(0o600)
    assert receipt["primary_model"] == "grok-4.5"
    return home, session_id


def test_route_resolver_accepts_only_complete_canonical_tuples() -> None:
    legacy = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    )
    assert legacy.fallback_trigger == "primary_quota_exhausted"
    assert legacy.fallback_reasoning_effort == "medium"
    quota_high = llm_router.resolve_agent_implementation_route(
        **{
            **legacy.as_dict(),
            "fallback_reasoning_effort": "high",
        }
    )
    assert quota_high.fallback_trigger == "primary_quota_exhausted"
    assert quota_high.fallback_reasoning_effort == "high"
    assert quota_high.route_id == (
        "agent-supervisor-grok45-terra56-high-hard-quota-v1"
    )
    assert quota_high.authorization is None
    assert quota_high.permits_authentication_unavailable is False
    with pytest.raises(ValueError, match="complete six-field"):
        llm_router.resolve_agent_implementation_route(
            primary_provider_id="grok_cli"
        )
    with pytest.raises(ValueError, match="reviewed legacy"):
        llm_router.resolve_agent_implementation_route(
            **{
                **legacy.as_dict(),
                "fallback_reasoning_effort": "low",
            }
        )
    first = llm_router.create_legacy_agent_implementation_route_invocation()
    second = llm_router.create_legacy_agent_implementation_route_invocation()
    assert first.route_plan == legacy == second.route_plan
    assert first.failure_receipt_nonce != second.failure_receipt_nonce
    assert re.fullmatch(r"[0-9a-f]{64}", first.failure_receipt_nonce)
    assert re.fullmatch(r"[0-9a-f]{64}", second.failure_receipt_nonce)


def test_quota_high_route_denies_auth_and_requires_independent_quota() -> None:
    legacy = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    )
    route = llm_router.resolve_agent_implementation_route(
        **{
            **legacy.as_dict(),
            "fallback_reasoning_effort": "high",
        }
    )

    auth_nonce = "a" * 64
    auth_receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="not signed in",
        nonce=auth_nonce,
        model="grok-4.5",
        probe_returncode=1,
    )
    auth_decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=Path.cwd(),
        failure_receipt=auth_receipt,
        expected_nonce=auth_nonce,
        expected_model="grok-4.5",
        expected_probe_returncode=1,
    )
    assert auth_decision.authorized is False
    assert auth_decision.requires_independent_quota_verification is False
    assert auth_decision.reason_code == "authentication_fallback_not_in_route"

    quota_nonce = "b" * 64
    quota_receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text=SPENDING_LIMIT_MESSAGE,
        nonce=quota_nonce,
        model="grok-4.5",
        probe_returncode=41,
    )
    quota_decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=Path.cwd(),
        failure_receipt=quota_receipt,
        expected_nonce=quota_nonce,
        expected_model="grok-4.5",
        expected_probe_returncode=41,
    )
    assert quota_decision.authorized is False
    assert quota_decision.requires_independent_quota_verification is True
    assert quota_decision.reason_code == "independent_quota_verification_required"


def test_scoped_high_route_binds_artifact_source_and_full_plan(
    tmp_path: Path,
) -> None:
    repo, plan, invocation = _signed_high_plan(tmp_path)
    rebound = llm_router.resolve_agent_implementation_route_binding(
        plan.as_binding_dict(),
        repo_root=repo,
        now_ms=invocation.issued_at_ms,
        max_age_ms=60_000,
    )
    assert rebound == plan
    assert plan.route_id == ROUTE_ID
    assert plan.authorization is not None
    environment = plan.as_environment()
    assert environment[
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE"
    ] == BOARD_NAMESPACE
    assert environment[
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
    ] == "high"
    assert environment[
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD"
    ] == plan.authorization.source_head


def test_exact_auth_authorizes_but_mixed_or_overflowed_evidence_denies(
    tmp_path: Path,
) -> None:
    repo, plan, invocation = _signed_high_plan(tmp_path)
    exact = _receipt("Error: Not signed in")
    decision = llm_router.decide_agent_implementation_fallback(
        plan,
        repo_root=repo,
        failure_receipt=exact,
        expected_nonce="a" * 64,
        expected_model="grok-4.5",
        expected_probe_returncode=41,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=invocation.issued_at_ms,
        max_age_ms=60_000,
    )
    assert decision.authorized is True
    assert decision.verifier_status == "not_required_exact_auth"

    for receipt in (
        _receipt("Not signed in\nHTTP 429"),
        _receipt("HTTP 403\nNot signed in"),
        _receipt("Error: Not signed in", overflow=True),
    ):
        denied = llm_router.decide_agent_implementation_fallback(
            plan,
            repo_root=repo,
            failure_receipt=receipt,
            expected_nonce="a" * 64,
            expected_model="grok-4.5",
            expected_probe_returncode=41,
            expected_invocation_binding=invocation.signed_payload(),
            now_ms=invocation.issued_at_ms,
            max_age_ms=60_000,
        )
        assert denied.authorized is False


def test_native_quota_evidence_is_opaque_and_bound_to_receipt(
    tmp_path: Path,
) -> None:
    repo, plan, invocation = _signed_high_plan(tmp_path)
    receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="a" * 64,
        model="grok-4.5",
        probe_returncode=41,
        observed_at_ms=invocation.issued_at_ms,
    )
    home, session_id = _native_quota_home(repo, receipt=receipt)
    verifier_root = tmp_path / "verifier"
    verifier_workspace = verifier_root / "workspace"
    verifier_workspace.mkdir(parents=True, mode=0o700)
    verifier_prompt = verifier_root / "prompt.txt"
    verifier_prompt.write_text(
        "Reply with exactly the single word OK.\n",
        encoding="utf-8",
    )
    verifier_prompt.chmod(0o600)
    grok = tmp_path / "grok"
    grok.write_text("#!/bin/sh\nexit 41\n", encoding="utf-8")
    grok.chmod(0o700)
    verifier_command = [
        str(grok.resolve()),
        "--model",
        "grok-4.5",
        "--max-turns",
        "1",
        "--cwd",
        str(verifier_workspace.resolve()),
        "--permission-mode",
        "dontAsk",
        "--output-format",
        "streaming-json",
        "--no-plan",
        "--no-subagents",
        "--disable-web-search",
        "--no-memory",
        "--verbatim",
        "--tools",
        "",
        "--prompt-file",
        str(verifier_prompt.resolve()),
        "--session-id",
        session_id,
        "--disallowed-tools",
        llm_router.AGENT_IMPLEMENTATION_QUOTA_VERIFIER_DISALLOWED_TOOLS,
    ]
    evidence = _validate_quota_evidence_in_accepted_child(
        grok_home=home,
        expected_session_id=session_id,
        verifier_returncode=41,
        failure_receipt=receipt,
        invocation_binding=invocation,
        verifier_command=verifier_command,
        verifier_workspace=verifier_workspace,
        verifier_prompt_path=verifier_prompt,
        observed_at_ms=invocation.issued_at_ms,
    )
    assert isinstance(evidence, llm_router.AgentImplementationQuotaEvidence)
    authorized = llm_router.decide_agent_implementation_fallback(
        plan,
        repo_root=repo,
        failure_receipt=receipt,
        expected_nonce="a" * 64,
        expected_model="grok-4.5",
        expected_probe_returncode=41,
        independent_quota_evidence=evidence,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=invocation.issued_at_ms,
        max_age_ms=60_000,
    )
    assert authorized.authorized is True
    assert authorized.verifier_status == "confirmed_quota"

    forged_mapping = evidence.audit_dict()
    forged_copy = replace(evidence, verifier_result="usage_pool_exhausted")
    for forged in (forged_mapping, forged_copy):
        denied = llm_router.decide_agent_implementation_fallback(
            plan,
            repo_root=repo,
            failure_receipt=receipt,
            expected_nonce="a" * 64,
            expected_model="grok-4.5",
            expected_probe_returncode=41,
            independent_quota_evidence=forged,
            expected_invocation_binding=invocation.signed_payload(),
            now_ms=invocation.issued_at_ms,
            max_age_ms=60_000,
        )
        assert denied.authorized is False


@pytest.mark.parametrize("record_name", ("updates.jsonl", "summary.json"))
@pytest.mark.parametrize("mutation", ("growth", "swap"))
def test_native_quota_evidence_rejects_concurrent_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_name: str,
    mutation: str,
) -> None:
    receipt = _receipt("Grok Build usage balance exhausted")
    home, session_id = _native_quota_home(tmp_path, receipt=receipt)
    target = home / "sessions" / session_id / record_name
    target_resolved = target.resolve()
    original = target.read_bytes()
    real_read = os.read
    mutated = False

    def racing_read(descriptor: int, amount: int) -> bytes:
        nonlocal mutated
        data = real_read(descriptor, amount)
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        try:
            opened_path = descriptor_path.resolve(strict=True)
        except OSError:
            opened_path = Path()
        if data and not mutated and opened_path == target_resolved:
            mutated = True
            if mutation == "growth":
                with target.open("ab") as handle:
                    handle.write(b" ")
            else:
                replacement = target.with_name(target.name + ".replacement")
                replacement.write_bytes(original)
                os.replace(replacement, target)
        return data

    monkeypatch.setattr(llm_router.os, "read", racing_read)
    assert (
        llm_router.validate_agent_implementation_quota_evidence(
            grok_home=home,
            expected_session_id=session_id,
            verifier_returncode=41,
            failure_receipt=receipt,
        )
        is None
    )
    assert mutated is True


@pytest.mark.parametrize("drift", ("blob", "symlink", "wrong_tree"))
def test_authorization_loader_rejects_artifact_or_source_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    fixture = _authorized_repo(tmp_path)
    repo = fixture.repo
    artifact = fixture.artifact
    if drift == "blob":
        artifact.chmod(0o600)
        artifact.write_text(artifact.read_text() + "\n", encoding="utf-8")
        artifact.chmod(0o400)
    elif drift == "symlink":
        saved = artifact.with_suffix(".saved")
        artifact.rename(saved)
        artifact.symlink_to(saved.name)
    else:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["authorization_source"]["source_tree"] = "0" * 40
        artifact.chmod(0o600)
        artifact.write_text(json.dumps(payload, sort_keys=True) + "\n")
        artifact.chmod(0o400)
        _git(repo, "add", AUTHORIZATION_PATH.as_posix())
        _git(repo, "commit", "-m", "wrong source tree")
    with pytest.raises(ValueError):
        llm_router.load_agent_implementation_route_authorization(
            repo_root=repo,
            artifact_path=AUTHORIZATION_PATH.as_posix(),
            board_namespace=BOARD_NAMESPACE,
        )


def test_copied_authorization_in_unrelated_repository_denies(
    tmp_path: Path,
) -> None:
    fixture = _authorized_repo(tmp_path)
    source_repo = fixture.repo
    copied = tmp_path / "copied"
    copied.mkdir()
    _git(copied, "init", "-b", "main")
    _git(copied, "config", "user.name", "Copy Test")
    _git(copied, "config", "user.email", "copy@example.invalid")
    copied_paths = (
        AUTHORIZATION_PATH,
        LIFECYCLE_ROOT_PIN_PATH,
        LIFECYCLE_WITNESS_PATH,
    )
    for relative in copied_paths:
        source = source_repo / relative
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    _git(copied, "add", *(relative.as_posix() for relative in copied_paths))
    _git(copied, "commit", "-m", "copied authority")
    for relative in copied_paths:
        (copied / relative).chmod(0o400)
    assert source_repo != copied
    with pytest.raises(ValueError, match="repository binding"):
        llm_router.load_agent_implementation_route_authorization(
            repo_root=copied,
            artifact_path=AUTHORIZATION_PATH.as_posix(),
            board_namespace=BOARD_NAMESPACE,
        )


def test_authorization_loader_denies_head_drift_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _authorized_repo(tmp_path).repo
    real_run = subprocess.run
    drifted = False

    def drifting_run(command, *args, **kwargs):
        nonlocal drifted
        completed = real_run(command, *args, **kwargs)
        if (
            not drifted
            and list(command)[-3:]
            == ["rev-parse", "--verify", "HEAD^{commit}"]
        ):
            drifted = True
            (repo / "HEAD-DRIFT.txt").write_text("drift\n", encoding="utf-8")
            real_run(["git", "add", "HEAD-DRIFT.txt"], cwd=repo, check=True)
            real_run(
                ["git", "commit", "-m", "drift during route validation"],
                cwd=repo,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        return completed

    monkeypatch.setattr(llm_router.subprocess, "run", drifting_run)

    with pytest.raises(ValueError, match="descendant tree"):
        llm_router.load_agent_implementation_route_authorization(
            repo_root=repo,
            artifact_path=AUTHORIZATION_PATH.as_posix(),
            board_namespace=BOARD_NAMESPACE,
        )


def test_generic_side_effecting_router_fallback_remains_denied() -> None:
    assert (
        llm_router.llm_fallback_compatible(
            {"side_effecting": "true", "router_provider": "grok_cli"},
            {"router_provider": "codex_cli"},
        )
        is False
    )
