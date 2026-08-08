from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router

AUTHORIZATION_PATH = Path(
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "provider_fallback_policy_authorization_20260808.json"
)
BOARD_NAMESPACE = "agent-supervisor-prompt-only-self-improvement-v3"
ROUTE_ID = (
    "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
)
SPENDING_LIMIT_MESSAGE = (
    "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
    "You have run out of credits or need a Grok subscription. Add credits at "
    "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
)


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


def _authorized_repo(tmp_path: Path) -> tuple[Path, Path]:
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
    artifact = repo / AUTHORIZATION_PATH
    artifact.parent.mkdir(parents=True)
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-policy-authorization@1"
        ),
        "board_namespace": BOARD_NAMESPACE,
        "authorization_source": {
            "kind": "explicit_operator_override",
            "source_head": source_head,
            "source_tree": source_tree,
            "prospective_only": True,
            "requires_descendant_tree": True,
        },
        "route": {
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
        },
        "ownership_contract": {
            "canonical_route_plan_owner": "ipfs_accelerate_py.llm_router",
            "typed_fallback_decision_owner": "ipfs_accelerate_py.llm_router",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed": False,
        },
        "bootstrap_route_guarantees": {
            "explicit_codex_review_conflict_denied": True,
        },
    }
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", AUTHORIZATION_PATH.as_posix())
    _git(repo, "commit", "-m", "authorize route")
    return repo, artifact


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
    (session / "updates.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    (session / "summary.json").write_text(
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
    assert receipt["primary_model"] == "grok-4.5"
    return home, session_id


def test_route_resolver_accepts_only_complete_canonical_tuples() -> None:
    legacy = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    )
    assert legacy.fallback_trigger == "primary_quota_exhausted"
    assert legacy.fallback_reasoning_effort == "medium"
    with pytest.raises(ValueError, match="complete six-field"):
        llm_router.resolve_agent_implementation_route(
            primary_provider_id="grok_cli"
        )
    with pytest.raises(ValueError, match="reviewed legacy"):
        llm_router.resolve_agent_implementation_route(
            **{
                **legacy.as_dict(),
                "fallback_reasoning_effort": "high",
            }
        )
    first = llm_router.create_legacy_agent_implementation_route_invocation()
    second = llm_router.create_legacy_agent_implementation_route_invocation()
    assert first.route_plan == legacy == second.route_plan
    assert first.failure_receipt_nonce != second.failure_receipt_nonce
    assert re.fullmatch(r"[0-9a-f]{64}", first.failure_receipt_nonce)
    assert re.fullmatch(r"[0-9a-f]{64}", second.failure_receipt_nonce)


def test_scoped_high_route_binds_artifact_source_and_full_plan(
    tmp_path: Path,
) -> None:
    repo, _artifact = _authorized_repo(tmp_path)
    plan = _high_plan(repo)
    rebound = llm_router.resolve_agent_implementation_route_binding(
        plan.as_binding_dict(),
        repo_root=repo,
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
    repo, _artifact = _authorized_repo(tmp_path)
    plan = _high_plan(repo)
    exact = _receipt("Error: Not signed in")
    decision = llm_router.decide_agent_implementation_fallback(
        plan,
        repo_root=repo,
        failure_receipt=exact,
        expected_nonce="a" * 64,
        expected_model="grok-4.5",
        expected_probe_returncode=41,
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
        )
        assert denied.authorized is False


def test_native_quota_evidence_is_opaque_and_bound_to_receipt(
    tmp_path: Path,
) -> None:
    repo, _artifact = _authorized_repo(tmp_path)
    plan = _high_plan(repo)
    receipt = _receipt("Grok Build usage balance exhausted")
    home, session_id = _native_quota_home(repo, receipt=receipt)
    evidence = llm_router.validate_agent_implementation_quota_evidence(
        grok_home=home,
        expected_session_id=session_id,
        verifier_returncode=41,
        failure_receipt=receipt,
    )
    assert evidence is not None
    authorized = llm_router.decide_agent_implementation_fallback(
        plan,
        repo_root=repo,
        failure_receipt=receipt,
        expected_nonce="a" * 64,
        expected_model="grok-4.5",
        expected_probe_returncode=41,
        independent_quota_evidence=evidence,
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
    repo, artifact = _authorized_repo(tmp_path)
    if drift == "blob":
        artifact.write_text(artifact.read_text() + "\n", encoding="utf-8")
    elif drift == "symlink":
        saved = artifact.with_suffix(".saved")
        artifact.rename(saved)
        artifact.symlink_to(saved.name)
    else:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["authorization_source"]["source_tree"] = "0" * 40
        artifact.write_text(json.dumps(payload, sort_keys=True) + "\n")
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
    source_repo, artifact = _authorized_repo(tmp_path)
    copied = tmp_path / "copied"
    copied.mkdir()
    _git(copied, "init", "-b", "main")
    _git(copied, "config", "user.name", "Copy Test")
    _git(copied, "config", "user.email", "copy@example.invalid")
    target = copied / AUTHORIZATION_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(artifact.read_bytes())
    _git(copied, "add", AUTHORIZATION_PATH.as_posix())
    _git(copied, "commit", "-m", "copied authority")
    assert source_repo != copied
    with pytest.raises(ValueError, match="descendant tree"):
        llm_router.load_agent_implementation_route_authorization(
            repo_root=copied,
            artifact_path=AUTHORIZATION_PATH.as_posix(),
            board_namespace=BOARD_NAMESPACE,
        )


def test_authorization_loader_denies_head_drift_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _artifact = _authorized_repo(tmp_path)
    real_run = subprocess.run
    drifted = False

    def drifting_run(command, *args, **kwargs):
        nonlocal drifted
        completed = real_run(command, *args, **kwargs)
        if (
            not drifted
            and list(command)
            == ["git", "rev-parse", "--verify", "HEAD^{commit}"]
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
