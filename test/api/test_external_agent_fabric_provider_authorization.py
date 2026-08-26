from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    local_profile as local_profile_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
    export_local_profile_lifecycle_witness,
    initialize_local_profile,
    lifecycle_root_identity_did,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler,
)

from ipfs_accelerate_py import agent_implementation_route as routes


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


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write_eaaef_authorization(repo: Path, tmp_path: Path) -> tuple[str, str]:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "EAAEF Route Test")
    _git(repo, "config", "user.email", "eaaef-route@example.invalid")
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")
    source_head = _git(repo, "rev-parse", "HEAD")
    source_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    authorization_relative = routes.eaaef_agent_route_authorization_path(
        source_tree
    )
    root_pin_relative = routes.eaaef_agent_lifecycle_root_pin_path(source_tree)

    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_identity = ed25519_did_key(reviewer_key.public_key())
    repository_cid = "sha256:" + "1" * 64
    budget_cid = "sha256:" + "2" * 64
    resource_cid = "sha256:" + "3" * 64
    profile = initialize_local_profile(
        repository_cid=repository_cid,
        baseline_commit=source_head,
        profile_dir=tmp_path / "reviewer-profile",
        lifecycle_dir=tmp_path / "reviewer-lifecycle",
        signing_key=reviewer_key.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        ),
        effect_bounds=("edit", "isolated_worktree", "test"),
        budget_cid=budget_cid,
        resource_cid=resource_cid,
        route_id=routes._EAAEF_AGENT_IMPLEMENTATION_ROUTE_ID,
        reviewer_identity=reviewer_identity,
        reviewer_provider="local_operator",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_reasoning_effort="high",
    )

    root_identity = lifecycle_root_identity_did()
    event_time_ms = int(time.time()) * 1000
    root_pin = {
        "schema": routes._AGENT_LIFECYCLE_ROOT_PIN_SCHEMA,
        "board_namespace": routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        "base_head": source_head,
        "base_tree": source_tree,
        "root_identity_did": root_identity,
        "pinned_at_ms": event_time_ms,
    }
    root_pin["pin_id"] = routes._content_addressed_mapping(
        root_pin,
        identity_field="pin_id",
    )
    root_pin_path = repo / root_pin_relative
    root_pin_path.parent.mkdir(parents=True)
    root_pin_path.write_bytes(_canonical(root_pin))
    _git(repo, "add", root_pin_relative)
    _git(repo, "commit", "-m", "pin EAAEF lifecycle root")
    root_pin_path.chmod(0o400)
    root_pin_digest = "sha256:" + hashlib.sha256(
        root_pin_path.read_bytes()
    ).hexdigest()

    nonce = "eaaef:" + hashlib.sha256(str(repo).encode()).hexdigest()
    witness = export_local_profile_lifecycle_witness(
        repository_cid=repository_cid,
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        base_head=source_head,
        base_tree=source_tree,
        nonce=nonce,
        profile_dir=tmp_path / "reviewer-profile",
        lifecycle_dir=tmp_path / "reviewer-lifecycle",
        observed_at_ms=event_time_ms,
        expires_at_ms=event_time_ms + 10 * 60 * 1000,
    )
    witness_relative = (
        routes._EAAEF_AGENT_LIFECYCLE_WITNESS_PREFIX
        + source_tree
        + "-test.json"
    )
    witness_path = repo / witness_relative
    witness_path.write_bytes(_canonical(witness))
    witness_digest = "sha256:" + hashlib.sha256(
        witness_path.read_bytes()
    ).hexdigest()

    route = {
        "route_id": routes._EAAEF_AGENT_IMPLEMENTATION_ROUTE_ID,
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.6",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
        "allowed_trigger_classes": [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ],
    }
    authority_bounds = {
        "repository_cid": repository_cid,
        "baseline_commit": source_head,
        "effects": ["edit", "isolated_worktree", "test"],
        "budget_cid": budget_cid,
        "resource_cid": resource_cid,
        "authority_cid": profile.content_id,
    }
    review_payload = routes.agent_implementation_route_review_payload(
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
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
        reviewer_witness_path=witness_relative,
        reviewer_witness_sha256=witness_digest,
        lifecycle_root_identity_did=root_identity,
        lifecycle_witness_nonce=nonce,
        lifecycle_root_pin_path=(
            root_pin_relative
        ),
        lifecycle_root_pin_sha256=root_pin_digest,
        authorized_at_ms=event_time_ms,
        fallback_implementer_identity="codex",
    )
    signature = base64.b64encode(
        reviewer_key.sign(_canonical(review_payload))
    ).decode("ascii")
    authorization = {
        "schema": routes._AGENT_ROUTE_AUTHORIZATION_SCHEMA,
        "board_namespace": routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
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
            "witness_path": witness_relative,
            "witness_sha256": witness_digest,
            "signature": signature,
        },
        "authority_bounds": authority_bounds,
        "fallback_implementer_identity": "codex",
        "lifecycle_root_identity_did": root_identity,
        "lifecycle_witness_nonce": nonce,
        "lifecycle_root_pin_path": (
            root_pin_relative
        ),
        "lifecycle_root_pin_sha256": root_pin_digest,
        "authorized_at_ms": event_time_ms,
    }
    artifact = repo / authorization_relative
    artifact.write_bytes(_canonical(authorization))
    _git(repo, "add", authorization_relative, witness_relative)
    _git(repo, "commit", "-m", "authorize EAAEF provider route")
    for path in (artifact, witness_path, root_pin_path):
        path.chmod(0o400)
    return source_head, source_tree


def _add_umask_0002_worktree(repo: Path, destination: Path) -> Path:
    created = subprocess.run(
        [
            "bash",
            "-c",
            'umask 0002; exec git worktree add --detach "$1" HEAD',
            "eaaef-route-worktree",
            str(destination),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    return destination


def _eaaef_signed_paths(repo: Path, source_tree: str) -> tuple[Path, Path, Path]:
    artifact = repo / routes.eaaef_agent_route_authorization_path(source_tree)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return (
        artifact,
        repo / payload["lifecycle_root_pin_path"],
        repo / payload["reviewer"]["witness_path"],
    )


def test_signed_eaaef_grok46_route_loads_without_changing_legacy_route(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)

    authorization = routes.load_agent_implementation_route_authorization(
        repo_root=repo,
        artifact_path=routes.eaaef_agent_route_authorization_path(source_tree),
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
    )
    plan = routes.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_or_auth_unavailable",
        fallback_reasoning_effort="high",
        authorization=authorization,
    )
    legacy = routes.resolve_agent_implementation_route(default_route="legacy")

    assert plan.route_id == routes._EAAEF_AGENT_IMPLEMENTATION_ROUTE_ID
    assert plan.authorization is authorization
    assert authorization.source_head == source_head
    assert authorization.source_tree == source_tree
    assert legacy.route_id == routes._LEGACY_AGENT_IMPLEMENTATION_ROUTE_ID
    assert legacy.primary_model_id == "grok-4.5"
    assert legacy.fallback_reasoning_effort == "medium"


def test_signed_eaaef_route_loads_from_real_umask_0002_worktree_without_chmod(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    worktree = _add_umask_0002_worktree(
        repo,
        tmp_path / "umask-0002-worktree",
    )
    signed_paths = _eaaef_signed_paths(worktree, source_tree)

    assert all(path.stat().st_mode & 0o777 == 0o664 for path in signed_paths)

    authorization = routes.load_agent_implementation_route_authorization(
        repo_root=worktree,
        artifact_path=routes.eaaef_agent_route_authorization_path(source_tree),
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
    )

    assert authorization.source_tree == source_tree
    assert all(path.stat().st_mode & 0o777 == 0o664 for path in signed_paths)
    assert _git(worktree, "status", "--porcelain") == ""


@pytest.mark.parametrize(
    "mutation",
    ("dirty", "world_writable", "symlink", "hardlink"),
)
def test_source_addressed_eaaef_route_still_rejects_untrusted_worktree_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    worktree = _add_umask_0002_worktree(
        repo,
        tmp_path / f"{mutation}-worktree",
    )
    artifact, _root_pin, _witness = _eaaef_signed_paths(worktree, source_tree)

    if mutation == "dirty":
        # Trailing JSON whitespace preserves the signed payload but not the
        # exact Git blob, so the repository binding must still reject it.
        artifact.write_bytes(artifact.read_bytes() + b"\n")
    elif mutation == "world_writable":
        artifact.chmod(0o666)
    else:
        saved = artifact.with_suffix(".saved")
        artifact.rename(saved)
        if mutation == "symlink":
            artifact.symlink_to(saved.name)
        else:
            artifact.hardlink_to(saved)

    with pytest.raises(ValueError):
        routes.load_agent_implementation_route_authorization(
            repo_root=worktree,
            artifact_path=routes.eaaef_agent_route_authorization_path(
                source_tree
            ),
            board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        )


def test_source_addressed_eaaef_route_rejects_inter_read_artifact_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    worktree = _add_umask_0002_worktree(
        repo,
        tmp_path / "artifact-race-worktree",
    )
    artifact, _root_pin, _witness = _eaaef_signed_paths(worktree, source_tree)
    real_read = routes._agent_read_stable_file
    artifact_reads = 0

    def racing_read(path: Path, **kwargs: object) -> bytes:
        nonlocal artifact_reads
        if Path(path) == artifact:
            artifact_reads += 1
            if artifact_reads == 2:
                artifact.write_bytes(artifact.read_bytes() + b"\n")
        return real_read(path, **kwargs)

    monkeypatch.setattr(routes, "_agent_read_stable_file", racing_read)

    with pytest.raises(ValueError, match="not bound to this descendant tree"):
        routes.load_agent_implementation_route_authorization(
            repo_root=worktree,
            artifact_path=routes.eaaef_agent_route_authorization_path(
                source_tree
            ),
            board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        )

    assert artifact_reads == 2


def test_source_addressed_eaaef_route_rejects_head_drift_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    worktree = _add_umask_0002_worktree(
        repo,
        tmp_path / "head-race-worktree",
    )
    real_git_output = routes._agent_git_output
    head_reads = 0

    def drifting_git_output(
        root: Path,
        arguments: tuple[str, ...],
        **kwargs: object,
    ) -> bytes:
        nonlocal head_reads
        if tuple(arguments) == ("rev-parse", "--verify", "HEAD^{commit}"):
            head_reads += 1
            if head_reads == 2:
                (worktree / "head-drift.txt").write_text(
                    "drift\n",
                    encoding="utf-8",
                )
                _git(worktree, "add", "head-drift.txt")
                _git(worktree, "commit", "-m", "move head during validation")
        return real_git_output(root, arguments, **kwargs)

    monkeypatch.setattr(routes, "_agent_git_output", drifting_git_output)

    with pytest.raises(ValueError, match="not bound to this descendant tree"):
        routes.load_agent_implementation_route_authorization(
            repo_root=worktree,
            artifact_path=routes.eaaef_agent_route_authorization_path(
                source_tree
            ),
            board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        )

    assert head_reads == 2


@pytest.mark.parametrize("signed_path_index", (0, 1, 2))
def test_source_addressed_eaaef_route_requires_nonexecutable_git_blobs(
    tmp_path: Path,
    signed_path_index: int,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    signed_path = _eaaef_signed_paths(repo, source_tree)[signed_path_index]
    signed_path.chmod(0o500)
    relative = signed_path.relative_to(repo).as_posix()
    _git(repo, "add", relative)
    _git(repo, "commit", "-m", "make signed authority executable")

    assert _git(repo, "ls-tree", "HEAD", "--", relative).startswith(
        "100755 blob "
    )
    with pytest.raises(ValueError, match="not bound to this descendant tree"):
        routes.load_agent_implementation_route_authorization(
            repo_root=repo,
            artifact_path=routes.eaaef_agent_route_authorization_path(
                source_tree
            ),
            board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        )


@pytest.mark.parametrize(
    ("board_namespace", "artifact_path"),
    (
        (
            routes._V3_AGENT_ROUTE_BOARD_NAMESPACE,
            routes._V3_AGENT_ROUTE_AUTHORIZATION_PATH,
        ),
        (
            routes._VGO_AGENT_ROUTE_BOARD_NAMESPACE,
            routes._VGO_AGENT_ROUTE_AUTHORIZATION_PATH,
        ),
    ),
)
def test_fixed_path_route_authority_remains_strictly_non_group_writable(
    tmp_path: Path,
    board_namespace: str,
    artifact_path: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate = repo / artifact_path
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{}", encoding="utf-8")
    candidate.chmod(0o660)

    with pytest.raises(ValueError, match="not immutable enough"):
        routes.load_agent_implementation_route_authorization(
            repo_root=repo,
            artifact_path=artifact_path,
            board_namespace=board_namespace,
        )


def test_eaaef_capacity_rejects_primary_model_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    authorization = routes.load_agent_implementation_route_authorization(
        repo_root=repo,
        artifact_path=routes.eaaef_agent_route_authorization_path(source_tree),
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
    )
    plan = routes.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_or_auth_unavailable",
        fallback_reasoning_effort="high",
        authorization=authorization,
    )
    observed_at_ms = int(time.time()) * 1000
    observations = [
        {
            "provider_id": provider_id,
            "healthy": True,
            "quota_remaining": 1,
            "latency_ms": 1,
            "context_window_tokens": 1,
            "token_budget_remaining": 1,
            "max_concurrency": 1,
            "active_requests": 0,
            "capabilities": ["implementation"],
            "observed_at_ms": observed_at_ms,
            "retry_after_ms": 0,
            "available_concurrency": 1,
        }
        for provider_id in ("grok_cli", "codex_cli")
    ]

    with pytest.raises(
        ValueError,
        match="route capacity authorization scope drifted",
    ):
        routes.project_agent_implementation_route_capacity(
            replace(plan, primary_model_id="grok-4.5"),
            observations=observations,
            now_ms=observed_at_ms,
            max_age_ms=60_000,
        )


def test_eaaef_authorization_cannot_cross_into_prompt_v3_route(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    authorization = routes.load_agent_implementation_route_authorization(
        repo_root=repo,
        artifact_path=routes.eaaef_agent_route_authorization_path(source_tree),
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
    )

    with pytest.raises(ValueError, match="scoped operator authorization"):
        routes.resolve_agent_implementation_route(
            primary_provider_id="grok_cli",
            primary_model_id="grok-4.5",
            fallback_provider_id="codex",
            fallback_model_id="gpt-5.6-terra",
            fallback_trigger="primary_quota_or_auth_unavailable",
            fallback_reasoning_effort="high",
            authorization=authorization,
        )


def test_eaaef_authorization_path_cannot_be_loaded_under_legacy_namespace(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)

    with pytest.raises(ValueError, match="not authorized for this board scope"):
        routes.load_agent_implementation_route_authorization(
            repo_root=repo,
            artifact_path=routes.eaaef_agent_route_authorization_path(source_tree),
            board_namespace=routes._V3_AGENT_ROUTE_BOARD_NAMESPACE,
        )


def test_eaaef_authorization_path_is_bound_to_its_source_tree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    wrong_tree = "f" * len(source_tree)
    wrong_path = repo / routes.eaaef_agent_route_authorization_path(wrong_tree)
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.write_bytes(
        (
            repo
            / routes.eaaef_agent_route_authorization_path(source_tree)
        ).read_bytes()
    )
    wrong_path.chmod(0o400)

    with pytest.raises(
        ValueError,
        match="does not grant the exact scoped route",
    ):
        routes.load_agent_implementation_route_authorization(
            repo_root=repo,
            artifact_path=routes.eaaef_agent_route_authorization_path(
                wrong_tree
            ),
            board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        )


def test_configured_board_derives_post_freeze_route_path_without_config_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_head, source_tree = _write_eaaef_authorization(repo, tmp_path)
    monkeypatch.setattr(
        scheduler,
        "_git_identity",
        lambda _repo_root: (source_head, source_tree),
    )

    plan = scheduler._resolved_ordered_provider_route(
        {
            "primary_provider_id": "grok_cli",
            "primary_model_id": "grok-4.6",
            "fallback_provider_id": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_trigger": "primary_quota_or_auth_unavailable",
            "fallback_reasoning_effort": "high",
        },
        repo_root=repo,
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
    )

    assert plan.authorization is not None
    assert plan.authorization.artifact_path == (
        routes.eaaef_agent_route_authorization_path(source_tree)
    )
