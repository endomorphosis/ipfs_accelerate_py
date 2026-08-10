from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.core.protected_acceptance_contracts import (
    ArtifactBytes,
    EvidenceHandle,
    GitFileIdentity,
    PhaseAuthority,
    PhaseCandidateRequest,
    PhasePolicy,
    ProductGenerationRecord,
    ProductProvenance,
    ProductProvenanceRequest,
    PromptV3Phase,
    PromptV3QInventory,
    ProtectedAcceptanceDenied,
    RepositoryBinding,
    SignedArtifactRequest,
    StableQPolicy,
    canonical_json_bytes,
    content_id,
    phase_authority_content_id,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.protected_acceptance_transition import (
    build_prompt_v3_phase_candidate,
    build_prompt_v3_root_pin,
    canonical_prompt_v3_review_bytes,
    freeze_prompt_v3_product_provenance,
    publish_prompt_v3_phase_candidate,
    reject_prompt_v3_phase_candidate,
    run_prompt_v3_phase_evidence,
    sign_prompt_v3_operator_artifact,
    validate_prompt_v3_phase_candidate,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.protected_acceptance_transition_cli import (
    build_parser,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.protected_acceptance_transition_cli import (
    main as transition_cli_main,
)
REPO_ROOT = Path(__file__).resolve().parents[2]

from ipfs_accelerate_py.agent_supervisor.merge.protected_acceptance_transition import (
    ProtectedTransitionGitError,
    ProtectedTransitionRace,
    TransitionHooks,
)


def _git(repo: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    actual_env = dict(os.environ)
    actual_env.pop("GIT_INDEX_FILE", None)
    if env:
        actual_env.update(env)
    return subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repo,
        env=actual_env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _repo(tmp_path: Path, *, object_format: str | None = None) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)
    init_arguments = ["init", "-q", "-b", "main"]
    if object_format is not None:
        init_arguments.append(f"--object-format={object_format}")
    _git(repo, *init_arguments)
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="ascii")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-q", "-m", "base")
    parent = _git(repo, "rev-parse", "HEAD")
    target = "refs/heads/protected/acceptance"
    _git(repo, "update-ref", target, parent)
    _git(repo, "checkout", "-q", "--detach", parent)
    for directory, child_directories, _files in os.walk(repo / ".git"):
        Path(directory).chmod(0o755)
        for child in child_directories:
            (Path(directory) / child).chmod(0o755)
    return repo, parent, target


def _handle(kind: str = "suite", tag: str = "one") -> EvidenceHandle:
    raw = tag.encode("ascii")
    return EvidenceHandle(
        kind=kind,
        content_id=content_id(raw),
        byte_length=len(raw),
    )


def _request(
    repo: Path,
    parent: str,
    target: str,
    *,
    phase: PromptV3Phase = PromptV3Phase.Q,
    parent_phase: PromptV3Phase | None = None,
    dry_run: bool = False,
) -> PhaseCandidateRequest:
    now_ns = time.time_ns()
    nonce = (
        base64.urlsafe_b64encode(hashlib.sha256(phase.value.encode()).digest())
        .decode()
        .rstrip("=")
    )
    identity_did = "did:key:z6MkhTestIdentity"
    authority = PhaseAuthority(
        phase=phase,
        authority_id=phase_authority_content_id(
            phase=phase,
            nonce=nonce,
            parent_commit=parent,
            identity_did=identity_did,
            issued_at_ns=now_ns - 1_000_000_000,
            expires_at_ns=now_ns + 3_600_000_000_000,
        ),
        nonce=nonce,
        parent_commit=parent,
        identity_did=identity_did,
        issued_at_ns=now_ns - 1_000_000_000,
        expires_at_ns=now_ns + 3_600_000_000_000,
    )
    path = f"protected/{phase.value.replace('/', '-').lower()}.json"
    return PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target),
        policy=PhasePolicy(
            phase=phase,
            expected_parent_phase=parent_phase,
            allowed_paths=(path,),
            required_evidence_kinds=("suite",),
            validator_ids=("validator",),
        ),
        parent_commit=parent,
        parent_phase=parent_phase,
        authority=authority,
        artifacts=(ArtifactBytes(path=path, data=b'{"ok":true}\n'),),
        evidence_handles=(_handle(),),
        commit_message=f"protected {phase.value}",
        commit_timestamp="1700000000 +0000",
        observed_at_ns=now_ns,
        dry_run=dry_run,
    )


def _authority_validator(authority: PhaseAuthority, now_ns: int) -> bool:
    return authority.issued_at_ns <= now_ns < authority.expires_at_ns


_NO_TRANSITION_HOOKS = TransitionHooks()


def _build(
    request: PhaseCandidateRequest, *, hooks: TransitionHooks = _NO_TRANSITION_HOOKS
):
    return build_prompt_v3_phase_candidate(
        request, authority_validator=_authority_validator, hooks=hooks
    )


def _bound_evidence_bytes(candidate, kind: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.phase-evidence-binding@1",
            "candidate_commit": candidate.commit_id,
            "authority_id": candidate.request.authority.authority_id,
            "kind": kind,
        }
    )


def _bound_handle(candidate, kind: str) -> EvidenceHandle:
    raw = _bound_evidence_bytes(candidate, kind)
    return EvidenceHandle(kind=kind, content_id=content_id(raw), byte_length=len(raw))


def _evidence_loader(candidate, handle: EvidenceHandle) -> bytes:
    return _bound_evidence_bytes(candidate, handle.kind)


def _validated(request: PhaseCandidateRequest):
    candidate = _build(request)
    evidence = run_prompt_v3_phase_evidence(
        candidate,
        runner=lambda observed: (_bound_handle(observed, "suite"),),
        evidence_loader=_evidence_loader,
    )
    validated = validate_prompt_v3_phase_candidate(
        candidate,
        evidence,
        validator=lambda observed, _evidence: (_bound_handle(observed, "validator"),),
        evidence_loader=_evidence_loader,
    )
    return candidate, validated


def _publish(
    validated,
    *,
    authority_validator=_authority_validator,
    pre_cas_validator=lambda _value: True,
    hooks=None,
):
    arguments = {
        "authority_validator": authority_validator,
        "pre_cas_validator": pre_cas_validator,
    }
    if hooks is not None:
        arguments["hooks"] = hooks
    return publish_prompt_v3_phase_candidate(validated, **arguments)


def _record(
    role: str,
    commit: str,
    *,
    mode: str = "100644",
    patch: bytes = b"patch",
) -> ProductGenerationRecord:
    file_identity = GitFileIdentity(
        path="product.py",
        mode=mode,
        blob_id="b" * 40,
        raw_content_id=content_id(b"product"),
        byte_length=7,
    )
    return ProductGenerationRecord(
        role=role,
        commit=commit,
        parent="a" * 40,
        tree="c" * 40,
        files=(file_identity,),
        test_evidence=(_handle("tests", role),),
        canonical_patch_content_id=content_id(patch),
    )


def _provenance(task_id: str) -> ProductProvenance:
    return ProductProvenance(
        task_id=task_id,
        source=_record("source", "1" * 40),
        replay=_record("replay", "2" * 40),
        integrated=_record("integrated", "3" * 40),
        canonical_diff_content_id=content_id(b"patch"),
    )


def test_q_inventory_has_exact_no_future_authority_surface() -> None:
    inventory = PromptV3QInventory(
        lifecycle_root_identity_did="did:key:z6MkhRoot",
        stable_policy=StableQPolicy(policy_id=content_id(b"policy")),
        product_provenance=tuple(
            _provenance(task)
            for task in sorted(
                {"ASE3-019", "ASE3-030", "ASE3-031", "ASE3-032", "ASE3-023", "ASE3-027"}
            )
        ),
    )
    assert set(inventory.to_dict()) == {
        "schema",
        "lifecycle_root_identity_did",
        "stable_policy",
        "product_provenance",
    }
    hostile = inventory.to_dict()
    hostile["reviewer_identity"] = "did:key:zFuture"
    with pytest.raises(ProtectedAcceptanceDenied):
        PromptV3QInventory.from_mapping(hostile)


def test_q_provenance_rejects_self_pin_and_integrated_or_mode_tamper() -> None:
    with pytest.raises(ProtectedAcceptanceDenied):
        ProductProvenance(
            task_id="ASE3-033",
            source=_record("source", "1" * 40),
            replay=_record("replay", "2" * 40),
            integrated=_record("integrated", "3" * 40),
            canonical_diff_content_id=content_id(b"patch"),
        )
    with pytest.raises(ProtectedAcceptanceDenied):
        ProductProvenance(
            task_id="ASE3-031",
            source=_record("source", "1" * 40),
            replay=_record("replay", "2" * 40),
            integrated=_record("integrated", "3" * 40, patch=b"other"),
            canonical_diff_content_id=content_id(b"patch"),
        )
    with pytest.raises(ProtectedAcceptanceDenied):
        ProductProvenance(
            task_id="ASE3-031",
            source=_record("source", "1" * 40),
            replay=_record("replay", "2" * 40),
            integrated=_record("integrated", "2" * 40),
            canonical_diff_content_id=content_id(b"patch"),
        )


def test_provenance_freeze_exactly_binds_requested_inspection() -> None:
    request = ProductProvenanceRequest(
        task_id="ASE3-031",
        source_commit="1" * 40,
        replay_commit="2" * 40,
        integrated_commit="3" * 40,
        product_paths=("product.py",),
        source_test_evidence=(_handle("tests", "source"),),
        replay_test_evidence=(_handle("tests", "replay"),),
        integrated_test_evidence=(_handle("tests", "integrated"),),
    )
    expected = _provenance("ASE3-031")
    assert (
        freeze_prompt_v3_product_provenance(
            request,
            inspector=lambda observed: expected if observed == request else None,
        )
        == expected
    )
    substituted = ProductProvenance(
        task_id="ASE3-031",
        source=_record("source", "4" * 40),
        replay=_record("replay", "2" * 40),
        integrated=_record("integrated", "3" * 40),
        canonical_diff_content_id=content_id(b"patch"),
    )
    with pytest.raises(ProtectedAcceptanceDenied):
        freeze_prompt_v3_product_provenance(
            request, inspector=lambda _request: substituted
        )
    with pytest.raises(ProtectedAcceptanceDenied):
        ProductProvenance(
            task_id="ASE3-031",
            source=_record("source", "1" * 40, mode="100755"),
            replay=_record("replay", "2" * 40, mode="100644"),
            integrated=_record("integrated", "3" * 40, mode="100755"),
            canonical_diff_content_id=content_id(b"patch"),
        )


def test_phase_types_reject_skip_and_traversal(tmp_path: Path) -> None:
    with pytest.raises(ProtectedAcceptanceDenied):
        PhasePolicy(
            phase=PromptV3Phase.A019,
            expected_parent_phase=PromptV3Phase.Q,
            allowed_paths=("safe.json",),
            required_evidence_kinds=(),
            validator_ids=("validator",),
        )
    with pytest.raises(ProtectedAcceptanceDenied):
        ArtifactBytes(path="../receipt.json", data=b"x")
    repo, parent, target = _repo(tmp_path)
    with pytest.raises(ProtectedAcceptanceDenied):
        replace(_request(repo, parent, target), observed_at_ns=2)


def test_builder_requires_fresh_verifier_authenticated_authority(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    request = _request(repo, parent, target)
    with pytest.raises(ProtectedAcceptanceDenied, match="verifier-authenticated"):
        build_prompt_v3_phase_candidate(
            request, authority_validator=lambda _authority, _now: False
        )
    old_authority = PhaseAuthority(
        phase=request.authority.phase,
        authority_id=phase_authority_content_id(
            phase=request.authority.phase,
            nonce=request.authority.nonce,
            parent_commit=parent,
            identity_did=request.authority.identity_did,
            issued_at_ns=1,
            expires_at_ns=2,
        ),
        nonce=request.authority.nonce,
        parent_commit=parent,
        identity_did=request.authority.identity_did,
        issued_at_ns=1,
        expires_at_ns=2,
    )
    stale_request = replace(request, authority=old_authority, observed_at_ns=1)
    with pytest.raises(ProtectedAcceptanceDenied, match="verifier-authenticated"):
        _build(stale_request)


def test_root_pin_matches_exact_schema_and_digest() -> None:
    nonce = "A" * 22
    authority = PhaseAuthority(
        phase=PromptV3Phase.R,
        authority_id=phase_authority_content_id(
            phase=PromptV3Phase.R,
            nonce=nonce,
            parent_commit="a" * 40,
            identity_did="did:key:zRoot",
            issued_at_ns=1,
            expires_at_ns=2,
        ),
        nonce=nonce,
        parent_commit="a" * 40,
        identity_did="did:key:zRoot",
        issued_at_ns=1,
        expires_at_ns=2,
    )
    raw = build_prompt_v3_root_pin(
        SignedArtifactRequest(
            phase=PromptV3Phase.R,
            authority=authority,
            body={
                "board_namespace": "prompt_only_self_improvement_v3",
                "base_head": "a" * 40,
                "base_tree": "b" * 40,
                "root_identity_did": "did:key:zRoot",
                "pinned_at_ms": 1,
            },
        )
    )
    payload = json.loads(raw)
    pin_id = payload.pop("pin_id")
    assert payload.pop("schema").endswith("local-profile-lifecycle-root-pin@1")
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.local-profile-lifecycle-root-pin@1",
        **payload,
    }
    assert pin_id == content_id(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def test_operator_signer_transcodes_exact_ed25519_bytes() -> None:
    private = Ed25519PrivateKey.generate()
    payload = {
        "schema": "receipt@1",
        "value": "ascii",
        "review": {"identity_did": "did:key:zReviewer", "signature": ""},
    }
    observed: list[bytes] = []

    def signer(unsigned):
        canonical = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        observed.append(canonical)
        return {
            "identity": "did:key:zReviewer",
            "profile_id": "profile",
            "signature": base64.b64encode(private.sign(canonical)).decode(),
        }

    signed = sign_prompt_v3_operator_artifact(payload, signer=signer)
    token = signed["review"]["signature"]
    assert token.startswith("ed25519:") and "=" not in token
    assert observed == [canonical_prompt_v3_review_bytes(payload)]
    signature = base64.urlsafe_b64decode(token.removeprefix("ed25519:") + "==")
    private.public_key().verify(signature, observed[0])


def test_operator_signer_rejects_unicode_wrong_authority_and_raw_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unicode_payload = {"review": {"signature": ""}, "value": "snowman-☃"}
    with pytest.raises(ProtectedAcceptanceDenied):
        canonical_prompt_v3_review_bytes(unicode_payload)
    payload = {"review": {"identity_did": "did:key:zExpected", "signature": ""}}
    with pytest.raises(ProtectedAcceptanceDenied):
        sign_prompt_v3_operator_artifact(
            payload,
            signer=lambda _unsigned: {
                "identity": "did:key:zWrong",
                "profile_id": "profile",
                "signature": base64.b64encode(b"x" * 64).decode(),
            },
        )
    monkeypatch.setenv("AGENT_SUPERVISOR_LOCAL_PROFILE_KEY", "raw-secret")
    with pytest.raises(ProtectedAcceptanceDenied):
        sign_prompt_v3_operator_artifact(payload)


def test_builder_uses_alternate_index_and_explicit_100644(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    ambient_index = repo / _git(repo, "rev-parse", "--git-path", "index")
    before = ambient_index.read_bytes()
    candidate = _build(_request(repo, parent, target))
    assert ambient_index.read_bytes() == before
    assert candidate.file_identities[0].mode == "100644"
    assert _git(
        repo, "rev-list", "--parents", "-n", "1", candidate.commit_id
    ).split() == [
        candidate.commit_id,
        parent,
    ]
    reject_prompt_v3_phase_candidate(candidate)


def test_builder_seals_alternate_index_under_group_writable_umask(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    previous_umask = os.umask(0o002)
    try:
        candidate = _build(_request(repo, parent, target))
    finally:
        os.umask(previous_umask)
    assert all(identity.mode == "100644" for identity in candidate.file_identities)
    reject_prompt_v3_phase_candidate(candidate)


def test_builder_uses_repository_native_sha256_null_oid(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path, object_format="sha256")
    assert len(parent) == 64
    candidate = _build(_request(repo, parent, target))
    assert len(candidate.commit_id) == 64
    assert _git(repo, "rev-parse", candidate.rescue_ref) == candidate.commit_id
    reject_prompt_v3_phase_candidate(candidate)


@pytest.mark.parametrize("boundary", ["before_tree", "after_commit"])
def test_construction_exception_settles_lease_and_rescue(
    tmp_path: Path, boundary: str
) -> None:
    repo, parent, target = _repo(tmp_path)
    request = _request(repo, parent, target)

    def fail(*_arguments) -> None:
        raise RuntimeError("injected construction failure")

    hooks = TransitionHooks(**{boundary: fail})
    with pytest.raises(RuntimeError, match="injected construction failure"):
        _build(request, hooks=hooks)
    candidate = _build(request)
    reject_prompt_v3_phase_candidate(candidate)


def test_phase_evidence_loader_recomputes_bytes_and_candidate_binding(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    candidate = _build(_request(repo, parent, target))
    wrong_raw = canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.phase-evidence-binding@1",
            "candidate_commit": parent,
            "authority_id": candidate.request.authority.authority_id,
            "kind": "suite",
        }
    )
    wrong_handle = EvidenceHandle(
        kind="suite",
        content_id=content_id(wrong_raw),
        byte_length=len(wrong_raw),
    )
    with pytest.raises(ProtectedAcceptanceDenied, match="exact candidate"):
        run_prompt_v3_phase_evidence(
            candidate,
            runner=lambda _candidate: (wrong_handle,),
            evidence_loader=lambda _candidate, _handle: wrong_raw,
        )
    assert not (repo / ".git" / "implementation-main-merge.lock").exists()


def test_build_validate_publish_cas_and_next_phase_from_fresh_exact_checkout(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    candidate, validated = _validated(_request(repo, parent, target))
    result = _publish(validated)
    assert result.published is True
    assert _git(repo, "rev-parse", target) == candidate.commit_id
    # The detached construction checkout remains exact and clean; the next
    # phase explicitly refreshes that isolated checkout to the published pin.
    assert _git(repo, "rev-parse", "HEAD") == parent
    assert _git(repo, "status", "--porcelain") == ""
    _git(repo, "checkout", "-q", "--detach", candidate.commit_id)
    next_request = _request(
        repo,
        candidate.commit_id,
        target,
        phase=PromptV3Phase.R,
        parent_phase=PromptV3Phase.Q,
    )
    next_candidate = _build(next_request)
    reject_prompt_v3_phase_candidate(next_candidate)


def test_checked_out_target_fast_forwards_cleanly_across_two_phases(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    held = tmp_path / "held-target"
    _git(repo, "worktree", "add", "-q", str(held), "protected/acceptance")
    for directory, child_directories, _files in os.walk(repo / ".git"):
        Path(directory).chmod(0o755)
        for child in child_directories:
            (Path(directory) / child).chmod(0o755)
    for directory, child_directories, _files in os.walk(held):
        Path(directory).chmod(0o755)
        for child in child_directories:
            path = Path(directory) / child
            if not path.is_symlink():
                path.chmod(0o755)

    first, first_validated = _validated(_request(repo, parent, target))
    _publish(first_validated)
    assert _git(held, "rev-parse", "HEAD") == first.commit_id
    assert _git(held, "status", "--porcelain") == ""

    _git(repo, "checkout", "-q", "--detach", first.commit_id)
    second_request = _request(
        repo,
        first.commit_id,
        target,
        phase=PromptV3Phase.R,
        parent_phase=PromptV3Phase.Q,
    )
    second, second_validated = _validated(second_request)
    _publish(second_validated)
    assert _git(held, "rev-parse", "HEAD") == second.commit_id
    assert _git(held, "status", "--porcelain") == ""


def test_builder_denies_dirty_ignored_attached_or_held_target(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty", encoding="ascii")
    with pytest.raises(ProtectedAcceptanceDenied):
        _build(_request(repo, parent, target))
    (repo / "dirty.txt").unlink()
    (repo / ".gitignore").write_text("shadow.py\n", encoding="ascii")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "ignore")
    parent = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", target, parent)
    (repo / "shadow.py").write_text("hostile", encoding="ascii")
    with pytest.raises(ProtectedAcceptanceDenied):
        _build(_request(repo, parent, target))
    (repo / "shadow.py").unlink()
    _git(repo, "checkout", "-q", "main")
    with pytest.raises(ProtectedAcceptanceDenied):
        _build(_request(repo, parent, target))


def test_builder_denies_real_index_only_drift(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    staged_only = repo / "staged-only.txt"
    staged_only.write_text("staged\n", encoding="ascii")
    _git(repo, "add", staged_only.name)
    staged_only.unlink()
    with pytest.raises(ProtectedAcceptanceDenied, match="real index"):
        _build(_request(repo, parent, target))


@pytest.mark.parametrize(
    "key,value",
    [
        ("GIT_INDEX_FILE", "/tmp/hostile-index"),
        ("GIT_DIR", "/tmp/hostile-repo"),
        ("GIT_EXTERNAL_DIFF", "/tmp/hostile-diff"),
        ("GIT_CONFIG_COUNT", "1"),
        ("GIT_REPLACE_REF_BASE", "refs/hostile"),
    ],
)
def test_builder_rejects_git_environment_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    repo, parent, target = _repo(tmp_path)
    monkeypatch.setenv(key, value)
    with pytest.raises(ProtectedAcceptanceDenied):
        _build(_request(repo, parent, target))


def test_builder_does_not_execute_hostile_path_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, parent, target = _repo(tmp_path)
    hostile = tmp_path / "bin"
    hostile.mkdir()
    fake = hostile / "git"
    fake.write_text("#!/bin/sh\nexit 99\n", encoding="ascii")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(hostile))
    candidate = _build(_request(repo, parent, target))
    reject_prompt_v3_phase_candidate(candidate)


def test_builder_denies_replace_graft_hook_config_and_submodule_ignore(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    _git(repo, "config", "core.fsmonitor", "true")
    with pytest.raises(ProtectedAcceptanceDenied):
        _build(_request(repo, parent, target))
    _git(repo, "config", "--unset", "core.fsmonitor")
    _git(repo, "config", "submodule.fake.ignore", "all")
    with pytest.raises(ProtectedAcceptanceDenied):
        _build(_request(repo, parent, target))


def test_builder_rejects_ref_and_lease_race_before_cas(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    _candidate, validated = _validated(_request(repo, parent, target))

    def race(_candidate):
        _git(repo, "update-ref", target, "f" * 40, parent)

    with pytest.raises((ProtectedTransitionGitError, subprocess.CalledProcessError)):
        _publish(validated, hooks=TransitionHooks(before_cas=race))


def test_pre_cas_denial_settles_rescue_and_lease(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    candidate, validated = _validated(_request(repo, parent, target))
    with pytest.raises(ProtectedAcceptanceDenied):
        _publish(validated, pre_cas_validator=lambda _value: False)
    assert _git(repo, "rev-parse", target) == parent
    assert not (repo / ".git" / "implementation-main-merge.lock").exists()
    assert (
        subprocess.run(
            ["/usr/bin/git", "show-ref", "--verify", "--quiet", candidate.rescue_ref],
            cwd=repo,
            check=False,
        ).returncode
        == 1
    )


def test_pre_cas_rechecks_authority_rotation_and_revocation(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    candidate, validated = _validated(_request(repo, parent, target))
    with pytest.raises(ProtectedAcceptanceDenied, match="rotated, or was revoked"):
        _publish(
            validated,
            authority_validator=lambda _authority, _now: False,
        )
    assert _git(repo, "rev-parse", target) == parent
    assert not (repo / ".git" / "implementation-main-merge.lock").exists()
    assert (
        subprocess.run(
            ["/usr/bin/git", "show-ref", "--verify", "--quiet", candidate.rescue_ref],
            cwd=repo,
            check=False,
        ).returncode
        == 1
    )


def test_post_publish_rejection_cannot_roll_back_terminal_publication(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    candidate, validated = _validated(_request(repo, parent, target))
    _publish(validated)
    assert not (repo / ".git" / "implementation-main-merge.lock").exists()
    with pytest.raises(ProtectedAcceptanceDenied, match="terminal"):
        reject_prompt_v3_phase_candidate(candidate)
    assert _git(repo, "rev-parse", target) == candidate.commit_id


def test_dry_run_never_updates_target_or_rescue_ref(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    candidate, validated = _validated(_request(repo, parent, target, dry_run=True))
    result = _publish(validated)
    assert not result.published and result.dry_run
    assert _git(repo, "rev-parse", target) == parent
    assert (
        subprocess.run(
            ["/usr/bin/git", "show-ref", "--verify", "--quiet", candidate.rescue_ref],
            cwd=repo,
            check=False,
        ).returncode
        == 1
    )


def _make_transition_lease_conclusively_dead(repo: Path) -> None:
    lock = repo / ".git" / "implementation-main-merge.lock"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["pid"] = 2**30
    payload["process_start_identity"] = "dead"
    lock.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    lock.chmod(0o600)


def test_dead_crash_lease_reuses_exact_candidate_rescue(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    request = _request(repo, parent, target)
    first = _build(request)
    assert _git(repo, "rev-parse", first.rescue_ref) == first.commit_id
    _make_transition_lease_conclusively_dead(repo)
    second = _build(request)
    assert second.commit_id == first.commit_id
    reject_prompt_v3_phase_candidate(second)


def test_stale_rejection_cannot_release_live_replacement_lease(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    request = _request(repo, parent, target)
    stale = _build(request)
    _make_transition_lease_conclusively_dead(repo)
    live = _build(request)
    lock = repo / ".git" / "implementation-main-merge.lock"
    live_lock_bytes = lock.read_bytes()
    with pytest.raises(ProtectedTransitionRace, match="foreign checkout lease"):
        reject_prompt_v3_phase_candidate(stale)
    assert lock.read_bytes() == live_lock_bytes
    reject_prompt_v3_phase_candidate(live)


def test_crash_after_target_cas_recovers_as_terminal_publication(
    tmp_path: Path,
) -> None:
    repo, parent, target = _repo(tmp_path)
    candidate, validated = _validated(_request(repo, parent, target))

    def crash(_candidate):
        raise SystemExit(91)

    with pytest.raises(SystemExit):
        _publish(validated, hooks=TransitionHooks(after_cas=crash))
    assert _git(repo, "rev-parse", target) == candidate.commit_id
    assert _git(repo, "rev-parse", candidate.rescue_ref) == candidate.commit_id
    _make_transition_lease_conclusively_dead(repo)
    recovered = _publish(validated)
    assert recovered.published and recovered.settlement_pending
    assert _git(repo, "rev-parse", target) == candidate.commit_id
    assert not (repo / ".git" / "implementation-main-merge.lock").exists()
    assert (
        subprocess.run(
            ["/usr/bin/git", "show-ref", "--verify", "--quiet", candidate.rescue_ref],
            cwd=repo,
            check=False,
        ).returncode
        == 1
    )


def test_symlink_repo_and_group_writable_source_are_denied(tmp_path: Path) -> None:
    repo, parent, target = _repo(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ProtectedTransitionGitError):
        _build(_request(linked, parent, target))
    repo.chmod(0o770)
    with pytest.raises(ProtectedTransitionGitError):
        _build(_request(repo, parent, target))


def test_cli_surface_is_bounded_and_has_no_run_all_or_raw_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert all(
        name in help_text
        for name in (
            "inspect",
            "readiness",
            "prepare-q",
            "advance-one-phase",
            "birth",
        )
    )
    assert "run-all" not in help_text and "raw-key" not in help_text
    assert transition_cli_main(["inspect"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_all"] is False
    assert "readiness" in payload["commands"]
    with pytest.raises(SystemExit):
        transition_cli_main(["birth", "--raw-key", "secret"])


def test_gitignore_anchors_root_core_without_ignoring_nested_module() -> None:
    repo = Path(__file__).resolve().parents[2]
    nested = (
        "ipfs_accelerate_py/agent_supervisor/core/protected_acceptance_contracts.py"
    )
    result = subprocess.run(
        ["/usr/bin/git", "check-ignore", "-q", nested], cwd=repo, check=False
    )
    assert result.returncode == 1
    assert (
        subprocess.run(
            ["/usr/bin/git", "check-ignore", "-q", "core"], cwd=repo, check=False
        ).returncode
        == 0
    )
    assert stat.S_IMODE((repo / ".gitignore").stat().st_mode) == 0o644


def test_q_construction_readiness_reports_tooling_and_blockers() -> None:
    from ipfs_accelerate_py.agent_supervisor.entrypoints.protected_acceptance_q_readiness import (
        assess_prompt_v3_q_construction_readiness,
    )

    report = assess_prompt_v3_q_construction_readiness(REPO_ROOT)
    assert report["schema"].endswith("prompt-v3-q-readiness@1")
    assert report["ready_for_prepare_q"] is True
    assert report["blocker_count"] == 0
    assert report["blockers"] == []
    assert report["q_inventory_present"] is False
    assert all(report["ase3_033_tooling"].values())
    products = report["pre_q_products"]
    assert set(products) == {
        "ASE3-019",
        "ASE3-023",
        "ASE3-027",
        "ASE3-030",
        "ASE3-031",
        "ASE3-032",
    }
    # ASE3-027 final blob freeze is sealed; generations and blobs verify.
    ase3027 = products["ASE3-027"]
    assert ase3027["sealed_ready_flag"] is True
    assert ase3027["generation_count"] == 2
    assert ase3027["generations"][0]["ok"] is True
    assert ase3027["generations"][1]["ok"] is True
    assert ase3027["final_blob_count"] == 5
    assert ase3027["blob_errors"] == []
    assert ase3027["product_generation_v1_ready"] is True
    # ASE3-023 three hermetic generations and final P-tree blobs are sealed.
    ase3023 = products["ASE3-023"]
    assert ase3023["sealed_ready_flag"] is True
    assert ase3023["generation_count"] == 3
    assert all(generation["ok"] is True for generation in ase3023["generations"])
    assert ase3023["final_blob_count"] == 7
    assert ase3023["blob_errors"] == []
    assert ase3023["product_generation_v1_ready"] is True
    # Product-generation@1 triples are sealed for all six pre-Q products.
    for task_id in (
        "ASE3-019",
        "ASE3-023",
        "ASE3-027",
        "ASE3-030",
        "ASE3-031",
        "ASE3-032",
    ):
        assert products[task_id]["product_generation_v1_ready"] is True, task_id
        assert products[task_id]["sealed_ready_flag"] is True, task_id
        assert products[task_id]["ready"] is True, task_id
    # ASE3-030 hermetic acceptance freeze is sealed.
    ase3030 = products["ASE3-030"]
    assert ase3030["generation_count"] == 2
    assert ase3030["final_blob_count"] == 7
    assert ase3030["blob_errors"] == []
    # ASE3-019 product-generation@1 triples are sealed (acceptance uses salvage ids).
    ase3019 = products["ASE3-019"]
    assert ase3019["generation_count"] == 0
    assert len(ase3019.get("product_generation_generations") or []) == 2
    # All six pre-Q products are sealed; prepare-q readiness is open.
    assert report["ready_for_prepare_q"] is True


def test_cli_readiness_command_is_available_without_injected_handlers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert transition_cli_main(["readiness", "--repo-root", str(REPO_ROOT)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_for_prepare_q"] is True
    assert payload["blocker_count"] == 0
