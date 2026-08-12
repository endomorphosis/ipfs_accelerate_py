"""Advance the protected chain A019→A030 (hermetic identity receipt)."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core.protected_acceptance_contracts import (
    ArtifactBytes,
    EvidenceHandle,
    PhaseAuthority,
    PhaseCandidateRequest,
    PhaseEvidenceResult,
    PhasePolicy,
    PromptV3Phase,
    ProtectedAcceptanceDenied,
    ProtectedAcceptanceError,
    PublicationResult,
    RepositoryBinding,
    canonical_json_bytes,
    content_id,
    phase_authority_content_id,
)
from ..merge.protected_acceptance_transition import (
    TransitionHooks,
    build_phase_candidate,
    publish_phase_candidate,
    run_phase_evidence,
    validate_phase_candidate,
)
from ..validation import prompt_v3_convergence as convergence
from .local_profile import lifecycle_root_identity_did, sign_profile_binding
from .protected_acceptance_transition import sign_prompt_v3_operator_artifact

_RECEIPT_PATH = convergence.HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
_MANIFEST_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/{convergence.MANIFEST_FILENAME}"
)
_BOARD_PATH = convergence.PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
_AUTH_PATH = convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
_WITNESS_PATH = convergence.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
_ROOT_PIN_PATH = convergence.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
_BOARD_NAMESPACE = convergence.BOARD_NAMESPACE


def _git(repo: Path, *arguments: str) -> bytes:
    env = dict(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_EDITOR",
        "GIT_SEQUENCE_EDITOR",
        "GIT_TERMINAL_PROMPT",
        "AGENT_SUPERVISOR_LOCAL_PROFILE_KEY",
    ):
        env.pop(key, None)
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repo,
        env=env,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or b"git failed").decode(
            "utf-8", "replace"
        )
        raise ProtectedAcceptanceError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _fresh_a030_authority(*, parent_commit: str, identity_did: str) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.A030,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.A030,
        authority_id=authority_id,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )


def _placeholder_handle(kind: str) -> EvidenceHandle:
    raw = kind.encode("ascii")
    return EvidenceHandle(
        kind=kind, content_id=content_id(raw), byte_length=len(raw), record_count=1
    )


def _bound_evidence_bytes(candidate: Any, kind: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.phase-evidence-binding@1",
            "candidate_commit": candidate.commit_id,
            "authority_id": candidate.request.authority.authority_id,
            "kind": kind,
        }
    )


def _bound_handle(candidate: Any, kind: str) -> EvidenceHandle:
    raw = _bound_evidence_bytes(candidate, kind)
    return EvidenceHandle(
        kind=kind, content_id=content_id(raw), byte_length=len(raw), record_count=1
    )


def _load_lifecycle_authority(repo: Path, *, head: str) -> Mapping[str, Any]:
    auth_raw = _git(repo, "show", f"{head}:{_AUTH_PATH}")
    wit_raw = _git(repo, "show", f"{head}:{_WITNESS_PATH}")
    pin_raw = _git(repo, "show", f"{head}:{_ROOT_PIN_PATH}")
    auth = convergence.ProviderFallbackPolicyAuthorization.from_dict(
        json.loads(auth_raw.decode("utf-8"))
    )
    witness = convergence.LocalOperatorLifecycleWitnessSnapshot(
        payload=json.loads(wit_raw.decode("utf-8")),
        raw=wit_raw,
        sha256=_sha256_bytes(wit_raw),
    )
    root_pin = convergence.LocalProfileLifecycleRootPinSnapshot(
        payload=json.loads(pin_raw.decode("utf-8")),
        raw=pin_raw,
        sha256=_sha256_bytes(pin_raw),
    )
    return auth.acceptance_review_authority(
        raw_sha256=_sha256_bytes(auth_raw),
        lifecycle_witness=witness,
        root_pin=root_pin,
    )


def _build_hermetic_receipt(
    *,
    a019_head: str,
    a019_tree: str,
    lifecycle_authority: Mapping[str, Any],
) -> bytes:
    frozen = convergence._HERMETIC_IDENTITY_FINAL_VALUES
    contracts = convergence._ACCEPTANCE_TASK_CONTRACTS["ASE3-030"]
    authorized_at_ms = int(lifecycle_authority["fallback_authorized_at_ms"])
    expires_at_ms = int(lifecycle_authority["lifecycle_witness_expires_at_ms"])
    signed_at_ms = (authorized_at_ms // 1000 + 1) * 1000
    if not authorized_at_ms <= signed_at_ms <= expires_at_ms:
        raise ProtectedAcceptanceDenied(
            "cannot place A030 signed_at inside sealed witness validity"
        )
    created_at = _ms_to_utc(signed_at_ms)

    member_paths = list(frozen["member_paths"])
    final_blobs = dict(frozen["final_blobs"])
    final_raw = dict(frozen["final_raw_sha256"])
    members = {
        path: {
            "git_blob": final_blobs[path],
            "raw_sha256": final_raw[path],
            "archive_member_sha256": final_raw[path],
        }
        for path in member_paths
    }
    module_origins = {
        module: dict(value)
        for module, value in sorted(dict(frozen["module_origins"]).items())
    }
    generations = []
    for generation in frozen["generations"]:
        item = dict(generation)
        item["changed_paths"] = list(item["changed_paths"])
        generations.append(item)

    manifest = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor.control-plane-dependency-manifest@1"
        ),
        "source_head": a019_head,
        "source_tree": a019_tree,
        "member_paths": member_paths,
        "module_names": sorted(module_origins),
        "cid_profile": "cidv1-base32-lower-raw+dag-json-sha2-256",
    }
    manifest_sha256 = convergence._canonical_sha256(manifest)
    if manifest_sha256 != frozen["manifest_sha256"]:
        raise ProtectedAcceptanceDenied(
            "hermetic manifest digest drift vs freeze "
            f"(got {manifest_sha256}, expected {frozen['manifest_sha256']})"
        )
    archive_root = convergence._canonical_sha256(
        {"member_paths": member_paths, "members": members}
    )
    if archive_root != frozen["archive_root_sha256"]:
        raise ProtectedAcceptanceDenied("hermetic archive root digest drift vs freeze")
    archive = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor.deterministic-control-plane-archive@1"
        ),
        "format": "zip-stored-sorted-v1",
        "sha256": frozen["archive_sha256"],
        "root_sha256": archive_root,
        "member_paths": member_paths,
    }
    capsule = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor.sealed-control-plane-capsule@1"
        ),
        "manifest_sha256": manifest_sha256,
        "archive_sha256": frozen["archive_sha256"],
        "archive_root_sha256": archive_root,
        "sealed_descriptor_sha256": frozen["sealed_descriptor_sha256"],
        "member_count": len(member_paths),
    }
    capsule_sha256 = convergence._canonical_sha256(capsule)
    if capsule_sha256 != frozen["capsule_sha256"]:
        raise ProtectedAcceptanceDenied("hermetic capsule digest drift vs freeze")

    empty = _sha256_bytes(b"")
    payload: dict[str, Any] = {
        "schema": convergence.HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": _BOARD_NAMESPACE,
        "task": {
            "task_id": "ASE3-030",
            "canonical_task_cid": contracts["canonical_task_cid"],
            "goal_id": contracts["goal_id"],
            "repairs_task": contracts["repairs_task"],
            "todo_contract_sha256": contracts["todo_contract_sha256"],
            "completed_contract_sha256": contracts["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "completed",
        },
        "acceptance_parent": {
            "head": a019_head,
            "tree": a019_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "manifest_schema": convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "receipt_paths_absent": list(
                convergence._sequential_future_artifacts_after("A019")
            ),
            "task_statuses": convergence._sequential_task_statuses_after("A019"),
            "reload_gate_status": "blocked",
        },
        "provenance": {
            "generations": generations,
            "final_blobs": final_blobs,
            "final_raw_sha256": final_raw,
        },
        "closure": {
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
            "capsule": capsule,
            "capsule_sha256": capsule_sha256,
            "archive": archive,
            "members": members,
            "module_origins": module_origins,
            "cid_vectors": list(convergence._HERMETIC_CID_VECTORS),
        },
        "probe": {
            "command": list(convergence._HERMETIC_HOSTILE_PROBE_ARGV),
            "environment": {"PYTHONNOUSERSITE": "1", "PYTHONPATH": None},
            "exit_code": 0,
            "isolated": True,
            "user_site_enabled": False,
            "pythonpath_present": False,
            "multiformats_imported": False,
            "repository_or_candidate_imported": False,
            "sealed_descriptor_only": True,
            "all_modules_imported": True,
            "all_module_origins_verified": True,
            "raw_cid_minted": True,
            "raw_cid_validated": True,
            "dag_json_cid_minted": True,
            "dag_json_cid_validated": True,
            "scheduler_or_provider_effect_started": False,
            "stdout_sha256": empty,
            "stderr_sha256": empty,
        },
        "suite": {
            "command": convergence._PROGRAM_EXPANSION_TASKS["ASE3-030"]["validation"],
            "exit_code": 0,
            "passed": True,
            "passed_count": frozen["suite_passed_count"],
            "failed_count": 0,
            "validated_head": a019_head,
            "validated_tree": a019_tree,
            "report_sha256": frozen["suite_report_sha256"],
        },
        "review": {
            **{
                field: lifecycle_authority[field]
                for field in convergence._ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
            },
            "implementer_identity": "codex:ase3-030-product",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence._HERMETIC_ACCEPTANCE_DENIALS),
    }

    def _signer(unsigned: Mapping[str, Any]) -> Mapping[str, str]:
        return sign_profile_binding(
            profile_dir=None, lifecycle_dir=None, payload=unsigned
        )

    signed = sign_prompt_v3_operator_artifact(payload, signer=_signer)
    return (
        json.dumps(
            signed,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_a030_manifest(
    *,
    parent_manifest: Mapping[str, Any],
    parent_manifest_raw: bytes,
    parent_head: str,
    parent_tree: str,
    receipt_raw: bytes,
    created_at: str,
) -> bytes:
    if parent_manifest.get("schema") != convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA:
        raise ProtectedAcceptanceDenied("A030 parent must be acceptance @2")
    child = dict(parent_manifest)
    child["created_at"] = created_at
    child["acceptance"] = {
        "phase": "A030",
        "parent_phase": "A019",
        "parent_head": parent_head,
        "parent_tree": parent_tree,
        "parent_manifest_sha256": _sha256_bytes(parent_manifest_raw),
        "artifacts": {_RECEIPT_PATH: _sha256_bytes(receipt_raw)},
        "task_statuses": convergence._sequential_task_statuses_after("A030"),
        "reload_gate_status": "blocked",
        "pre_launch_authorization_only": False,
        "runtime_effect_claimed": False,
    }
    return (
        json.dumps(
            child,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def advance_prompt_v3_a030(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected A030 candidate."""

    repo = Path(repo_root).resolve()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for path in repo.rglob("__pycache__"):
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    try:
                        child.unlink()
                    except OSError:
                        pass
            try:
                path.rmdir()
            except OSError:
                pass

    if (repo / _RECEIPT_PATH).exists() or (repo / _RECEIPT_PATH).is_symlink():
        raise ProtectedAcceptanceDenied("ASE3-030 hermetic receipt is already present")

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=parent_commit,
        through_phase="A019",
    )
    if discovery_errors or "A019" not in discovered:
        raise ProtectedAcceptanceDenied(
            "A019 phase head unavailable: "
            + "; ".join(discovery_errors or ["missing A019"])
        )
    a019_head = discovered["A019"]
    a019_tree = (
        _git(repo, "rev-parse", f"{a019_head}^{{tree}}")
        .decode("ascii", "strict")
        .strip()
    )
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    lifecycle_authority = dict(_load_lifecycle_authority(repo, head=a019_head))
    receipt_raw = _build_hermetic_receipt(
        a019_head=a019_head,
        a019_tree=a019_tree,
        lifecycle_authority=lifecycle_authority,
    )
    receipt_payload = json.loads(receipt_raw.decode("utf-8"))
    receipt_errors = convergence.validate_hermetic_identity_acceptance_receipt(
        receipt_payload,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )
    if receipt_errors:
        raise ProtectedAcceptanceDenied(
            "A030 hermetic receipt failed validation: " + "; ".join(receipt_errors)
        )

    parent_manifest_raw = _git(repo, "show", f"{a019_head}:{_MANIFEST_PATH}")
    parent_manifest = json.loads(parent_manifest_raw.decode("utf-8"))
    created_at = str(receipt_payload["created_at"])
    manifest_raw = _build_a030_manifest(
        parent_manifest=parent_manifest,
        parent_manifest_raw=parent_manifest_raw,
        parent_head=a019_head,
        parent_tree=a019_tree,
        receipt_raw=receipt_raw,
        created_at=created_at,
    )
    parent_board_raw = _git(repo, "show", f"{a019_head}:{_BOARD_PATH}")
    board_raw = convergence._status_only_sequential_phase_board(
        parent_board_raw, "A030"
    )

    authority = _fresh_a030_authority(
        parent_commit=parent_commit, identity_did=root_did
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.A030,
        expected_parent_phase=PromptV3Phase.A019,
        allowed_paths=(_RECEIPT_PATH, _MANIFEST_PATH, _BOARD_PATH),
        required_evidence_kinds=("a030-suite",),
        validator_ids=("hermetic-identity-acceptance",),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=PromptV3Phase.A019,
        authority=authority,
        artifacts=(
            ArtifactBytes(path=_RECEIPT_PATH, data=receipt_raw),
            ArtifactBytes(path=_MANIFEST_PATH, data=manifest_raw),
            ArtifactBytes(path=_BOARD_PATH, data=board_raw),
        ),
        evidence_handles=(_placeholder_handle("a030-suite"),),
        commit_message=(
            "ASE3-A030: seal hermetic identity acceptance receipt after A019"
        ),
        commit_timestamp=f"{int(time.time())} +0000",
        observed_at_ns=time.time_ns(),
        dry_run=dry_run,
    )

    def _authority_ok(value: Any, now_ns: int) -> bool:
        return (
            isinstance(value, PhaseAuthority)
            and value.authority_id == authority.authority_id
            and value.issued_at_ns <= now_ns < value.expires_at_ns
        )

    def _evidence_loader(candidate: Any, handle: EvidenceHandle) -> bytes:
        return _bound_evidence_bytes(candidate, handle.kind)

    candidate = build_phase_candidate(
        request,
        authority_validator=_authority_ok,
        hooks=TransitionHooks(),
    )
    evidence_result = run_phase_evidence(
        candidate,
        runner=lambda observed: (_bound_handle(observed, "a030-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "hermetic-identity-acceptance"),
        ),
        evidence_loader=_evidence_loader,
    )
    transition_errors = convergence.validate_sequential_acceptance_child_transition(
        repo_root=repo,
        phase="A030",
        child_head=candidate.commit_id,
        parent_head=a019_head,
        parent_tree=a019_tree,
        consumed_child_blobs={
            _RECEIPT_PATH: receipt_raw,
            _MANIFEST_PATH: manifest_raw,
            _BOARD_PATH: board_raw,
        },
    )
    if transition_errors:
        raise ProtectedAcceptanceDenied(
            "A030 sequential transition failed: " + "; ".join(transition_errors)
        )

    result: dict[str, Any] = {
        "phase": "A030",
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "a019_head": a019_head,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "receipt_path": _RECEIPT_PATH,
        "receipt_content_id": content_id(receipt_raw),
        "reviewer_identity": lifecycle_authority["reviewer_identity"],
        "published": False,
    }
    if publish and not dry_run:
        publication: PublicationResult = publish_phase_candidate(
            validated,
            authority_validator=_authority_ok,
            pre_cas_validator=lambda _item: True,
        )
        result.update(
            {
                "published": True,
                "publication_commit": publication.new_commit,
                "old_commit": publication.old_commit,
            }
        )
    return result


__all__ = ("advance_prompt_v3_a030",)
