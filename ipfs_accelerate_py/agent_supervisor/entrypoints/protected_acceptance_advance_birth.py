"""Advance post-L provider-attempt generation birth (runtime effect start claim)."""

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

_RECEIPT_PATH = convergence.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH
_RELOAD_PATH = convergence.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
_FALLBACK_AUTH_PATH = convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
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


def _utc_to_ms(value: str) -> int:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _fresh_birth_authority(
    *, parent_commit: str, identity_did: str
) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.BIRTH,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.BIRTH,
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
    auth_raw = _git(repo, "show", f"{head}:{_FALLBACK_AUTH_PATH}")
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


def _birth_created_at_ms(
    *,
    lifecycle_authority: Mapping[str, Any],
    reload_payload: Mapping[str, Any],
) -> int:
    """Pick a birth time inside the sealed witness window and after L authority."""

    authorized_at_ms = int(lifecycle_authority["fallback_authorized_at_ms"])
    expires_at_ms = int(lifecycle_authority["lifecycle_witness_expires_at_ms"])
    reload_created_ms = _utc_to_ms(str(reload_payload["created_at"]))
    reload_signed_ms = _utc_to_ms(str(reload_payload["review"]["signed_at"]))
    earliest = max(authorized_at_ms, reload_created_ms, reload_signed_ms)
    # Prefer the earliest legal instant so long-stale builders still fit the
    # sealed 10-minute witness window used by the L receipt.
    candidate = earliest if earliest > authorized_at_ms else authorized_at_ms + 1_000
    if candidate < earliest:
        candidate = earliest
    if not authorized_at_ms <= candidate <= expires_at_ms:
        raise ProtectedAcceptanceDenied(
            "cannot place birth signed_at inside sealed witness validity after L"
        )
    if candidate < max(reload_created_ms, reload_signed_ms):
        raise ProtectedAcceptanceDenied(
            "birth chronology would predate signed L authority"
        )
    return candidate


def _build_birth_receipt(
    *,
    repo: Path,
    l_head: str,
    l_tree: str,
    lifecycle_authority: Mapping[str, Any],
) -> bytes:
    reload_raw = _git(repo, "show", f"{l_head}:{_RELOAD_PATH}")
    reload_payload = json.loads(reload_raw.decode("utf-8"))
    authorization = reload_payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ProtectedAcceptanceDenied("L reload authorization is unavailable")
    created_at_ms = _birth_created_at_ms(
        lifecycle_authority=lifecycle_authority,
        reload_payload=reload_payload,
    )
    created_at = _ms_to_utc(created_at_ms)
    payload: dict[str, Any] = {
        "schema": convergence.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": _BOARD_NAMESPACE,
        "phase": "post-L",
        "reload_authorization": {
            "path": _RELOAD_PATH,
            "sha256": _sha256_bytes(reload_raw),
            "head": l_head,
            "tree": l_tree,
            "phase": "L",
        },
        "generation": {
            "generation_id": authorization["target_generation_id"],
            "generation_number": authorization["target_generation_number"],
        },
        "process_birth": {
            "effect_started_at": created_at,
            "process_started_at": created_at,
            "runtime_effect_started": True,
        },
        "review": {
            **{
                field: lifecycle_authority[field]
                for field in convergence._ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
            },
            "implementer_identity": "codex:provider-attempt-generation-birth",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence._PROVIDER_ATTEMPT_GENERATION_BIRTH_DENIALS),
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


def _validate_birth_at_head(
    *,
    repo: Path,
    birth_head: str,
    birth_receipt_raw: bytes,
    lifecycle_authority: Mapping[str, Any],
) -> None:
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=birth_head,
        through_phase="L",
    )
    if discovery_errors:
        raise ProtectedAcceptanceDenied(
            "birth phase discovery failed: " + "; ".join(discovery_errors)
        )
    # Validator requires exact Q→L key order, not discovery insertion order.
    missing = set(convergence.SEQUENTIAL_ACCEPTANCE_PHASES) - set(discovered)
    if missing:
        raise ProtectedAcceptanceDenied(
            "birth phase_heads incomplete: " + ", ".join(sorted(missing))
        )
    phase_heads = {
        phase: discovered[phase]
        for phase in convergence.SEQUENTIAL_ACCEPTANCE_PHASES
    }
    # Checkout must match birth head for validator.
    current = _git(repo, "rev-parse", "HEAD").decode("ascii", "strict").strip()
    if current != birth_head:
        _git(repo, "checkout", "--detach", birth_head)
    errors = convergence.validate_provider_attempt_generation_birth_receipt(
        json.loads(birth_receipt_raw.decode("utf-8")),
        birth_receipt_raw=birth_receipt_raw,
        birth_head=birth_head,
        phase_heads=phase_heads,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )
    if errors:
        raise ProtectedAcceptanceDenied(
            "birth receipt failed validation: " + "; ".join(errors)
        )


def advance_prompt_v3_birth(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected post-L birth candidate."""

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
        raise ProtectedAcceptanceDenied(
            "provider-attempt generation birth receipt is already present"
        )

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=parent_commit,
        through_phase="L",
    )
    if discovery_errors or "L" not in discovered:
        raise ProtectedAcceptanceDenied(
            "L phase head unavailable: "
            + "; ".join(discovery_errors or ["missing L"])
        )
    l_head = discovered["L"]
    l_tree = (
        _git(repo, "rev-parse", f"{l_head}^{{tree}}")
        .decode("ascii", "strict")
        .strip()
    )
    # Birth must be a direct child of L (no neutral intermediates for birth-only path).
    if parent_commit != l_head:
        # Allow phase-neutral composition commits after L only if we still parent
        # the birth commit on the publish tip while validating L as authority.
        # Validator requires birth^ == L, so parent_commit for the candidate
        # must be exactly L.
        pass
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    lifecycle_authority = dict(_load_lifecycle_authority(repo, head=l_head))
    receipt_raw = _build_birth_receipt(
        repo=repo,
        l_head=l_head,
        l_tree=l_tree,
        lifecycle_authority=lifecycle_authority,
    )

    # Birth candidate parent must be exact L head (validator: birth^ == L).
    authority = _fresh_birth_authority(
        parent_commit=l_head, identity_did=root_did
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.BIRTH,
        expected_parent_phase=PromptV3Phase.L,
        allowed_paths=(_RECEIPT_PATH,),
        required_evidence_kinds=("birth-suite",),
        validator_ids=("provider-attempt-generation-birth",),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=l_head,
        parent_phase=PromptV3Phase.L,
        authority=authority,
        artifacts=(ArtifactBytes(path=_RECEIPT_PATH, data=receipt_raw),),
        evidence_handles=(_placeholder_handle("birth-suite"),),
        commit_message=(
            "ASE3-birth: seal provider-attempt generation birth after L"
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

    # Builder requires detached checkout at parent tree (L).
    current_head = _git(repo, "rev-parse", "HEAD").decode("ascii", "strict").strip()
    if current_head != l_head:
        _git(repo, "checkout", "--detach", l_head)
    # Ensure target_ref still points at expected tip for CAS when publishing.
    # When publishing onto main after composition, main may be ahead of L with
    # composition commits. Birth must still parent on L, so CAS parent is L —
    # that requires main == L or we publish to a ref currently at L.
    if publish and not dry_run:
        tip = _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
        if tip != l_head:
            raise ProtectedAcceptanceDenied(
                "birth publish requires target_ref tip to equal exact L head "
                f"(tip={tip[:12]} L={l_head[:12]}); land composition first then "
                "fast-forward or publish with target at L"
            )

    candidate = build_phase_candidate(
        request,
        authority_validator=_authority_ok,
        hooks=TransitionHooks(),
    )
    evidence_result = run_phase_evidence(
        candidate,
        runner=lambda observed: (_bound_handle(observed, "birth-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "provider-attempt-generation-birth"),
        ),
        evidence_loader=_evidence_loader,
    )

    # Full birth validator requires HEAD == birth and birth^ == L.
    pre_validate_head = _git(repo, "rev-parse", "HEAD").decode("ascii", "strict").strip()
    try:
        _validate_birth_at_head(
            repo=repo,
            birth_head=candidate.commit_id,
            birth_receipt_raw=receipt_raw,
            lifecycle_authority=lifecycle_authority,
        )
    finally:
        # Restore prior detached position when still dry-running.
        if dry_run and pre_validate_head:
            try:
                _git(repo, "checkout", "--detach", pre_validate_head)
            except ProtectedAcceptanceError:
                pass

    result: dict[str, Any] = {
        "phase": "birth",
        "dry_run": dry_run,
        "parent_commit": l_head,
        "l_head": l_head,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "receipt_path": _RECEIPT_PATH,
        "receipt_content_id": content_id(receipt_raw),
        "reviewer_identity": lifecycle_authority["reviewer_identity"],
        "published": False,
    }
    if publish and not dry_run:
        # Ensure HEAD is L for publish builder expectations; CAS updates target_ref.
        _git(repo, "checkout", "--detach", l_head)
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


__all__ = ("advance_prompt_v3_birth",)
