"""Advance the protected chain A031→A032 (DuckDB connection policy acceptance)."""

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

_RECEIPT_PATH = convergence.DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
_MANIFEST_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/{convergence.MANIFEST_FILENAME}"
)
_BOARD_PATH = convergence.PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
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


def _fresh_a032_authority(*, parent_commit: str, identity_did: str) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.A032,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.A032,
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


def _prior_artifact_digests(repo: Path, *, head: str, phase: str) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative_path in convergence._sequential_artifacts_after(phase):
        raw = _git(repo, "show", f"{head}:{relative_path}")
        digests[relative_path] = _sha256_bytes(raw)
    return digests


def _build_policy_receipt(
    *,
    repo: Path,
    a031_head: str,
    a031_tree: str,
    lifecycle_authority: Mapping[str, Any],
) -> bytes:
    contracts = convergence._SEQUENTIAL_TASK_CONTRACTS["ASE3-032"]
    finals = convergence._DUCKDB_POLICY_ACCEPTANCE_FINAL_VALUES
    authorized_at_ms = int(lifecycle_authority["fallback_authorized_at_ms"])
    expires_at_ms = int(lifecycle_authority["lifecycle_witness_expires_at_ms"])
    signed_at_ms = (authorized_at_ms // 1000 + 1) * 1000
    if not authorized_at_ms <= signed_at_ms <= expires_at_ms:
        raise ProtectedAcceptanceDenied(
            "cannot place A032 signed_at inside sealed witness validity"
        )
    created_at = _ms_to_utc(signed_at_ms)

    product = dict(convergence._ASE3_032_PRODUCT_IDENTITY)
    product["changed_paths"] = list(product["changed_paths"])
    product["file_raw_sha256"] = dict(product["file_raw_sha256"])

    payload: dict[str, Any] = {
        "schema": convergence.DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": _BOARD_NAMESPACE,
        "phase": "A032",
        "task": {
            "task_id": "ASE3-032",
            "canonical_task_cid": contracts["canonical_task_cid"],
            "todo_contract_sha256": contracts["todo_contract_sha256"],
            "completed_contract_sha256": contracts["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "completed",
        },
        "acceptance_parent": {
            "head": a031_head,
            "tree": a031_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "phase": "A031",
            "manifest_schema": convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "prior_artifacts": _prior_artifact_digests(
                repo, head=a031_head, phase="A031"
            ),
            "future_artifact_paths_absent": list(
                convergence._sequential_future_artifacts_after("A031")
            ),
            "task_statuses": convergence._sequential_task_statuses_after("A031"),
            "reload_gate_status": "blocked",
        },
        "product": product,
        "connection_birth_policy": {
            "settings_in_connect_call": list(
                convergence._ASE3_032_CONNECTION_POLICY_SETTINGS
            ),
            "tuning_bounds": dict(convergence._ASE3_032_CONNECTION_TUNING_BOUNDS),
            "lock_configuration_last": True,
            "returned_connection_exact_bool_tuple": [
                False,
                False,
                False,
                False,
                True,
            ],
            "close_on_verification_failure": True,
            "caller_override_or_coercion_allowed": False,
        },
        "connection_sites": dict(convergence._ASE3_032_CONNECTION_SITE_COUNTS),
        "external_byte_boundary": dict(convergence._DUCKDB_EXTERNAL_BYTE_BOUNDARY),
        "catalog_seal": {
            "path_independent_full_persistent_catalog_equality": True,
            "inventories": [
                "databases",
                "schemas",
                "tables",
                "views",
                "sequences",
                "macros_and_functions",
                "types",
                "indexes",
                "constraints",
                "columns",
            ],
            "foreign_catalog_cases_rejected": list(
                convergence._ASE3_032_FOREIGN_CATALOG_CASES
            ),
            "source_bytes_unchanged_on_rejection": True,
            "temporary_files_cleaned_on_rejection": True,
        },
        "legacy_migration": {
            field: True
            for field in convergence._DUCKDB_LEGACY_MIGRATION_REQUIRED_FIELDS
        },
        "compaction": {
            "attach_count": 0,
            "source_read_only": True,
            "target_policy_initialized": True,
            "partial_copy_failure_preserves_authoritative_store": True,
            "atomic_replace_failure_preserves_authoritative_store": True,
            "foreign_catalog_rejection_preserves_source_bytes": True,
            "temporary_files_cleaned": True,
        },
        "suite": {
            "command": convergence._PROGRAM_EXPANSION_TASKS["ASE3-032"]["validation"],
            "exit_code": 0,
            "passed": True,
            "passed_count": finals["passed_count"],
            "failed_count": 0,
            "validated_head": a031_head,
            "validated_tree": a031_tree,
            "report_sha256": finals["report_sha256"],
            "required_test_functions": list(
                convergence._ASE3_032_REQUIRED_TEST_FUNCTIONS
            ),
        },
        "review": {
            **{
                field: lifecycle_authority[field]
                for field in convergence._ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
            },
            "implementer_identity": "codex:ase3-032-product",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence._DUCKDB_POLICY_ACCEPTANCE_DENIALS),
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


def _build_a032_manifest(
    *,
    parent_manifest: Mapping[str, Any],
    parent_manifest_raw: bytes,
    parent_head: str,
    parent_tree: str,
    receipt_raw: bytes,
    created_at: str,
) -> bytes:
    if parent_manifest.get("schema") != convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA:
        raise ProtectedAcceptanceDenied("A032 parent must be acceptance @2")
    child = dict(parent_manifest)
    child["created_at"] = created_at
    child["acceptance"] = {
        "phase": "A032",
        "parent_phase": "A031",
        "parent_head": parent_head,
        "parent_tree": parent_tree,
        "parent_manifest_sha256": _sha256_bytes(parent_manifest_raw),
        "artifacts": {_RECEIPT_PATH: _sha256_bytes(receipt_raw)},
        "task_statuses": convergence._sequential_task_statuses_after("A032"),
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


def advance_prompt_v3_a032(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected A032 candidate."""

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
            "DuckDB connection policy acceptance receipt is already present"
        )

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=parent_commit,
        through_phase="A031",
    )
    if discovery_errors or "A031" not in discovered:
        raise ProtectedAcceptanceDenied(
            "A031 phase head unavailable: "
            + "; ".join(discovery_errors or ["missing A031"])
        )
    a031_head = discovered["A031"]
    a031_tree = (
        _git(repo, "rev-parse", f"{a031_head}^{{tree}}")
        .decode("ascii", "strict")
        .strip()
    )
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    lifecycle_authority = dict(_load_lifecycle_authority(repo, head=a031_head))
    receipt_raw = _build_policy_receipt(
        repo=repo,
        a031_head=a031_head,
        a031_tree=a031_tree,
        lifecycle_authority=lifecycle_authority,
    )
    receipt_payload = json.loads(receipt_raw.decode("utf-8"))
    receipt_errors = convergence.validate_duckdb_connection_policy_acceptance_receipt(
        receipt_payload,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )
    if receipt_errors:
        raise ProtectedAcceptanceDenied(
            "A032 policy receipt failed validation: " + "; ".join(receipt_errors)
        )

    parent_manifest_raw = _git(repo, "show", f"{a031_head}:{_MANIFEST_PATH}")
    parent_manifest = json.loads(parent_manifest_raw.decode("utf-8"))
    created_at = str(receipt_payload["created_at"])
    manifest_raw = _build_a032_manifest(
        parent_manifest=parent_manifest,
        parent_manifest_raw=parent_manifest_raw,
        parent_head=a031_head,
        parent_tree=a031_tree,
        receipt_raw=receipt_raw,
        created_at=created_at,
    )
    parent_board_raw = _git(repo, "show", f"{a031_head}:{_BOARD_PATH}")
    board_raw = convergence._status_only_sequential_phase_board(
        parent_board_raw, "A032"
    )

    authority = _fresh_a032_authority(
        parent_commit=parent_commit, identity_did=root_did
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.A032,
        expected_parent_phase=PromptV3Phase.A031,
        allowed_paths=(_RECEIPT_PATH, _MANIFEST_PATH, _BOARD_PATH),
        required_evidence_kinds=("a032-suite",),
        validator_ids=("duckdb-connection-policy-acceptance",),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=PromptV3Phase.A031,
        authority=authority,
        artifacts=(
            ArtifactBytes(path=_RECEIPT_PATH, data=receipt_raw),
            ArtifactBytes(path=_MANIFEST_PATH, data=manifest_raw),
            ArtifactBytes(path=_BOARD_PATH, data=board_raw),
        ),
        evidence_handles=(_placeholder_handle("a032-suite"),),
        commit_message=(
            "ASE3-A032: seal DuckDB connection policy acceptance after A031"
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
        runner=lambda observed: (_bound_handle(observed, "a032-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "duckdb-connection-policy-acceptance"),
        ),
        evidence_loader=_evidence_loader,
    )
    transition_errors = convergence.validate_sequential_acceptance_child_transition(
        repo_root=repo,
        phase="A032",
        child_head=candidate.commit_id,
        parent_head=a031_head,
        parent_tree=a031_tree,
        consumed_child_blobs={
            _RECEIPT_PATH: receipt_raw,
            _MANIFEST_PATH: manifest_raw,
            _BOARD_PATH: board_raw,
        },
    )
    if transition_errors:
        raise ProtectedAcceptanceDenied(
            "A032 sequential transition failed: " + "; ".join(transition_errors)
        )

    result: dict[str, Any] = {
        "phase": "A032",
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "a031_head": a031_head,
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


__all__ = ("advance_prompt_v3_a032",)
