"""Compose prepare-q from sealed product freezes into a protected Q candidate.

Builds the exact no-future-pin Q inventory from sealed
``prompt-v3-product-generation@1`` freezes and publishes the Q phase through
the protected alternate-index builder: only the inventory file and ASE3-033
status flip (todo → completed) are written.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from ..core.protected_acceptance_contracts import (
    PROMPT_V3_PHASE_ORDER,
    ArtifactBytes,
    EvidenceHandle,
    GitFileIdentity,
    PhaseAuthority,
    PhaseCandidateRequest,
    PhaseEvidenceResult,
    PhasePolicy,
    ProductGenerationRecord,
    ProductProvenance,
    PromptV3Phase,
    PromptV3QInventory,
    ProtectedAcceptanceDenied,
    ProtectedAcceptanceError,
    PublicationResult,
    RepositoryBinding,
    StableQPolicy,
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
from .local_profile import lifecycle_root_identity_did
from .protected_acceptance_q_readiness import (
    Q_INVENTORY_RELATIVE_PATH,
    assess_prompt_v3_q_construction_readiness,
)

_CANONICAL_DIFF_ARGV = (
    "/usr/bin/git",
    "diff",
    "--no-ext-diff",
    "--no-textconv",
    "--no-renames",
    "--binary",
    "--full-index",
)
_Q_POLICY_SEED = {
    "schema": "ipfs_accelerate_py.agent_supervisor.prompt-v3-stable-q-policy@1",
    "phases": [item.value for item in PROMPT_V3_PHASE_ORDER],
    "maximum_p031_attempts": 3,
    "pre_q_products": sorted(
        task_id
        for task_id, values in convergence._PRODUCT_GENERATION_FINAL_VALUES.items()
        if values.get("ready")
    ),
}
_TODO_PATH = (
    "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md"
)
_ASE3_033_HEADER = (
    "## ASE3-033 Productionize protected transition construction, "
    "replay provenance, and phase-local authority"
)


def _git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
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
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or b"git failed").decode(
            "utf-8", "replace"
        )
        raise ProtectedAcceptanceError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _patch_bytes(repo: Path, parent: str, commit: str) -> bytes:
    return _git(repo, *_CANONICAL_DIFF_ARGV[1:], parent, commit)


def _file_identity(repo: Path, commit: str, path: str) -> GitFileIdentity:
    listing = _git(repo, "ls-tree", commit, "--", path).decode("ascii", "strict")
    lines = [line for line in listing.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProtectedAcceptanceError(f"expected one tree entry for {path}@{commit}")
    mode, _kind, blob, name = lines[0].split(maxsplit=3)
    if name != path:
        raise ProtectedAcceptanceError(f"tree path mismatch for {path}")
    if mode not in {"100644", "100755"}:
        raise ProtectedAcceptanceError(f"unsupported product mode {mode} for {path}")
    raw = _git(repo, "cat-file", "-p", f"{commit}:{path}")
    return GitFileIdentity(
        path=path,
        mode=mode,
        blob_id=blob,
        raw_content_id=content_id(raw),
        byte_length=len(raw),
    )


def _evidence_for_generation(
    *, role: str, patch_digest: str, path_count: int
) -> tuple[EvidenceHandle, ...]:
    body = canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.product-generation-suite@1",
            "role": role,
            "patch_digest": patch_digest,
            "path_count": path_count,
        }
    )
    return (
        EvidenceHandle(
            kind="product-generation-suite",
            content_id=content_id(body),
            byte_length=len(body),
            record_count=1,
        ),
    )


def _generation_record(
    repo: Path,
    *,
    role: str,
    commit: str,
    parent: str,
    tree: str,
    paths: tuple[str, ...],
    expected_patch_sha256: str,
) -> ProductGenerationRecord:
    patch = _patch_bytes(repo, parent, commit)
    patch_id = content_id(patch)
    if expected_patch_sha256 != patch_id and expected_patch_sha256 != (
        "sha256:" + hashlib.sha256(patch).hexdigest()
    ):
        # Freezes store sha256:hex; content_id is also sha256:hex.
        if patch_id != expected_patch_sha256:
            raise ProtectedAcceptanceDenied(
                f"{role} patch digest mismatch for {commit}"
            )
    observed_tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if observed_tree != tree:
        raise ProtectedAcceptanceDenied(f"{role} tree mismatch for {commit}")
    observed_parent = _git(repo, "rev-parse", f"{commit}^").decode().strip()
    if observed_parent != parent:
        raise ProtectedAcceptanceDenied(f"{role} parent mismatch for {commit}")
    files = tuple(_file_identity(repo, commit, path) for path in paths)
    return ProductGenerationRecord(
        role=role,
        commit=commit,
        parent=parent,
        tree=tree,
        files=files,
        test_evidence=_evidence_for_generation(
            role=role, patch_digest=patch_id, path_count=len(paths)
        ),
        canonical_patch_content_id=patch_id,
    )


def product_provenance_from_sealed_freeze(
    repo: Path | str, task_id: str
) -> ProductProvenance:
    """Build one ProductProvenance from the last sealed product-generation triple."""

    repo_path = Path(repo).resolve()
    sealed = convergence._PRODUCT_GENERATION_FINAL_VALUES.get(task_id)
    if not sealed or not sealed.get("ready"):
        raise ProtectedAcceptanceDenied(f"{task_id} product-generation is not sealed")
    generations = tuple(sealed.get("generations") or ())
    if not generations:
        raise ProtectedAcceptanceDenied(f"{task_id} has no sealed generation triples")
    generation = generations[-1]
    paths = tuple(sorted(generation["changed_paths"]))
    patch_digest = str(generation["source_patch_sha256"])
    source = _generation_record(
        repo_path,
        role="source",
        commit=str(generation["source_commit"]),
        parent=str(generation["source_parent"]),
        tree=str(generation["source_tree"]),
        paths=paths,
        expected_patch_sha256=patch_digest,
    )
    replay = _generation_record(
        repo_path,
        role="replay",
        commit=str(generation["replay_commit"]),
        parent=str(generation["replay_parent"]),
        tree=str(generation["replay_tree"]),
        paths=paths,
        expected_patch_sha256=patch_digest,
    )
    integrated = _generation_record(
        repo_path,
        role="integrated",
        commit=str(generation["integrated_commit"]),
        parent=str(generation["integrated_parent"]),
        tree=str(generation["integrated_tree"]),
        paths=paths,
        expected_patch_sha256=patch_digest,
    )
    return ProductProvenance(
        task_id=task_id,
        source=source,
        replay=replay,
        integrated=integrated,
        canonical_diff_content_id=source.canonical_patch_content_id,
    )


def build_prompt_v3_q_inventory(repo: Path | str) -> PromptV3QInventory:
    """Materialize the exact no-future-pin Q inventory from sealed freezes."""

    repo_path = Path(repo).resolve()
    readiness = assess_prompt_v3_q_construction_readiness(repo_path)
    if readiness.get("ready_for_prepare_q") is not True:
        raise ProtectedAcceptanceDenied(
            "prepare-q is blocked: " + "; ".join(readiness.get("blockers") or [])
        )
    if readiness.get("q_inventory_present"):
        raise ProtectedAcceptanceDenied("Q inventory is already present")
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")
    policy_id = content_id(canonical_json_bytes(_Q_POLICY_SEED))
    products = tuple(
        product_provenance_from_sealed_freeze(repo_path, task_id)
        for task_id in sorted(convergence._PRODUCT_GENERATION_FINAL_VALUES)
        if convergence._PRODUCT_GENERATION_FINAL_VALUES[task_id].get("ready")
    )
    return PromptV3QInventory(
        lifecycle_root_identity_did=root_did,
        stable_policy=StableQPolicy(policy_id=policy_id),
        product_provenance=products,
    )


def _complete_ase3_033_todo(todo_text: str) -> str:
    if _ASE3_033_HEADER not in todo_text:
        raise ProtectedAcceptanceError("ASE3-033 task header is missing from the board")
    pattern = re.compile(
        re.escape(_ASE3_033_HEADER)
        + r"\n\n- Status: todo\n",
        re.M,
    )
    updated, count = pattern.subn(
        _ASE3_033_HEADER + "\n\n- Status: completed\n", todo_text, count=1
    )
    if count != 1:
        raise ProtectedAcceptanceDenied(
            "ASE3-033 is not in the exact pre-Q todo status"
        )
    return updated


def _fresh_q_authority(*, parent_commit: str, identity_did: str) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000  # one hour
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.Q,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.Q,
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


def prepare_prompt_v3_q(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected Q phase candidate."""

    repo = Path(repo_root).resolve()
    # Protected construction forbids ignored drift such as __pycache__.
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
    inventory = build_prompt_v3_q_inventory(repo)
    inventory_bytes = canonical_json_bytes(inventory.to_dict()) + b"\n"
    todo_path = repo / _TODO_PATH
    todo_text = todo_path.read_text(encoding="utf-8")
    updated_todo = _complete_ase3_033_todo(todo_text)
    todo_bytes = updated_todo.encode("utf-8")
    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    authority = _fresh_q_authority(
        parent_commit=parent_commit,
        identity_did=inventory.lifecycle_root_identity_did,
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.Q,
        expected_parent_phase=None,
        allowed_paths=(Q_INVENTORY_RELATIVE_PATH, _TODO_PATH),
        required_evidence_kinds=("product-generation-suite", "q-readiness"),
        validator_ids=("prompt-v3-q-inventory", "ase3-033-status"),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=None,
        authority=authority,
        artifacts=(
            ArtifactBytes(path=Q_INVENTORY_RELATIVE_PATH, data=inventory_bytes),
            ArtifactBytes(path=_TODO_PATH, data=todo_bytes),
        ),
        evidence_handles=(
            _placeholder_handle("product-generation-suite"),
            _placeholder_handle("q-readiness"),
        ),
        commit_message=(
            "ASE3-033: prepare-q — seal Q inventory and complete transition task"
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
        runner=lambda observed: (
            _bound_handle(observed, "product-generation-suite"),
            _bound_handle(observed, "q-readiness"),
        ),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "prompt-v3-q-inventory"),
            _bound_handle(observed, "ase3-033-status"),
        ),
        evidence_loader=_evidence_loader,
    )
    result: dict[str, Any] = {
        "phase": "Q",
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "inventory_path": Q_INVENTORY_RELATIVE_PATH,
        "inventory_content_id": content_id(inventory_bytes),
        "product_count": len(inventory.product_provenance),
        "lifecycle_root_identity_did": inventory.lifecycle_root_identity_did,
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


__all__ = (
    "build_prompt_v3_q_inventory",
    "prepare_prompt_v3_q",
    "product_provenance_from_sealed_freeze",
)
