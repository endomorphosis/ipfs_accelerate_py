"""Fail-closed readiness probe for ASE3-033 Q construction.

Does not create Q inventory, change task status, sign artifacts, or start a
supervisor.  Reports exact blockers for the six pre-Q product freezes, the
lifecycle root DID, tooling presence, and inventory absence.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..core.protected_acceptance_contracts import (
    PRE_Q_PRODUCT_TASKS,
    Q_INVENTORY_SCHEMA,
)
from ..validation import prompt_v3_convergence as convergence

Q_INVENTORY_RELATIVE_PATH = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "protected_acceptance_q_inventory.json"
)
ASE3_033_TOOLING_PATHS = (
    "ipfs_accelerate_py/agent_supervisor/core/protected_acceptance_contracts.py",
    "ipfs_accelerate_py/agent_supervisor/merge/protected_acceptance_transition.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition_cli.py",
    "test/api/test_agent_supervisor_prompt_v3_transition.py",
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


def _git_ok(repo: Path, *arguments: str) -> tuple[bool, str]:
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
    ):
        env.pop(key, None)
    try:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "git failed").strip()
    return True, completed.stdout.strip()



def git_merge_base_is_ancestor(repo: Path, maybe_ancestor: str, tip: str) -> bool:
    ok, _ = _git_ok(repo, "merge-base", "--is-ancestor", maybe_ancestor, tip)
    return ok

def _object_exists(repo: Path, object_id: str) -> bool:
    ok, _ = _git_ok(repo, "cat-file", "-e", f"{object_id}^{{commit}}")
    if ok:
        return True
    ok, _ = _git_ok(repo, "cat-file", "-e", object_id)
    return ok


def _verify_generation(repo: Path, generation: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "source_commit",
        "source_parent",
        "source_tree",
        "integrated_commit",
        "integrated_parent",
        "integrated_tree",
        "binary_full_index_patch_sha256",
        "changed_paths",
    )
    missing = [key for key in required if key not in generation]
    if missing:
        return [f"generation missing fields: {','.join(missing)}"]
    for field in (
        "source_commit",
        "source_parent",
        "source_tree",
        "integrated_commit",
        "integrated_parent",
        "integrated_tree",
    ):
        value = str(generation[field])
        if value.startswith("FILL_AFTER_") or not _object_exists(repo, value):
            errors.append(f"{field} unavailable: {value}")
    if errors:
        return errors
    ok, observed_tree = _git_ok(
        repo, "rev-parse", f"{generation['source_commit']}^{{tree}}"
    )
    if not ok or observed_tree != generation["source_tree"]:
        errors.append("source_tree mismatch against git")
    ok, observed_parent = _git_ok(repo, "rev-parse", f"{generation['source_commit']}^")
    if not ok or observed_parent != generation["source_parent"]:
        errors.append("source_parent mismatch against git")
    ok, observed_tree = _git_ok(
        repo, "rev-parse", f"{generation['integrated_commit']}^{{tree}}"
    )
    if not ok or observed_tree != generation["integrated_tree"]:
        errors.append("integrated_tree mismatch against git")
    ok, observed_parent = _git_ok(
        repo, "rev-parse", f"{generation['integrated_commit']}^"
    )
    if not ok or observed_parent != generation["integrated_parent"]:
        errors.append("integrated_parent mismatch against git")
    env = dict(os.environ)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        env.pop(key, None)
    patch = subprocess.run(
        [
            *_CANONICAL_DIFF_ARGV,
            str(generation["source_parent"]),
            str(generation["source_commit"]),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        check=False,
    )
    # Note: for hermetic acceptance generations, the sealed patch is parent→source
    # for source role; integrated patch uses integrated_parent→integrated_commit.
    # Prefer integrated stream hash when binary_full_index is sealed for the product
    # role that binds integrated topology.
    patch_i = subprocess.run(
        [
            *_CANONICAL_DIFF_ARGV,
            str(generation["integrated_parent"]),
            str(generation["integrated_commit"]),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        check=False,
    )
    if patch_i.returncode != 0:
        errors.append("unable to reconstruct integrated full-index patch")
    else:
        digest = "sha256:" + hashlib.sha256(patch_i.stdout).hexdigest()
        if digest != generation["binary_full_index_patch_sha256"]:
            # Fall back to source-parent→source stream (some seals bind source side)
            if patch.returncode == 0:
                digest_s = "sha256:" + hashlib.sha256(patch.stdout).hexdigest()
                if digest_s != generation["binary_full_index_patch_sha256"]:
                    errors.append(
                        "binary_full_index_patch_sha256 mismatch against git "
                        f"(integrated={digest}, source={digest_s})"
                    )
            else:
                errors.append(
                    f"binary_full_index_patch_sha256 mismatch against git ({digest})"
                )
    return errors


def _product_status(repo: Path, task_id: str) -> dict[str, Any]:
    blockers: list[str] = []
    final_values = convergence._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES.get(task_id)
    if final_values is None:
        # Hermetic / native / duckdb tasks use other tables
        if task_id == "ASE3-030":
            final_values = convergence._HERMETIC_IDENTITY_FINAL_VALUES
        elif task_id == "ASE3-031":
            final_values = convergence._NATIVE_DEPENDENCY_ACCEPTANCE_FINAL_VALUES
        elif task_id == "ASE3-032":
            final_values = convergence._DUCKDB_POLICY_ACCEPTANCE_FINAL_VALUES
        else:
            return {
                "task_id": task_id,
                "ready": False,
                "blockers": [f"no final-value table for {task_id}"],
            }
    ready = bool(final_values.get("ready"))
    pending = final_values.get("pending")
    if not ready:
        blockers.append(f"final values not ready ({pending})")
    # ASE3-019 freezes salvage identities rather than generation triples.
    source_candidate = final_values.get("source_candidate") or {}
    salvage_base = final_values.get("salvage_base") or {}
    if task_id == "ASE3-019" and ready:
        for field in ("source_commit", "source_tree"):
            value = str(source_candidate.get(field) or "")
            if not value or value.startswith("FILL_AFTER_") or not _object_exists(
                repo, value
            ):
                blockers.append(f"source_candidate.{field} unavailable: {value}")
        salvage_head = str(salvage_base.get("head") or "")
        salvage_tree = str(salvage_base.get("tree") or "")
        if not salvage_head or not _object_exists(repo, salvage_head):
            blockers.append(f"salvage_base.head unavailable: {salvage_head}")
        else:
            ok, observed = _git_ok(repo, "rev-parse", f"{salvage_head}^{{tree}}")
            if not ok or observed != salvage_tree:
                blockers.append("salvage_base.tree mismatch against git")
            if not git_merge_base_is_ancestor(repo, salvage_head, "HEAD"):
                blockers.append("salvage_base.head is not an ancestor of HEAD")
    generations = final_values.get("generations") or ()
    generation_reports: list[dict[str, Any]] = []
    if not generations and task_id in {"ASE3-019", "ASE3-023", "ASE3-027"}:
        blockers.append("no sealed source/integrated generation records")
    for index, generation in enumerate(generations):
        if not isinstance(generation, Mapping):
            generation_reports.append(
                {"index": index, "ok": False, "errors": ["generation is not a mapping"]}
            )
            blockers.append(f"generation[{index}] is not a mapping")
            continue
        errors = _verify_generation(repo, generation)
        generation_reports.append(
            {
                "index": index,
                "ok": not errors,
                "role": generation.get("role"),
                "source_commit": generation.get("source_commit"),
                "integrated_commit": generation.get("integrated_commit"),
                "errors": errors,
            }
        )
        if errors:
            blockers.extend(f"generation[{index}]: {item}" for item in errors)
    final_blobs = final_values.get("final_blobs") or {}
    blob_errors: list[str] = []
    if final_blobs and generations:
        # Bind blobs against the last integrated commit when available.
        tip = None
        for generation in reversed(tuple(generations)):
            if isinstance(generation, Mapping) and generation.get("integrated_commit"):
                tip = str(generation["integrated_commit"])
                break
        if tip and _object_exists(repo, tip):
            for path, blob in final_blobs.items():
                ok, observed = _git_ok(repo, "rev-parse", f"{tip}:{path}")
                if not ok or observed != blob:
                    blob_errors.append(f"{path}: blob mismatch or missing at {tip[:12]}")
        else:
            blob_errors.append("cannot bind final_blobs without integrated tip")
    if blob_errors:
        blockers.extend(blob_errors)
    # Distinct replay commit still required for prompt-v3-product-generation@1
    product_generation_ready = False
    if ready and generations and not blob_errors:
        # Even with hermetic source/integrated pairs, Q inventory requires
        # independent source/replay/integrated triples — report as partial.
        blockers.append(
            "prompt-v3-product-generation@1 requires independent source, "
            "clean-replay, and integrated commits with identical inventories"
        )
    return {
        "task_id": task_id,
        "ready": ready and not blockers,
        "sealed_ready_flag": ready,
        "pending": pending,
        "generation_count": len(tuple(generations)),
        "generations": generation_reports,
        "final_blob_count": len(final_blobs),
        "blob_errors": blob_errors,
        "product_generation_v1_ready": product_generation_ready,
        "blockers": blockers,
    }


def assess_prompt_v3_q_construction_readiness(
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return a structured, fail-closed Q construction readiness report."""

    repo = Path(repo_root or Path.cwd()).resolve()
    tooling = {
        path: (repo / path).is_file() for path in ASE3_033_TOOLING_PATHS
    }
    inventory_path = repo / Q_INVENTORY_RELATIVE_PATH
    inventory_present = inventory_path.exists() or inventory_path.is_symlink()
    products = {
        task_id: _product_status(repo, task_id)
        for task_id in sorted(PRE_Q_PRODUCT_TASKS)
    }
    lifecycle_blockers: list[str] = []
    root_did = None
    try:
        from .local_profile import lifecycle_root_identity_did

        root_did = lifecycle_root_identity_did()
        if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
            lifecycle_blockers.append("lifecycle root DID is not a did:key")
            root_did = None
    except Exception as exc:  # local profile may be uninitialized
        lifecycle_blockers.append(f"lifecycle root unavailable: {type(exc).__name__}")

    product_blockers = [
        f"{task_id}: {blocker}"
        for task_id, report in products.items()
        for blocker in report["blockers"]
    ]
    tooling_blockers = [
        f"missing tooling: {path}" for path, present in tooling.items() if not present
    ]
    inventory_blockers = (
        ["Q inventory already present (expected only after prepare-q)"]
        if inventory_present
        else []
    )
    blockers = (
        tooling_blockers
        + inventory_blockers
        + lifecycle_blockers
        + product_blockers
    )
    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.prompt-v3-q-readiness@1",
        "q_inventory_schema": Q_INVENTORY_SCHEMA,
        "q_inventory_path": Q_INVENTORY_RELATIVE_PATH,
        "q_inventory_present": inventory_present,
        "ase3_033_tooling": tooling,
        "lifecycle_root_identity_did": root_did,
        "pre_q_products": products,
        "ready_for_prepare_q": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "notes": [
            "Does not create Q, receipts, signatures, or start a supervisor.",
            "prepare-q remains blocked until every pre-Q product has "
            "prompt-v3-product-generation@1 source/replay/integrated freezes.",
        ],
    }


__all__ = (
    "ASE3_033_TOOLING_PATHS",
    "Q_INVENTORY_RELATIVE_PATH",
    "assess_prompt_v3_q_construction_readiness",
)
