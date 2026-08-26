#!/usr/bin/env python3
"""Issue the source-addressed EAAEF grok_cli/codex provider authorization.

Signed by a dedicated EAAEF local-operator profile and the host lifecycle
root. The prospective supervisor does not sign this artifact.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.control.eaaef_provider_authority import (
    EAAEF_PROVIDER_LEGACY_PROFILE_DIR,
    EAAEF_PROVIDER_LIFECYCLE_DIR,
    EAAEF_PROVIDER_PROFILE_ROOT,
    eaaef_provider_profile_candidates,
)
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    PROFILE_FILENAME,
    LocalProfileWrongRepository,
    ed25519_did_key,
    export_local_profile_lifecycle_witness,
    initialize_local_profile,
    lifecycle_root_identity_did,
    load_local_profile,
)

from ipfs_accelerate_py import agent_implementation_route as routes

# ``PROFILE_DIR`` remains the exact legacy location so already issued,
# independently signed authority stays usable.  New repository/baseline
# identities are isolated below ``PROFILE_ROOT``.
PROFILE_DIR = EAAEF_PROVIDER_LEGACY_PROFILE_DIR
PROFILE_ROOT = EAAEF_PROVIDER_PROFILE_ROOT
LIFECYCLE_DIR = EAAEF_PROVIDER_LIFECYCLE_DIR


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "git failed")
    return result.stdout.strip()


def _write_stable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.chmod(path, stat.S_IRUSR)


def _profile_directory_for_source(
    *,
    repository_cid: str,
    baseline_commit: str,
) -> Path:
    """Select an exact source profile without rotating another repository.

    The legacy directory is reused only when its fully verified profile binds
    this exact repository CID *and* baseline commit.  A repository change (or
    the rarer same-tree/different-HEAD case) receives a new directory instead
    of attempting ``force`` rotation across incompatible authority.
    """

    source_specific, *legacy_candidates = eaaef_provider_profile_candidates(
        repository_cid=repository_cid,
        baseline_commit=baseline_commit,
        profile_root=PROFILE_ROOT,
        legacy_profile_dir=PROFILE_DIR,
    )
    if source_specific.exists():
        return source_specific
    if not legacy_candidates:
        return source_specific
    legacy = legacy_candidates[0]
    if not (legacy / PROFILE_FILENAME).is_file():
        return source_specific
    try:
        profile = load_local_profile(
            repository_cid=repository_cid,
            profile_dir=legacy,
            lifecycle_dir=LIFECYCLE_DIR,
        )
    except LocalProfileWrongRepository:
        return source_specific
    return legacy if profile.baseline_commit == baseline_commit else source_specific


def issue() -> dict[str, str]:
    source_head = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    authorization_relative = routes.eaaef_agent_route_authorization_path(source_tree)
    root_pin_relative = routes.eaaef_agent_lifecycle_root_pin_path(source_tree)
    witness_relative = (
        routes._EAAEF_AGENT_LIFECYCLE_WITNESS_PREFIX + source_tree + "-host.json"
    )
    repository_cid = "sha256:" + hashlib.sha256(
        f"eaaef-v1:{source_tree}".encode()
    ).hexdigest()
    profile_dir = _profile_directory_for_source(
        repository_cid=repository_cid,
        baseline_commit=source_head,
    )
    event_time_ms = int(time.time() * 1000)
    effects = ("edit", "isolated_worktree", "test")
    profile = initialize_local_profile(
        repository_cid=repository_cid,
        baseline_commit=source_head,
        profile_dir=profile_dir,
        lifecycle_dir=LIFECYCLE_DIR,
        effect_bounds=effects,
        route_id=routes._EAAEF_AGENT_IMPLEMENTATION_ROUTE_ID,
        reviewer_provider="local_operator",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_reasoning_effort="high",
    )
    reviewer_identity = profile.identity_did
    root_identity = lifecycle_root_identity_did()
    root_pin = {
        "schema": routes._AGENT_LIFECYCLE_ROOT_PIN_SCHEMA,
        "board_namespace": routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        "base_head": source_head,
        "base_tree": source_tree,
        "root_identity_did": root_identity,
        "pinned_at_ms": event_time_ms,
    }
    root_pin["pin_id"] = routes._content_addressed_mapping(
        root_pin, identity_field="pin_id"
    )
    root_pin_bytes = _canonical(root_pin)
    _write_stable(ROOT / root_pin_relative, root_pin_bytes)
    root_pin_digest = "sha256:" + hashlib.sha256(root_pin_bytes).hexdigest()

    nonce = "eaaef:" + hashlib.sha256(source_tree.encode()).hexdigest()
    witness = export_local_profile_lifecycle_witness(
        repository_cid=repository_cid,
        board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
        base_head=source_head,
        base_tree=source_tree,
        nonce=nonce,
        profile_dir=profile_dir,
        lifecycle_dir=LIFECYCLE_DIR,
        observed_at_ms=event_time_ms,
        expires_at_ms=event_time_ms + 10 * 60 * 1000,
    )
    witness_bytes = _canonical(witness)
    _write_stable(ROOT / witness_relative, witness_bytes)
    witness_digest = "sha256:" + hashlib.sha256(witness_bytes).hexdigest()

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
        "effects": list(effects),
        "budget_cid": profile.budget_cid,
        "resource_cid": profile.resource_cid,
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
        lifecycle_root_pin_path=root_pin_relative,
        lifecycle_root_pin_sha256=root_pin_digest,
        authorized_at_ms=event_time_ms,
        fallback_implementer_identity="codex",
    )
    key_path = profile_dir / "local_dev_profile.key"
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    reviewer_key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    if ed25519_did_key(reviewer_key.public_key()) != reviewer_identity:
        raise RuntimeError("EAAEF reviewer key does not match profile identity")
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
        "lifecycle_root_pin_path": root_pin_relative,
        "lifecycle_root_pin_sha256": root_pin_digest,
        "authorized_at_ms": event_time_ms,
    }
    _write_stable(ROOT / authorization_relative, _canonical(authorization))
    return {
        "source_head": source_head,
        "source_tree": source_tree,
        "authorization_path": authorization_relative,
        "witness_path": witness_relative,
        "root_pin_path": root_pin_relative,
        "reviewer_did": reviewer_identity,
        "lifecycle_root_did": root_identity,
        "route_id": route["route_id"],
        "process_started": "false",
        "supervisor_signed": "false",
    }


def main() -> int:
    result = issue()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
