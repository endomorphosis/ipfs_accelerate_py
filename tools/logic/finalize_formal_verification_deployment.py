#!/usr/bin/env python3
"""Post-merge deployment attestation finalizer (FVT-G214 / FVT-067).

``RoleAwareFormalVerificationRelease@1`` after the FVT-G213 release-candidate
merge. Sole post-merge finalizer for the toolchain-release-finalizer lane.

Contract (fail-closed):

* Runs only after FVT-G213 has a successful, durable, canonical
  ``member_completion_receipt@1`` and a reachable merged commit.
* Verifies event-chain continuity, expected outputs, validation result, source
  tree, merged tree, datasets gitlink, origin publication, candidate digest,
  supported-capability closure, hard-zero gates, authority boundaries,
  quarantines, and public surfaces.
* Publishes either:
  - a receipt commit whose parent is the certified release commit with a
    strictly limited generated-artifact diff, or
  - an external content-addressed attestation (default; no circular tree
    identity).
* Mutating any event, tree, artifact, check, binding, or publication fact
  invalidates ``receipt_identity``.
* Absent or stale terminal evidence remains partial and is never called
  deployment-ready.
* Reads live state without mutation; never attests the current task's future
  event (FVT-067) as the G213 terminal merge.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

INTERFACE: Final = "RoleAwareFormalVerificationRelease@1"
PROGRAM_INTERFACE: Final = "FormalVerificationTacticianRelease@1"
SCHEMA_VERSION: Final = "formal-verification-role-aware-deployment-receipt/v1"
PROGRAM_GOAL_ID: Final = "FVT-G000"
GOAL_ID: Final = "FVT-G214"
TASK_ID: Final = "FVT-067"
PROGRAM: Final = "formal-verification-tactician/toolchain-release-finalizer"

# Upstream release-candidate identity (FVT-G213 / FVT-066).
RELEASE_CANDIDATE_INTERFACE: Final = "RoleAwareFormalVerificationReleaseCandidate@1"
RELEASE_CANDIDATE_GOAL_ID: Final = "FVT-G213"
RELEASE_CANDIDATE_TASK_ID: Final = "FVT-066"
RELEASE_CANDIDATE_SCHEMA_VERSION: Final = (
    "formal-verification-role-aware-release-candidate/v1"
)

SUPERVISOR_COMPLETION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.member_completion_receipt@1"
)

DEFAULT_DEPLOYMENT_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_deployment_receipt.json"
)
DEFAULT_COMPLETION_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_tactician_readiness_completion_receipt.json"
)
DEFAULT_RELEASE_CANDIDATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_release_candidate.json"
)
DEFAULT_TOOLCHAIN_CERT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_toolchain_certificate.json"
)
DEFAULT_FINALIZER_RELATIVE: Final = Path(
    "tools/logic/finalize_formal_verification_deployment.py"
)
DEFAULT_POST_MERGE_TEST_RELATIVE: Final = Path(
    "test/integration/test_formal_verification_role_aware_post_merge_attestation.py"
)
DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE: Final = Path(
    "test/integration/test_formal_verification_role_aware_release_candidate.py"
)
DEFAULT_BUILDER_RELATIVE: Final = Path(
    "tools/logic/build_formal_verification_tactician_receipt.py"
)
DEFAULT_CERTIFIER_RELATIVE: Final = Path(
    "tools/logic/certify_formal_verification_toolchains.py"
)

# Strictly limited generated-artifact set for receipt-commit publication.
GENERATED_ARTIFACT_PATHS: Final[frozenset[str]] = frozenset(
    {
        DEFAULT_DEPLOYMENT_RECEIPT_RELATIVE.as_posix(),
        DEFAULT_COMPLETION_RECEIPT_RELATIVE.as_posix(),
        DEFAULT_TOOLCHAIN_CERT_RELATIVE.as_posix(),
    }
)

PUBLICATION_MODE_EXTERNAL: Final = "external_content_addressed"
PUBLICATION_MODE_RECEIPT_COMMIT: Final = "receipt_commit"
RECEIPT_IDENTITY_SELF_REFERENCE: Final = "self:receipt_identity"

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
GIT_TIMEOUT_SECONDS: Final = 10.0

HARD_ZERO_GATE_KEYS: Final[tuple[str, ...]] = (
    "false_proof_count",
    "false_closure_count",
    "secret_or_witness_leakage_count",
    "authority_boundary_violations",
    "unresolved_cross_provider_disagreement_count",
)

REQUIRED_SEMANTIC_ELEVATIONS: Final[tuple[str, ...]] = (
    "lean",
    "runtime-mtl",
    "datalog-authorization",
    "secpal-authorization",
    "coq",
    "isabelle",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists() and (parent / "tools" / "logic").is_dir():
            return parent
    return Path.cwd().resolve()


def content_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    import hashlib

    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _is_sha256(value: Any, *, prefixed: bool | None = None) -> bool:
    text = str(value or "")
    has_prefix = text.startswith("sha256:")
    if prefixed is True and not has_prefix:
        return False
    if prefixed is False and has_prefix:
        return False
    digest = text.removeprefix("sha256:")
    return bool(SHA256_RE.fullmatch(digest))


def _git(
    repository: Path,
    *arguments: str,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str] | None:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_stdout(
    repository: Path,
    *arguments: str,
    allow_empty: bool = False,
) -> str | None:
    completed = _git(repository, *arguments)
    if completed is None or completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if allow_empty:
        return value
    return value or None


def _load_module(path: Path, name: str):
    for candidate in (path.resolve().parents[1], path.resolve().parents[1] / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_builder(repo_root: Path):
    return _load_module(
        repo_root / DEFAULT_BUILDER_RELATIVE,
        "fvt_post_merge_builder",
    )


def load_certifier(repo_root: Path):
    return _load_module(
        repo_root / DEFAULT_CERTIFIER_RELATIVE,
        "fvt_post_merge_certifier",
    )


def write_receipt(receipt: Mapping[str, Any], output: Path) -> None:
    """Atomic write so partial files never look like a durable attestation."""

    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)


# ---------------------------------------------------------------------------
# Release-candidate binding
# ---------------------------------------------------------------------------


def verify_release_candidate_digest_material(
    candidate: Mapping[str, Any] | None,
    *,
    certifier,
    role_aware_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the candidate's compact digest projection is self-consistent.

    A live candidate rebuild is useful diagnostics, but it includes host and
    working-tree observations that can legitimately drift after FVT-066. The
    durable gate is therefore the checked-in candidate identity plus digest
    material that can be independently reproduced from its compact certificate
    projection.
    """

    payload = dict(candidate) if isinstance(candidate, Mapping) else {}
    material = _safe_dict(payload.get("digest_material"))
    certificate = _safe_dict(payload.get("role_aware_certificate"))

    projected_tools = [
        dict(tool)
        for tool in _safe_list(certificate.get("tools"))
        if isinstance(tool, Mapping) and str(tool.get("tool_id") or "")
    ]
    tool_ids = [str(tool.get("tool_id")) for tool in projected_tools]
    unique_tool_ids = len(tool_ids) == len(set(tool_ids))
    expected_tool_checks = {
        str(tool.get("tool_id")): tool.get("checks_digest_sha256")
        for tool in projected_tools
    }
    expected_tool_artifacts = {
        str(tool.get("tool_id")): sorted(
            {
                str(digest)
                for digest in _safe_list(tool.get("artifact_digests"))
                if str(digest)
            }
        )
        for tool in projected_tools
    }

    projected_lanes = [
        dict(lane)
        for lane in _safe_list(certificate.get("semantic_lane_results"))
        if isinstance(lane, Mapping) and str(lane.get("lane_id") or "")
    ]
    lane_ids = [str(lane.get("lane_id")) for lane in projected_lanes]
    unique_lane_ids = len(lane_ids) == len(set(lane_ids))
    expected_semantic_receipts = {
        str(lane.get("lane_id")): str(lane.get("digest_sha256"))
        for lane in projected_lanes
        if lane.get("digest_sha256")
    }

    tool_checks = _safe_dict(material.get("tool_check_digests"))
    tool_artifacts = _safe_dict(material.get("tool_artifact_digests"))
    semantic_receipts = _safe_dict(material.get("semantic_receipt_digests"))
    specialized_binding = _safe_dict(
        certificate.get("specialized_receipt_aggregation")
    )
    specialized = _safe_dict(specialized_binding.get("projection"))
    specialized_verification = _safe_dict(
        specialized_binding.get("verification")
    )
    bound_certificate = _safe_dict(role_aware_certificate)
    live_specialized = _safe_dict(
        bound_certificate.get("specialized_receipt_aggregation")
    )
    certificate_authority = _safe_dict(certificate.get("authority_roles"))
    candidate_authority = _safe_dict(payload.get("roles"))
    quarantines = _safe_list(certificate.get("disagreement_quarantines"))

    required_keys = {
        "certificate_digest_sha256",
        "tool_check_digests",
        "tool_artifact_digests",
        "semantic_receipt_digests",
        "specialized_projection_aggregation_digest",
        "specialized_source_aggregation_digest",
        "specialized_projection_handler_digests",
        "specialized_source_handler_digests",
        "authority_roles_policy_digest",
        "lock_digest",
        "quarantine_digest",
    }
    expected_handler_keys = {
        (
            f"{str(spec.get('property_lane_id') or spec.get('lane_id') or '')}"
            f"::{str(tool_id)}"
        )
        for spec in certifier.SEMANTIC_CERTIFIER_SPECS
        for tool_id in _safe_list(spec.get("tool_ids"))
    }
    specialized_handlers = _safe_dict(
        specialized.get("specialized_by_handler")
    )
    handler_keys = set(str(key) for key in specialized_handlers)
    projection_handler_digests = {
        str(handler_key): _safe_dict(handler).get(
            "tool_evidence_digest_sha256"
        )
        for handler_key, handler in sorted(specialized_handlers.items())
    }
    source_handler_digests = {
        str(handler_key): _safe_dict(handler).get(
            "source_tool_evidence_digest_sha256"
        )
        for handler_key, handler in sorted(specialized_handlers.items())
    }
    handler_self_digests_valid = bool(
        handler_keys == expected_handler_keys
        and len(specialized_handlers) == len(expected_handler_keys)
        and all(
            _is_sha256(
                _safe_dict(handler).get(
                    "tool_evidence_digest_sha256"
                ),
                prefixed=False,
            )
            and _safe_dict(handler).get(
                "tool_evidence_digest_sha256"
            )
            == certifier.content_digest(
                {
                    key: value
                    for key, value in _safe_dict(handler).items()
                    if key != "tool_evidence_digest_sha256"
                }
            )
            and _is_sha256(
                _safe_dict(handler).get(
                    "source_tool_evidence_digest_sha256"
                ),
                prefixed=False,
            )
            for handler in specialized_handlers.values()
        )
    )
    expected_composites: dict[str, set[str]] = {}
    for spec in certifier.SEMANTIC_CERTIFIER_SPECS:
        property_lane_id = str(
            spec.get("property_lane_id")
            or spec.get("lane_id")
            or ""
        )
        expected_composites.setdefault(property_lane_id, set()).update(
            f"{property_lane_id}::{str(tool_id)}"
            for tool_id in _safe_list(spec.get("tool_ids"))
        )
    composites = _safe_dict(specialized.get("composite_lanes"))
    composite_handler_occurrences: list[str] = []
    composite_rows_valid = bool(
        set(str(key) for key in composites)
        == set(expected_composites)
        and len(composites) == len(expected_composites) == 9
    )
    for lane_id, expected_keys in sorted(expected_composites.items()):
        row = _safe_dict(composites.get(lane_id))
        observed_keys = [
            str(item)
            for item in _safe_list(row.get("handler_keys"))
        ]
        composite_handler_occurrences.extend(observed_keys)
        if not (
            row.get("property_lane_id") == lane_id
            and set(observed_keys) == expected_keys
            and len(observed_keys) == len(set(observed_keys))
            and _is_sha256(
                row.get("digest_sha256"),
                prefixed=False,
            )
        ):
            composite_rows_valid = False
    composite_coverage_valid = bool(
        composite_rows_valid
        and len(composite_handler_occurrences)
        == len(expected_handler_keys)
        and len(composite_handler_occurrences)
        == len(set(composite_handler_occurrences))
        and set(composite_handler_occurrences)
        == expected_handler_keys
    )
    projection_digest_computed = certifier.content_digest(
        {
            key: value
            for key, value in specialized.items()
            if key != "aggregation_digest_sha256"
        }
    )
    bound_certificate_digest_valid = bool(
        bound_certificate.get("certificate_digest_sha256")
        and bound_certificate.get("certificate_digest_sha256")
        == certifier.content_digest(
            {
                key: value
                for key, value in bound_certificate.items()
                if key != "certificate_digest_sha256"
            }
        )
    )

    checks = {
        "required_keys_complete": required_keys <= set(material),
        "certificate_digest_well_formed": _is_sha256(
            material.get("certificate_digest_sha256"), prefixed=False
        ),
        "certificate_digest_matches_projection": (
            material.get("certificate_digest_sha256")
            == certificate.get("certificate_digest_sha256")
        ),
        "certificate_digest_matches_bound_certificate": bool(
            bound_certificate_digest_valid
            and material.get("certificate_digest_sha256")
            == bound_certificate.get("certificate_digest_sha256")
        ),
        "tool_ids_unique": bool(projected_tools) and unique_tool_ids,
        "tool_check_digests_well_formed": bool(tool_checks)
        and all(_is_sha256(value, prefixed=False) for value in tool_checks.values()),
        "tool_check_digests_match_projection": (
            tool_checks == expected_tool_checks
        ),
        "tool_artifact_digests_well_formed": bool(tool_artifacts)
        and all(
            _is_sha256(digest)
            for digests in tool_artifacts.values()
            for digest in _safe_list(digests)
        ),
        "tool_artifact_digests_match_projection": (
            tool_artifacts == expected_tool_artifacts
        ),
        "semantic_lane_ids_unique": bool(projected_lanes) and unique_lane_ids,
        "semantic_receipt_digests_well_formed": bool(semantic_receipts)
        and all(
            _is_sha256(value, prefixed=False)
            for value in semantic_receipts.values()
        ),
        "semantic_receipt_digests_match_projection": (
            semantic_receipts == expected_semantic_receipts
        ),
        "specialized_projection_aggregation_digest_well_formed": _is_sha256(
            material.get(
                "specialized_projection_aggregation_digest"
            ),
            prefixed=False,
        ),
        "specialized_projection_aggregation_digest_recomputed": (
            material.get(
                "specialized_projection_aggregation_digest"
            )
            == specialized.get("aggregation_digest_sha256")
            == projection_digest_computed
        ),
        "specialized_source_aggregation_digest_well_formed": _is_sha256(
            material.get("specialized_source_aggregation_digest"),
            prefixed=False,
        ),
        "specialized_source_aggregation_digest_matches_projection": (
            material.get("specialized_source_aggregation_digest")
            == specialized.get(
                "source_aggregation_digest_sha256"
            )
        ),
        "specialized_handler_population_exact": (
            handler_keys == expected_handler_keys
            and len(specialized_handlers)
            == len(expected_handler_keys)
        ),
        "specialized_handler_self_digests_recomputed": (
            handler_self_digests_valid
        ),
        "specialized_composite_coverage_exact": (
            composite_coverage_valid
        ),
        "specialized_projection_handler_digests_match": (
            _safe_dict(
                material.get(
                    "specialized_projection_handler_digests"
                )
            )
            == projection_handler_digests
        ),
        "specialized_source_handler_digests_match": (
            _safe_dict(
                material.get(
                    "specialized_source_handler_digests"
                )
            )
            == source_handler_digests
        ),
        "specialized_projection_matches_live_certificate": bool(
            live_specialized and specialized == live_specialized
        ),
        "specialized_source_binding_matches_live_certificate": bool(
            live_specialized
            and material.get("specialized_source_aggregation_digest")
            == live_specialized.get(
                "source_aggregation_digest_sha256"
            )
            and source_handler_digests
            == {
                str(handler_key): _safe_dict(handler).get(
                    "source_tool_evidence_digest_sha256"
                )
                for handler_key, handler in sorted(
                    _safe_dict(
                        live_specialized.get(
                            "specialized_by_handler"
                        )
                    ).items()
                )
            }
        ),
        "specialized_fvt066_independent_audit_bound": bool(
            specialized_verification.get("projection_valid") is True
            and specialized_verification.get("source_valid") is True
            and specialized_verification.get(
                "source_matches_independent_reconstruction"
            )
            is True
            and specialized_verification.get(
                "projection_aggregation_digest_sha256"
            )
            == specialized.get("aggregation_digest_sha256")
            and specialized_verification.get(
                "source_aggregation_digest_sha256"
            )
            == specialized.get(
                "source_aggregation_digest_sha256"
            )
        ),
        "authority_roles_policy_digest_well_formed": _is_sha256(
            material.get("authority_roles_policy_digest"), prefixed=False
        ),
        "authority_roles_policy_digest_matches_projection": (
            material.get("authority_roles_policy_digest")
            == certificate_authority.get("policy_digest_sha256")
            == candidate_authority.get("policy_digest_sha256")
        ),
        # The compact certificate omits the full lock body, so the immutable
        # candidate identity can bind only a syntactically valid lock digest.
        "lock_digest_well_formed": _is_sha256(
            material.get("lock_digest"), prefixed=False
        ),
        "lock_digest_matches_live_certificate": (
            material.get("lock_digest")
            == _safe_dict(bound_certificate.get("lock")).get(
                "digest_sha256"
            )
        ),
        "quarantine_digest_well_formed": _is_sha256(
            material.get("quarantine_digest"), prefixed=False
        ),
        "quarantine_digest_matches_projection": (
            material.get("quarantine_digest")
            == certifier.content_digest(quarantines)
        ),
        "quarantine_digest_matches_live_certificate": (
            material.get("quarantine_digest")
            == certifier.content_digest(
                _safe_list(
                    bound_certificate.get(
                        "disagreement_quarantines"
                    )
                )
            )
        ),
    }
    failures = sorted(key for key, passed in checks.items() if not passed)
    return {
        "valid": not failures,
        "digest_material_identity": content_digest(material) if material else None,
        "checks": checks,
        "failures": failures,
        "live_recompute_required": False,
        "binding_rule": (
            "Checked candidate identity plus independently reproduced compact "
            "certificate digest material; host-dependent live recomputation is "
            "diagnostic only."
        ),
    }


def bind_release_candidate(
    *,
    repo_root: Path,
    builder,
    certifier,
    role_aware_certificate: Mapping[str, Any] | None = None,
    candidate_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the FVT-G213 release candidate by content identity."""

    path = repo_root / DEFAULT_RELEASE_CANDIDATE_RELATIVE
    checked = candidate_payload if candidate_payload is not None else load_json(path)
    live_certificate = (
        dict(role_aware_certificate)
        if role_aware_certificate is not None
        else certifier.build_certificate(repo_root=repo_root, role_aware=True)
    )
    live = builder.build_role_aware_release_candidate(
        repo_root=repo_root,
        role_aware_certificate=live_certificate,
    )
    checked_certificate = load_json(
        repo_root / DEFAULT_TOOLCHAIN_CERT_RELATIVE
    )
    live_identity = str(live.get("candidate_identity") or "")
    checked_identity = (
        str(checked.get("candidate_identity") or "") if isinstance(checked, Mapping) else ""
    )
    checked_body = None
    checked_identity_valid = False
    if isinstance(checked, Mapping) and checked_identity:
        body = {
            key: value
            for key, value in checked.items()
            if key != "candidate_identity"
        }
        checked_body = body
        checked_identity_valid = checked_identity == builder.content_digest(body)

    interface_ok = bool(
        isinstance(checked, Mapping)
        and checked.get("interface") == RELEASE_CANDIDATE_INTERFACE
        and checked.get("goal_id") == RELEASE_CANDIDATE_GOAL_ID
        and checked.get("task_id") == RELEASE_CANDIDATE_TASK_ID
        and checked.get("schema_version") == RELEASE_CANDIDATE_SCHEMA_VERSION
    )
    digest_material = verify_release_candidate_digest_material(
        checked,
        certifier=certifier,
        role_aware_certificate=checked_certificate,
    )
    # Live recompute may drift with host tools; the durable gate is that the
    # checked-in candidate is content-addressed, names G213 correctly, and has
    # independently reproducible compact digest material.
    bound = bool(
        path.is_file()
        and interface_ok
        and checked_identity_valid
        and checked_identity
        and digest_material.get("valid")
    )
    matches_live = bool(bound and checked_identity == live_identity)
    return {
        "path": DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix(),
        "present": path.is_file(),
        "interface": (checked or {}).get("interface") if isinstance(checked, Mapping) else None,
        "goal_id": (checked or {}).get("goal_id") if isinstance(checked, Mapping) else None,
        "task_id": (checked or {}).get("task_id") if isinstance(checked, Mapping) else None,
        "schema_version": (
            (checked or {}).get("schema_version") if isinstance(checked, Mapping) else None
        ),
        "checked_candidate_identity": checked_identity or None,
        "live_candidate_identity": live_identity or None,
        "checked_identity_valid": checked_identity_valid,
        "matches_live_recompute": matches_live,
        "digest_material_verification": digest_material,
        "bound": bound,
        "readiness_stage": (
            (checked or {}).get("readiness_stage") if isinstance(checked, Mapping) else None
        ),
        "status": (checked or {}).get("status") if isinstance(checked, Mapping) else None,
        "claims": _safe_dict((checked or {}).get("claims")) if isinstance(checked, Mapping) else {},
        "file_sha256": sha256_file(path),
        "live": {
            "candidate_identity": live_identity,
            "status": live.get("status"),
            "readiness_stage": live.get("readiness_stage"),
            "blockers": list(_safe_list(live.get("blockers")))[:50],
        },
        "block_reasons": [
            reason
            for reason, condition in (
                ("release_candidate_missing", path.is_file()),
                ("release_candidate_interface_mismatch", interface_ok),
                ("release_candidate_identity_invalid", checked_identity_valid),
                (
                    "release_candidate_digest_material_invalid",
                    bool(digest_material.get("valid")),
                ),
            )
            if not condition
        ],
        # Compact projection used in the attestation body (no bulk dump).
        "projection": {
            "interface": RELEASE_CANDIDATE_INTERFACE,
            "goal_id": RELEASE_CANDIDATE_GOAL_ID,
            "task_id": RELEASE_CANDIDATE_TASK_ID,
            "candidate_identity": checked_identity or live_identity or None,
            "status": (
                (checked or {}).get("status") if isinstance(checked, Mapping) else live.get("status")
            ),
            "readiness_stage": (
                (checked or {}).get("readiness_stage")
                if isinstance(checked, Mapping)
                else live.get("readiness_stage")
            ),
            "digest_material": (
                _safe_dict((checked or {}).get("digest_material"))
                if isinstance(checked, Mapping)
                else _safe_dict(live.get("digest_material"))
            ),
        },
        "_checked_body": checked_body,
        "_live": live,
        "_certificate": live_certificate,
    }


def bind_release_candidate_to_terminal_merge(
    *,
    repo_root: Path,
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the current candidate bytes to the verified terminal merge tree."""

    relative = DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix()
    merge_commit = str(_safe_dict(terminal.get("merge")).get("merge_commit") or "")
    terminal_bound = terminal.get("bound") is True
    current_blob = _git_stdout(repo_root, "hash-object", "--", relative)
    merged_blob = (
        _git_stdout(repo_root, "rev-parse", "--verify", f"{merge_commit}:{relative}")
        if terminal_bound and COMMIT_RE.fullmatch(merge_commit)
        else None
    )
    bound = bool(
        terminal_bound
        and current_blob
        and merged_blob
        and current_blob == merged_blob
    )
    failures: list[str] = []
    if terminal_bound:
        if not current_blob:
            failures.append("release_candidate_current_blob_missing")
        if not merged_blob:
            failures.append("release_candidate_terminal_merge_blob_missing")
        elif current_blob != merged_blob:
            failures.append("release_candidate_terminal_merge_blob_mismatch")
    return {
        "bound": bound,
        "terminal_evidence_bound": terminal_bound,
        "merge_commit": merge_commit or None,
        "current_blob": current_blob,
        "merged_blob": merged_blob,
        "failures": failures,
        "binding_rule": (
            "The candidate file bytes must equal the blob published by the "
            "verified FVT-066 terminal merge commit."
        ),
    }


# ---------------------------------------------------------------------------
# Terminal G213 supervisor evidence (not G212 release-evidence authority)
# ---------------------------------------------------------------------------


def _event_content_id(event: Mapping[str, Any]) -> str:
    body = {key: value for key, value in event.items() if key != "event_id"}
    return content_digest(body)


def _is_canonical_supervisor_task_cid(value: Any) -> bool:
    """Accept only canonical CIDv1/dag-json/sha2-256 task identities."""

    text = str(value or "")
    if (
        not text
        or text != text.lower()
        or not text.startswith("b")
        or not re.fullmatch(r"b[a-z2-7]+", text)
    ):
        return False
    encoded = text[1:]
    padding = "=" * ((8 - len(encoded) % 8) % 8)
    try:
        raw = base64.b32decode(
            (encoded + padding).upper(),
            casefold=False,
        )
    except (ValueError, binascii.Error):
        return False
    # CIDv1 (0x01), dag-json (0x0129 varint), sha2-256 (0x12), 32 bytes
    # (0x20), followed by the exact digest.
    return bool(
        len(raw) == 37
        and raw[:5] == b"\x01\xa9\x02\x12\x20"
    )


def _derive_commit_binding(
    *,
    repo_root: Path,
    implementation_commit: str,
    merge_commit: str,
    target_branch: str,
    integration_proof: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "implementation_commit": implementation_commit or None,
        "merge_commit": merge_commit or None,
        "implementation_tree": None,
        "merge_tree": None,
        "implementation_commit_exists": False,
        "merge_commit_exists": False,
        "implementation_is_ancestor": False,
        "source_trees_bound": False,
        "target_branch_bound": False,
        "published_to_origin_main": False,
        "failures": [],
    }
    failures: list[str] = result["failures"]
    if not COMMIT_RE.fullmatch(implementation_commit or ""):
        failures.append("implementation_commit_not_sha")
    if not COMMIT_RE.fullmatch(merge_commit or ""):
        failures.append("merge_commit_not_sha")
    if failures:
        return result

    resolved_impl = _git_stdout(
        repo_root, "rev-parse", "--verify", f"{implementation_commit}^{{commit}}"
    )
    resolved_merge = _git_stdout(
        repo_root, "rev-parse", "--verify", f"{merge_commit}^{{commit}}"
    )
    result["implementation_commit_exists"] = resolved_impl == implementation_commit
    result["merge_commit_exists"] = resolved_merge == merge_commit
    if not result["implementation_commit_exists"]:
        failures.append("implementation_commit_unreachable")
    if not result["merge_commit_exists"]:
        failures.append("merge_commit_unreachable")
    if failures:
        return result

    impl_tree = _git_stdout(repo_root, "rev-parse", f"{implementation_commit}^{{tree}}")
    merge_tree = _git_stdout(repo_root, "rev-parse", f"{merge_commit}^{{tree}}")
    result["implementation_tree"] = impl_tree
    result["merge_tree"] = merge_tree
    ancestor = _git(
        repo_root, "merge-base", "--is-ancestor", implementation_commit, merge_commit
    )
    result["implementation_is_ancestor"] = bool(
        ancestor is not None and ancestor.returncode == 0
    )
    if not result["implementation_is_ancestor"]:
        failures.append("implementation_not_ancestor_of_merge")

    result["source_trees_bound"] = bool(
        COMMIT_RE.fullmatch(str(impl_tree or ""))
        and COMMIT_RE.fullmatch(str(merge_tree or ""))
        and integration_proof.get("implementation_tree") == impl_tree
        and integration_proof.get("merge_tree") == merge_tree
    )
    if not result["source_trees_bound"]:
        failures.append("commit_source_trees_not_bound")

    result["target_branch_bound"] = target_branch in {
        "origin/main",
        "refs/remotes/origin/main",
        "main",
    }
    if not result["target_branch_bound"]:
        failures.append("merge_target_not_origin_main")

    published = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        merge_commit,
        "refs/remotes/origin/main",
    )
    result["published_to_origin_main"] = bool(
        published is not None and published.returncode == 0
    )
    if not result["published_to_origin_main"]:
        failures.append("merge_commit_not_published_to_origin_main")

    result["valid"] = not failures
    return result


def verify_g213_terminal_evidence(
    *,
    repo_root: Path,
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify durable G213/FVT-066 terminal member completion + merge.

    This path is intentionally separate from G212 trusted release-evidence
    authority. It only validates that the release-candidate task finished with a
    coherent completion receipt and a reachable published merge. Raw files are
    never mutated.
    """

    payload = dict(evidence) if isinstance(evidence, Mapping) else {}
    block_reasons: list[str] = []
    result: dict[str, Any] = {
        "present": bool(payload),
        "bound": False,
        "task_id": RELEASE_CANDIDATE_TASK_ID,
        "goal_id": RELEASE_CANDIDATE_GOAL_ID,
        "canonical_task_cid": None,
        "canonical_task_key": None,
        "event_chain": {},
        "expected_outputs": {},
        "validation": {},
        "merge": {},
        "member_completion_receipts": [],
        "commit_binding": {},
        "assumed_completion_references": [],
        "assumed_completion_rejected": False,
        "claims_current_task_future_event": False,
        "block_reasons": block_reasons,
        "snapshot_digest_sha256": None,
    }
    if not payload:
        block_reasons.append("g213_terminal_evidence_missing")
        return result

    # Reject attesting FVT-067's own unfinished future as the G213 terminal.
    declared_task = str(payload.get("task_id") or "")
    if declared_task == TASK_ID:
        result["claims_current_task_future_event"] = True
        block_reasons.append("cannot_attest_current_task_future_event")
        return result
    if declared_task and declared_task != RELEASE_CANDIDATE_TASK_ID:
        block_reasons.append("terminal_task_id_not_fvt_066")

    identity = _safe_dict(
        _safe_dict(payload.get("task_state")).get("canonical_identity")
        or payload.get("canonical_identity")
        or {}
    )
    expected_cid = str(
        identity.get("canonical_task_cid")
        or payload.get("canonical_task_cid")
        or ""
    )
    expected_key = str(
        identity.get("canonical_task_key")
        or payload.get("canonical_task_key")
        or ""
    )
    result["canonical_task_cid"] = expected_cid or None
    result["canonical_task_key"] = expected_key or None
    if not expected_cid or not expected_key:
        block_reasons.append("canonical_task_identity_missing")
    if expected_cid and not _is_canonical_supervisor_task_cid(
        expected_cid
    ):
        block_reasons.append("canonical_task_cid_not_strict_cidv1")

    task_state = _safe_dict(payload.get("task_state"))
    assumed_references = sorted(
        {
            str(reference)
            for source in (
                payload.get("assumed_completed_task_ids"),
                task_state.get("assumed_completed_task_ids"),
            )
            for reference in _safe_list(source)
            if str(reference)
        }
    )
    raw_assumed_counts = (
        payload.get("assumed_completed_count"),
        task_state.get("assumed_completed_count"),
    )
    parsed_assumed_counts = [
        value
        for value in raw_assumed_counts
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ]
    invalid_assumed_count = any(
        value is not None
        and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        )
        for value in raw_assumed_counts
    )
    assumed_count = max(
        *parsed_assumed_counts,
        len(assumed_references),
        int(invalid_assumed_count),
    )
    result["assumed_completion_references"] = assumed_references
    target_assumed_references = sorted(
        set(assumed_references)
        & {
            RELEASE_CANDIDATE_TASK_ID,
            RELEASE_CANDIDATE_GOAL_ID,
            expected_cid,
            expected_key,
        }
    )
    result["assumed_completion_count"] = assumed_count
    result["target_assumed_completion_references"] = (
        target_assumed_references
    )
    result["assumed_completion_rejected"] = bool(
        target_assumed_references or invalid_assumed_count
    )
    if target_assumed_references or invalid_assumed_count:
        block_reasons.append(
            "g213_target_assumed_completion_forbidden"
        )

    chain = _safe_dict(payload.get("event_chain"))
    events = [
        event
        for event in _safe_list(payload.get("events"))
        if isinstance(event, Mapping)
    ]
    chain_errors: list[str] = list(_safe_list(chain.get("errors")))
    previous_event_id = ""
    previous_sequence = 0
    for index, event in enumerate(events, start=1):
        sequence = event.get("sequence")
        event_id = str(event.get("event_id") or "")
        expected_id = _event_content_id(event)
        if event_id != expected_id:
            chain_errors.append(f"event_{index}:event_id_not_canonical")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            chain_errors.append(f"event_{index}:sequence_not_int")
        elif sequence != previous_sequence + 1:
            chain_errors.append(f"event_{index}:sequence_not_contiguous")
        if previous_sequence and str(event.get("previous_event_id") or "") != previous_event_id:
            chain_errors.append(f"event_{index}:previous_event_id_mismatch")
        if not previous_sequence and str(event.get("previous_event_id") or ""):
            chain_errors.append(f"event_{index}:first_previous_event_id_not_empty")
        if str(event.get("task_id") or "") != RELEASE_CANDIDATE_TASK_ID:
            chain_errors.append(f"event_{index}:task_id_mismatch")
        if expected_cid and str(event.get("canonical_task_cid") or "") != expected_cid:
            chain_errors.append(f"event_{index}:canonical_task_cid_mismatch")
        if expected_key and str(event.get("canonical_task_key") or "") != expected_key:
            chain_errors.append(f"event_{index}:canonical_task_key_mismatch")
        if (
            event.get("assumed_completed") is True
            or str(event.get("completion_basis") or "").lower()
            in {"assumed", "assumed_completed", "legacy_assumption"}
            or str(event.get("type") or "").lower()
            in {"assumed_completed", "implementation_assumed_complete"}
        ):
            chain_errors.append(
                f"event_{index}:assumed_completion_forbidden"
            )
        previous_sequence = sequence if isinstance(sequence, int) else previous_sequence
        previous_event_id = event_id

    chain_projection_matches = bool(
        chain
        and chain.get("valid") is True
        and chain.get("event_count") == len(events)
        and chain.get("last_sequence") == previous_sequence
        and chain.get("last_event_id") == (previous_event_id or None)
        and not _safe_list(chain.get("errors"))
    )
    if not chain_projection_matches:
        chain_errors.append("declared_event_chain_projection_mismatch")
    event_chain_valid = bool(events) and not chain_errors and (
        chain_projection_matches
    )
    result["event_chain"] = {
        "valid": event_chain_valid,
        "event_count": len(events),
        "last_sequence": previous_sequence,
        "last_event_id": previous_event_id or None,
        "errors": chain_errors,
    }
    if not event_chain_valid:
        block_reasons.append("event_chain_not_continuous")

    # Expected G213 outputs must exist on disk.
    expected_outputs = [
        DEFAULT_CERTIFIER_RELATIVE.as_posix(),
        DEFAULT_BUILDER_RELATIVE.as_posix(),
        DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE.as_posix(),
        DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix(),
    ]
    missing_outputs = [
        relative
        for relative in expected_outputs
        if not (repo_root / relative).is_file()
    ]
    result["expected_outputs"] = {
        "required": expected_outputs,
        "missing": missing_outputs,
        "bound": not missing_outputs,
    }
    if missing_outputs:
        block_reasons.append("g213_expected_outputs_missing")

    finished = [
        event for event in events if event.get("type") == "implementation_finished"
    ]
    terminal_events: list[Mapping[str, Any]] = []
    successful_receipts: list[dict[str, Any]] = []
    commit_bindings: list[dict[str, Any]] = []
    for event in finished:
        validation = _safe_dict(event.get("validation") or event.get("validation_result"))
        merge = _safe_dict(event.get("merge") or event.get("merge_result"))
        implementation_commit = str(
            event.get("implementation_commit")
            or merge.get("implementation_commit")
            or ""
        )
        merge_commit = str(merge.get("merge_commit") or "")
        integration_proof = _safe_dict(merge.get("integration_commit_proof"))
        commit_binding = _derive_commit_binding(
            repo_root=repo_root,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            target_branch=str(merge.get("target_branch") or ""),
            integration_proof=integration_proof,
        )
        receipt_candidates = [
            dict(receipt)
            for receipt in _safe_list(event.get("completion_receipts"))
            if isinstance(receipt, Mapping)
            and receipt.get("schema") == SUPERVISOR_COMPLETION_SCHEMA
            and str(receipt.get("task_id") or "") == RELEASE_CANDIDATE_TASK_ID
        ]
        assumed_receipts = [
            receipt
            for receipt in receipt_candidates
            if (
                receipt.get("assumed_completed") is True
                or receipt.get("legacy") is True
                or str(receipt.get("completion_basis") or "").lower()
                in {"assumed", "assumed_completed", "legacy_assumption"}
                or str(receipt.get("status") or "").lower()
                in {"assumed", "assumed_completed"}
            )
        ]
        if assumed_receipts:
            block_reasons.append(
                "g213_assumed_completion_receipt_forbidden"
            )
        receipts = [
            receipt
            for receipt in receipt_candidates
            if receipt.get("status") == "succeeded"
            and not assumed_receipts
            and _is_canonical_supervisor_task_cid(
                receipt.get("canonical_task_cid")
            )
            and (
                not expected_cid
                or str(receipt.get("canonical_task_cid") or "") == expected_cid
            )
            and (
                not expected_key
                or str(receipt.get("canonical_task_key") or "") == expected_key
            )
            and str(receipt.get("implementation_commit") or "")
            == implementation_commit
            and str(receipt.get("merge_commit") or "") == merge_commit
        ]
        coherent = bool(
            receipts
            and validation.get("attempted") is True
            and validation.get("passed") is True
            and validation.get("returncode") == 0
            and str(validation.get("target_commit") or "") == implementation_commit
            and merge.get("merged") is True
            and commit_binding["valid"] is True
        )
        if coherent:
            terminal_events.append(event)
            successful_receipts.extend(receipts)
            commit_bindings.append(commit_binding)

    if not successful_receipts:
        block_reasons.append("g213_member_completion_receipt_missing")
    if not terminal_events:
        block_reasons.append("g213_terminal_validation_or_merge_missing")

    terminal = terminal_events[-1] if terminal_events else {}
    terminal_merge = _safe_dict(
        terminal.get("merge") or terminal.get("merge_result")
    )
    terminal_validation = _safe_dict(
        terminal.get("validation") or terminal.get("validation_result")
    )
    result["validation"] = {
        "bound": bool(terminal_events),
        "attempted": terminal_validation.get("attempted"),
        "passed": terminal_validation.get("passed"),
        "returncode": terminal_validation.get("returncode"),
        "target_commit": terminal_validation.get("target_commit"),
    }
    result["merge"] = {
        "bound": bool(terminal_events),
        "merged": terminal_merge.get("merged"),
        "implementation_commit": terminal_merge.get("implementation_commit")
        or terminal.get("implementation_commit"),
        "merge_commit": terminal_merge.get("merge_commit"),
        "target_branch": terminal_merge.get("target_branch"),
    }
    result["member_completion_receipts"] = successful_receipts
    result["commit_binding"] = commit_bindings[-1] if commit_bindings else {}
    result["snapshot_digest_sha256"] = content_digest(payload) if payload else None
    result["bound"] = not block_reasons
    return result


# ---------------------------------------------------------------------------
# Publication modes
# ---------------------------------------------------------------------------


def verify_external_publication(
    *,
    receipt_identity: str | None,
    output_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    """External content-addressed attestation (no circular tree claim)."""

    identity_self_reference = (
        str(receipt_identity or "") == RECEIPT_IDENTITY_SELF_REFERENCE
    )
    identity_is_digest = _is_sha256(receipt_identity, prefixed=True)
    # The embedded publication uses a canonical self-reference so the outer
    # receipt can be sealed exactly once. A concrete digest remains accepted
    # for callers that independently verify an already-written receipt.
    identity_bound = identity_self_reference or identity_is_digest
    inspect_output = bool(identity_is_digest)
    present = bool(output_path and output_path.is_file()) if inspect_output else None
    file_identity = (
        sha256_file(output_path)
        if inspect_output and present and output_path
        else None
    )
    relative = None
    if output_path is not None:
        try:
            relative = output_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative = str(output_path)
    return {
        "mode": PUBLICATION_MODE_EXTERNAL,
        "bound": identity_bound,
        "circular_tree_identity_forbidden": True,
        "self_referential_current_tree_claim": False,
        "receipt_identity": receipt_identity,
        "receipt_identity_is_self_reference": identity_self_reference,
        "receipt_identity_resolution": (
            "top_level.receipt_identity"
            if identity_self_reference
            else "concrete_receipt_identity"
            if identity_is_digest
            else None
        ),
        "output_path": relative,
        "output_present": present,
        "output_file_sha256": file_identity,
        "output_observation": (
            "deferred_until_after_atomic_write"
            if identity_self_reference
            else "observed"
            if identity_is_digest
            else "unavailable"
        ),
        "publication_rule": (
            "External content-addressed attestation: the receipt identity is "
            "the digest of the attestation body excluding itself; the source "
            "tree never includes this receipt. The embedded publication binds "
            "that identity through the canonical self:receipt_identity "
            "reference so no nested circular digest is required."
        ),
        "block_reasons": []
        if identity_bound
        else ["external_receipt_identity_missing"],
    }


def verify_receipt_commit_publication(
    *,
    repo_root: Path,
    certified_source_commit: str | None,
    receipt_commit: str | None,
) -> dict[str, Any]:
    """Verify a receipt commit parented on the certified release commit.

    Diff must be restricted to GENERATED_ARTIFACT_PATHS. Never creates commits.
    """

    block_reasons: list[str] = []
    result: dict[str, Any] = {
        "mode": PUBLICATION_MODE_RECEIPT_COMMIT,
        "bound": False,
        "certified_source_commit": certified_source_commit,
        "receipt_commit": receipt_commit,
        "parent_is_certified_source": False,
        "diff_paths": [],
        "diff_limited_to_generated_artifacts": False,
        "allowed_paths": sorted(GENERATED_ARTIFACT_PATHS),
        "circular_tree_identity_forbidden": True,
        "block_reasons": block_reasons,
    }
    if not certified_source_commit or not COMMIT_RE.fullmatch(certified_source_commit):
        block_reasons.append("certified_source_commit_missing")
    if not receipt_commit or not COMMIT_RE.fullmatch(receipt_commit):
        block_reasons.append("receipt_commit_missing")
    if block_reasons:
        return result

    parent = _git_stdout(repo_root, "rev-parse", f"{receipt_commit}^")
    result["parent_is_certified_source"] = parent == certified_source_commit
    if not result["parent_is_certified_source"]:
        # Also accept certified source as ancestor (not only direct parent) when
        # the direct parent equals certified source is preferred; otherwise fail.
        block_reasons.append("receipt_commit_parent_not_certified_source")

    diff = _git_stdout(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        receipt_commit,
        allow_empty=True,
    )
    paths = sorted({line.strip() for line in (diff or "").splitlines() if line.strip()})
    result["diff_paths"] = paths
    result["diff_limited_to_generated_artifacts"] = bool(paths) and set(paths).issubset(
        GENERATED_ARTIFACT_PATHS
    )
    if not paths:
        block_reasons.append("receipt_commit_empty_diff")
    elif not result["diff_limited_to_generated_artifacts"]:
        block_reasons.append("receipt_commit_diff_outside_generated_artifacts")

    result["bound"] = not block_reasons
    result["block_reasons"] = block_reasons
    return result


# ---------------------------------------------------------------------------
# Capability / hard-zero / authority gates from live certificate + completion
# ---------------------------------------------------------------------------


def _compact_certificate_projection(
    certificate: Mapping[str, Any],
    *,
    certifier,
) -> dict[str, Any]:
    """Digest-bound certificate projection (no multi-MB receipt dumps)."""

    tools = {
        str(tool.get("tool_id") or ""): tool
        for tool in _safe_list(certificate.get("tools"))
        if isinstance(tool, Mapping) and str(tool.get("tool_id") or "")
    }
    tool_check_digests: dict[str, str] = {}
    for tool_id, tool in tools.items():
        tool_check_digests[tool_id] = certifier.content_digest(
            _safe_list(tool.get("checks"))
        )
    semantic = []
    for result in _safe_list(certificate.get("semantic_lane_results")):
        if not isinstance(result, Mapping):
            continue
        integrity = _safe_dict(result.get("receipt_integrity"))
        semantic.append(
            {
                "lane_id": result.get("lane_id"),
                "status": result.get("status"),
                "digest_sha256": result.get("digest_sha256"),
                "receipt_integrity_valid": integrity.get("valid"),
                "block_reasons": list(_safe_list(result.get("block_reasons"))),
            }
        )
    managed = _safe_dict(certificate.get("managed_deployment_readiness"))
    authority = _safe_dict(certificate.get("authority_roles"))
    promotion = _safe_dict(certificate.get("promotion"))
    role_aware = _safe_dict(certificate.get("role_aware"))
    return {
        "interface": certificate.get("interface"),
        "schema_version": certificate.get("schema_version"),
        "goal_id": certificate.get("goal_id"),
        "task_id": certificate.get("task_id"),
        "certificate_digest_sha256": certificate.get("certificate_digest_sha256"),
        "role_aware": {
            "enabled": role_aware.get("enabled"),
            "elevated_tool_ids": list(role_aware.get("elevated_tool_ids") or []),
        },
        "promotion": {
            "production_certified_tool_ids": list(
                promotion.get("production_certified_tool_ids") or []
            ),
            "merely_usable_tool_ids": list(
                promotion.get("merely_usable_tool_ids") or []
            ),
            "unavailable_tool_ids": list(promotion.get("unavailable_tool_ids") or []),
        },
        "authority_roles": {
            "present": authority.get("present"),
            "interface": authority.get("interface"),
            "policy_digest_sha256": authority.get("policy_digest_sha256"),
            "boundary": authority.get("boundary"),
        },
        "disagreement_quarantines": list(
            _safe_list(certificate.get("disagreement_quarantines"))
        ),
        "semantic_lane_results": semantic,
        "tool_check_digests": tool_check_digests,
        "managed_deployment_readiness": {
            "ready": managed.get("ready"),
            "status": managed.get("status"),
            "host_platform": managed.get("host_platform"),
            "platform_exceptions": [
                dict(item)
                for item in _safe_list(managed.get("platform_exceptions"))
                if isinstance(item, Mapping)
            ],
            "capability_blockers": list(
                _safe_list(managed.get("capability_blockers"))
            ),
            "dependency_blockers": list(
                _safe_list(managed.get("dependency_blockers"))
            ),
            "all_blockers": [
                {
                    "tool_id": item.get("tool_id"),
                    "reasons": list(_safe_list(item.get("reasons"))),
                }
                for item in _safe_list(managed.get("all_blockers"))
                if isinstance(item, Mapping)
            ],
        },
        "certification_policy": {
            key: _safe_dict(certificate.get("certification_policy")).get(key)
            for key in (
                "offline_policy_satisfied",
                "forbid_install",
                "forbid_download",
                "forbid_network",
            )
        },
        "public_evidence_policy": {
            "satisfied": _safe_dict(
                certificate.get("public_evidence_policy")
            ).get("satisfied")
        },
    }


def build_post_merge_attestation(
    *,
    repo_root: Path,
    observed_at: str | None = None,
    publication_mode: str = PUBLICATION_MODE_EXTERNAL,
    g213_terminal_evidence: Mapping[str, Any] | None = None,
    role_aware_certificate: Mapping[str, Any] | None = None,
    completion_receipt: Mapping[str, Any] | None = None,
    release_candidate: Mapping[str, Any] | None = None,
    receipt_commit: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build fail-closed post-merge RoleAwareFormalVerificationRelease@1."""

    repo_root = repo_root.resolve()
    timestamp = observed_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if publication_mode not in {
        PUBLICATION_MODE_EXTERNAL,
        PUBLICATION_MODE_RECEIPT_COMMIT,
    }:
        raise ValueError(f"unsupported publication_mode: {publication_mode}")

    builder = load_builder(repo_root)
    certifier = load_certifier(repo_root)

    certificate = (
        dict(role_aware_certificate)
        if role_aware_certificate is not None
        else certifier.build_certificate(repo_root=repo_root, role_aware=True)
    )
    completion = (
        dict(completion_receipt)
        if completion_receipt is not None
        else builder.build_receipt(repo_root=repo_root, observed_at=timestamp)
    )

    candidate_binding = bind_release_candidate(
        repo_root=repo_root,
        builder=builder,
        certifier=certifier,
        role_aware_certificate=certificate,
        candidate_payload=release_candidate,
    )
    # Drop non-JSON private keys from the binding for the receipt body.
    release_candidate_public = {
        key: value
        for key, value in candidate_binding.items()
        if not key.startswith("_")
    }

    terminal = verify_g213_terminal_evidence(
        repo_root=repo_root,
        evidence=g213_terminal_evidence,
    )
    candidate_merge_binding = bind_release_candidate_to_terminal_merge(
        repo_root=repo_root,
        terminal=terminal,
    )
    release_candidate_public["terminal_merge_blob_binding"] = (
        candidate_merge_binding
    )

    source = builder.build_source_attestation(repo_root)
    datasets_gitlink_bound = bool(
        source.get("datasets_gitlink")
        and source.get("datasets_gitlink") == source.get("datasets_embedded_head")
    )

    tools = {
        str(tool.get("tool_id") or ""): tool
        for tool in _safe_list(certificate.get("tools"))
        if isinstance(tool, Mapping) and str(tool.get("tool_id") or "")
    }
    managed = _safe_dict(certificate.get("managed_deployment_readiness"))
    authority_roles = _safe_dict(certificate.get("authority_roles"))
    promotion = _safe_dict(certificate.get("promotion"))
    role_aware = _safe_dict(certificate.get("role_aware"))
    public_policy = _safe_dict(certificate.get("public_evidence_policy"))
    certification_policy = _safe_dict(certificate.get("certification_policy"))

    certificate_digest_valid = bool(
        certificate.get("certificate_digest_sha256")
        and certificate.get("certificate_digest_sha256")
        == certifier.content_digest(
            {
                key: value
                for key, value in certificate.items()
                if key != "certificate_digest_sha256"
            }
        )
    )
    completion_identity_valid = bool(
        completion.get("receipt_identity")
        and completion.get("receipt_identity")
        == builder.content_digest(
            {
                key: value
                for key, value in completion.items()
                if key != "receipt_identity"
            }
        )
    )

    hard_zero = _safe_dict(completion.get("hard_zero_gates"))
    hard_zero_derivation = _safe_dict(hard_zero.get("derivation"))
    hard_zero_clear = all(
        int(hard_zero.get(key) or 0) == 0 for key in HARD_ZERO_GATE_KEYS
    )
    hard_zero_derived = bool(
        hard_zero_derivation.get("source")
        and hard_zero_derivation.get("hardcoded_success_counters_forbidden") is True
        and hard_zero_derivation.get("complete") is True
        and not _safe_list(hard_zero_derivation.get("missing_measurements"))
    )

    missing_required = [
        tid
        for tid in REQUIRED_SEMANTIC_ELEVATIONS
        if not tools.get(tid, {}).get("production_certified")
    ]
    elevated = sorted(set(role_aware.get("elevated_tool_ids") or []))

    non_authoritative = {
        "identity_plus_fixture_parser",
        "hermetic_adapter_shim",
        "hermetic_shadow_shim",
        "proposal_only_semantics",
    }
    synthetic_ok = all(
        not (
            tool.get("production_certified")
            and (
                tool.get("evidence_class") in non_authoritative
                or tool.get("executable_artifact_class")
                == "generated_hermetic_shim"
            )
        )
        for tool in tools.values()
    )
    role_tools = _safe_dict(authority_roles.get("tools"))
    authority_ceiling_respected = all(
        not (
            tools.get(tool_id, {}).get("production_certified")
            and not _safe_dict(meta).get("can_satisfy_certified_authority")
        )
        for tool_id, meta in role_tools.items()
    )
    quarantines_bound = certificate_digest_valid and isinstance(
        certificate.get("disagreement_quarantines"), list
    )
    public_surfaces_bound = bool(public_policy.get("satisfied"))
    offline_ok = bool(certification_policy.get("offline_policy_satisfied"))
    supported_ready = bool(managed.get("ready"))

    platform_exceptions = [
        dict(item)
        for item in _safe_list(managed.get("platform_exceptions"))
        if isinstance(item, Mapping)
    ]
    platform_exceptions_valid = all(
        item.get("narrow_scope") is True
        and item.get("complete") is False
        and item.get("production_certified") is False
        and item.get("classification") == "unsupported_here"
        for item in platform_exceptions
    ) if platform_exceptions else True

    artifacts = {
        "finalizer": {
            "path": DEFAULT_FINALIZER_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_FINALIZER_RELATIVE).is_file(),
            "content_identity": sha256_file(repo_root / DEFAULT_FINALIZER_RELATIVE),
        },
        "post_merge_test": {
            "path": DEFAULT_POST_MERGE_TEST_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_POST_MERGE_TEST_RELATIVE).is_file(),
            "content_identity": sha256_file(
                repo_root / DEFAULT_POST_MERGE_TEST_RELATIVE
            ),
        },
        "release_candidate": {
            "path": DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_RELEASE_CANDIDATE_RELATIVE).is_file(),
            "content_identity": candidate_binding.get("checked_candidate_identity"),
            "file_sha256": candidate_binding.get("file_sha256"),
        },
        "completion_receipt": {
            "path": DEFAULT_COMPLETION_RECEIPT_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_COMPLETION_RECEIPT_RELATIVE).is_file(),
            "content_identity": completion.get("receipt_identity"),
        },
        "toolchain_certificate": {
            "path": DEFAULT_TOOLCHAIN_CERT_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_TOOLCHAIN_CERT_RELATIVE).is_file(),
            "content_identity": sha256_file(
                repo_root / DEFAULT_TOOLCHAIN_CERT_RELATIVE
            ),
            "role_aware_digest": certificate.get("certificate_digest_sha256"),
        },
        "deployment_receipt": {
            "path": DEFAULT_DEPLOYMENT_RECEIPT_RELATIVE.as_posix(),
            "present_before_generation": (
                repo_root / DEFAULT_DEPLOYMENT_RECEIPT_RELATIVE
            ).is_file(),
            "publication_identity": "self:receipt_identity",
        },
    }
    artifacts_present = all(
        bool(item.get("present"))
        for key, item in artifacts.items()
        if key != "deployment_receipt"
    )

    # External mode binds a canonical self-reference before the one final seal.
    if publication_mode == PUBLICATION_MODE_RECEIPT_COMMIT:
        publication = verify_receipt_commit_publication(
            repo_root=repo_root,
            certified_source_commit=str(source.get("certified_source_commit") or ""),
            receipt_commit=receipt_commit,
        )
    else:
        publication = verify_external_publication(
            receipt_identity=RECEIPT_IDENTITY_SELF_REFERENCE,
            output_path=output_path,
            repo_root=repo_root,
        )

    acceptance = {
        "release_candidate_bound": bool(candidate_binding.get("bound")),
        "candidate_digest_bound": bool(candidate_binding.get("checked_identity_valid")),
        "candidate_digest_material_bound": bool(
            _safe_dict(
                candidate_binding.get("digest_material_verification")
            ).get("valid")
        ),
        "release_candidate_merge_blob_bound": bool(
            candidate_merge_binding.get("bound")
        ),
        "g213_terminal_receipt_bound": bool(terminal.get("bound")),
        "event_chain_continuous": bool(
            _safe_dict(terminal.get("event_chain")).get("valid")
        ),
        "g213_expected_outputs_bound": bool(
            _safe_dict(terminal.get("expected_outputs")).get("bound")
        ),
        "validation_result_bound": bool(
            _safe_dict(terminal.get("validation")).get("bound")
        ),
        "merged_commit_bound": bool(_safe_dict(terminal.get("merge")).get("bound")),
        "source_tree_bound": bool(source.get("source_commit_bound")),
        "datasets_gitlink_bound": datasets_gitlink_bound,
        "origin_publication_bound": bool(
            _safe_dict(terminal.get("commit_binding")).get("published_to_origin_main")
        )
        if terminal.get("bound")
        else False,
        "supported_capability_closure": supported_ready,
        "hard_zero_gates_clear": hard_zero_clear,
        "hard_zero_gates_derived": hard_zero_derived,
        "authority_ceiling_respected": authority_ceiling_respected,
        "quarantines_bound": quarantines_bound,
        "public_surfaces_bound": public_surfaces_bound,
        "role_aware_certificate_bound": certificate_digest_valid,
        "completion_receipt_bound": completion_identity_valid,
        "synthetic_evidence_cannot_certify_production": synthetic_ok,
        "no_install_during_offline_certification": offline_ok,
        "platform_exceptions_derived_and_narrow": platform_exceptions_valid,
        "required_elevations_complete": not missing_required,
        "artifacts_present": artifacts_present,
        "publication_bound": bool(publication.get("bound")),
        "never_claims_current_task_future_event": not bool(
            terminal.get("claims_current_task_future_event")
        ),
        "circular_tree_identity_forbidden": True,
    }

    readiness_requirements = {
        key: bool(acceptance[key])
        for key in (
            "release_candidate_bound",
            "candidate_digest_bound",
            "candidate_digest_material_bound",
            "release_candidate_merge_blob_bound",
            "g213_terminal_receipt_bound",
            "event_chain_continuous",
            "g213_expected_outputs_bound",
            "validation_result_bound",
            "merged_commit_bound",
            "source_tree_bound",
            "datasets_gitlink_bound",
            "origin_publication_bound",
            "supported_capability_closure",
            "hard_zero_gates_clear",
            "hard_zero_gates_derived",
            "authority_ceiling_respected",
            "quarantines_bound",
            "public_surfaces_bound",
            "role_aware_certificate_bound",
            "completion_receipt_bound",
            "synthetic_evidence_cannot_certify_production",
            "no_install_during_offline_certification",
            "platform_exceptions_derived_and_narrow",
            "artifacts_present",
            "publication_bound",
            "never_claims_current_task_future_event",
        )
    }

    deployment_blockers = sorted(
        [key for key, ok in readiness_requirements.items() if not ok]
        + list(terminal.get("block_reasons") or [])
        + list(candidate_binding.get("block_reasons") or [])
        + list(candidate_merge_binding.get("failures") or [])
        + list(publication.get("block_reasons") or [])
        + [
            f"managed:{item.get('tool_id')}:{reason}"
            for item in _safe_list(managed.get("all_blockers"))
            if isinstance(item, Mapping)
            for reason in _safe_list(item.get("reasons"))
        ]
    )

    all_ready = all(readiness_requirements.values()) and not deployment_blockers
    # Fail-closed vocabulary: never call deployment-ready without terminal gates.
    if all_ready:
        status = "role_aware_deployment_ready"
        readiness_stage = "deployment_ready"
    else:
        status = "role_aware_deployment_blocked"
        readiness_stage = "blocked"

    binding_mode = (
        "post_merge_receipt_commit_publication"
        if publication_mode == PUBLICATION_MODE_RECEIPT_COMMIT
        else "post_merge_external_content_addressed_attestation"
    )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "program_interface": PROGRAM_INTERFACE,
        "program_goal_id": PROGRAM_GOAL_ID,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "observed_at": timestamp,
        "binding_mode": binding_mode,
        "publication_mode": publication_mode,
        "status": status,
        "readiness_stage": readiness_stage,
        "description": (
            "Fail-closed post-merge deployment attestation (FVT-G214). Binds the "
            "FVT-G213 release candidate, its durable terminal supervisor "
            "completion receipt, and publishes either an external content-"
            "addressed attestation or a receipt commit with a strictly limited "
            "generated-artifact diff. Never attests the current task's future "
            "event and never claims deployment-ready without terminal evidence."
        ),
        "source": source,
        "release_candidate": release_candidate_public,
        "post_merge": {
            "depends_on_goal": RELEASE_CANDIDATE_GOAL_ID,
            "depends_on_task": RELEASE_CANDIDATE_TASK_ID,
            "terminal": {
                key: value
                for key, value in terminal.items()
                if key != "member_completion_receipts"
            }
            | {
                "member_completion_receipt_count": len(
                    terminal.get("member_completion_receipts") or []
                ),
                "member_completion_receipts": terminal.get(
                    "member_completion_receipts"
                )
                or [],
            },
            "publication": publication,
            "generated_artifact_paths": sorted(GENERATED_ARTIFACT_PATHS),
            "claims_own_future_event": False,
            "current_task_id": TASK_ID,
        },
        "acceptance": acceptance,
        "readiness_requirements": readiness_requirements,
        "deployment_blockers": deployment_blockers,
        "hard_zero_gates": {
            key: hard_zero.get(key, 0) for key in HARD_ZERO_GATE_KEYS
        }
        | {"derivation": hard_zero.get("derivation")},
        "role_aware_certificate": _compact_certificate_projection(
            certificate, certifier=certifier
        ),
        "completion": {
            "interface": completion.get("interface"),
            "completion_goal_id": completion.get("completion_goal_id"),
            "task_id": completion.get("task_id"),
            "receipt_identity": completion.get("receipt_identity"),
            "implementation_status": _safe_dict(completion.get("implementation")).get(
                "status"
            ),
            "deployment_status": _safe_dict(completion.get("deployment")).get("status"),
            "child_goals_bound": _safe_dict(completion.get("implementation")).get(
                "child_goals_bound"
            ),
            "child_goal_count": _safe_dict(completion.get("implementation")).get(
                "child_goal_count"
            ),
        },
        "elevations": {
            "required": list(REQUIRED_SEMANTIC_ELEVATIONS),
            "elevated_tool_ids": elevated,
            "missing_required": missing_required,
            "production_certified_tool_ids": list(
                promotion.get("production_certified_tool_ids") or []
            ),
            "merely_usable_tool_ids": list(
                promotion.get("merely_usable_tool_ids") or []
            ),
        },
        "platform_exceptions": platform_exceptions,
        "artifacts": artifacts,
        "claims": {
            "merge": bool(
                terminal.get("bound")
                and candidate_merge_binding.get("bound")
            ),
            "deployment": bool(all_ready),
            "post_merge_attestation": bool(all_ready),
            "self_referential_current_tree": False,
            "current_task_future_event": False,
            "max_stage": "deployment_ready" if all_ready else "blocked",
        },
        "disclosures": {
            "unavailable_tools": list(promotion.get("unavailable_tool_ids") or []),
            "merely_usable_tools": list(promotion.get("merely_usable_tool_ids") or []),
            "missing_required_elevations": missing_required,
            "supported_managed_capability_blockers": list(
                _safe_list(managed.get("capability_blockers"))
            ),
            "supported_managed_dependency_blockers": list(
                _safe_list(managed.get("dependency_blockers"))
            ),
            "assurance_ceilings": {
                "path_presence_is_not_usability": True,
                "source_presence_is_not_usability": True,
                "fixture_is_not_production_certified": True,
                "synthetic_evidence_cannot_certify_production": True,
                "advisor_support_shadow_cannot_certify": True,
                "unavailable_cannot_count_as_complete": True,
                "release_candidate_is_not_deployment_certificate": True,
                "absent_terminal_evidence_is_never_deployment_ready": True,
            },
            "remaining_bounds": [
                "Post-merge attestation requires durable FVT-066 terminal "
                "member completion and a reachable published merge commit.",
                "External content-addressed publication never hashes the tree "
                "containing this receipt.",
                "Receipt-commit publication must parent on the certified "
                "release commit and limit the diff to generated artifacts.",
                "FVT-067 never attests its own future merge event.",
            ],
        },
        "notes": [
            "RoleAwareFormalVerificationRelease@1 post-merge finalizer for "
            "FVT-G214 / FVT-067 after FVT-G213 release-candidate merge.",
            "Bulk formal certificates are bound by digest; rebuild from the "
            "live certifier rather than embedding full semantic dumps.",
            "Sole post-merge finalizer; read live state without mutation.",
        ],
    }

    # Public evidence projection + audit (host-private paths forbidden).
    receipt = certifier.public_evidence_projection(receipt, repo_root=repo_root)
    public_evidence_policy = certifier.public_evidence_audit(receipt)
    receipt["public_evidence_policy"] = public_evidence_policy
    if not public_evidence_policy.get("satisfied"):
        receipt["acceptance"]["public_surfaces_bound"] = False
        receipt["readiness_requirements"]["public_surfaces_bound"] = False
        receipt["status"] = "role_aware_deployment_blocked"
        receipt["readiness_stage"] = "blocked"
        receipt["claims"]["deployment"] = False
        receipt["claims"]["post_merge_attestation"] = False
        blockers = receipt["deployment_blockers"]
        if "public_surfaces_bound" not in blockers:
            blockers.append("public_surfaces_bound")
        leakage = len(public_evidence_policy.get("failures") or [])
        receipt["hard_zero_gates"]["secret_or_witness_leakage_count"] = max(
            int(receipt["hard_zero_gates"].get("secret_or_witness_leakage_count") or 0),
            leakage,
        )

    # Seal exactly once. External publication already carries the canonical
    # self-reference to this top-level identity.
    body_for_identity = {
        key: value for key, value in receipt.items() if key != "receipt_identity"
    }
    receipt["receipt_identity"] = content_digest(body_for_identity)

    return receipt


def load_verified_receipt(
    output: Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read an on-disk receipt and fail closed on round-trip or digest drift."""

    written = load_json(output)
    if written is None:
        raise RuntimeError(f"deployment receipt was not readable after write: {output}")
    if expected is not None and written != dict(expected):
        raise RuntimeError(
            "deployment receipt changed during JSON write/read round trip"
        )
    stored_identity = written.get("receipt_identity")
    body = {
        key: value for key, value in written.items() if key != "receipt_identity"
    }
    if not _is_sha256(stored_identity, prefixed=True):
        raise RuntimeError("deployment receipt identity is missing or malformed")
    if stored_identity != content_digest(body):
        raise RuntimeError("deployment receipt identity failed on-disk verification")
    return written


def finalize_deployment(
    *,
    repo_root: Path,
    output: Path | None = None,
    observed_at: str | None = None,
    publication_mode: str = PUBLICATION_MODE_EXTERNAL,
    g213_terminal_evidence: Mapping[str, Any] | None = None,
    receipt_commit: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build and optionally write the post-merge deployment attestation."""

    repo_root = repo_root.resolve()
    output_path = (
        output.resolve()
        if output is not None
        else (repo_root / DEFAULT_DEPLOYMENT_RECEIPT_RELATIVE)
    )
    receipt = build_post_merge_attestation(
        repo_root=repo_root,
        observed_at=observed_at,
        publication_mode=publication_mode,
        g213_terminal_evidence=g213_terminal_evidence,
        receipt_commit=receipt_commit,
        output_path=output_path if write else None,
    )
    if write:
        write_receipt(receipt, output_path)
        # Return the exact mapping that was persisted. Publication observation
        # is deliberately not appended after sealing.
        receipt = load_verified_receipt(output_path, expected=receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize post-merge RoleAwareFormalVerificationRelease@1 "
            f"attestation for {GOAL_ID} / {TASK_ID}."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Attestation output (default: {DEFAULT_DEPLOYMENT_RECEIPT_RELATIVE})",
    )
    parser.add_argument(
        "--publication-mode",
        choices=(PUBLICATION_MODE_EXTERNAL, PUBLICATION_MODE_RECEIPT_COMMIT),
        default=PUBLICATION_MODE_EXTERNAL,
        help="Publication mode (default: external content-addressed)",
    )
    parser.add_argument(
        "--receipt-commit",
        type=str,
        default=None,
        help="Git commit SHA for receipt-commit publication verification",
    )
    parser.add_argument(
        "--g213-terminal-evidence",
        type=Path,
        default=None,
        help=(
            "Optional JSON snapshot of FVT-066 terminal supervisor evidence "
            "(events + event_chain + task identity). Never mutates live state."
        ),
    )
    parser.add_argument(
        "--observed-at",
        type=str,
        default=None,
        help="Override observed_at timestamp (ISO-8601 UTC)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print attestation JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.repo_root or repo_root_from()).resolve()
    terminal = None
    if args.g213_terminal_evidence is not None:
        terminal = load_json(args.g213_terminal_evidence.resolve())
        if terminal is None:
            print(
                f"error: cannot load G213 terminal evidence: "
                f"{args.g213_terminal_evidence}",
                file=sys.stderr,
            )
            return 2

    if args.stdout:
        receipt = build_post_merge_attestation(
            repo_root=root,
            observed_at=args.observed_at,
            publication_mode=args.publication_mode,
            g213_terminal_evidence=terminal,
            receipt_commit=args.receipt_commit,
            output_path=None,
        )
        json.dump(receipt, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        output = (
            args.output.resolve()
            if args.output
            else (root / DEFAULT_DEPLOYMENT_RECEIPT_RELATIVE)
        )
        receipt = finalize_deployment(
            repo_root=root,
            output=output,
            observed_at=args.observed_at,
            publication_mode=args.publication_mode,
            g213_terminal_evidence=terminal,
            receipt_commit=args.receipt_commit,
            write=True,
        )
        if not args.quiet:
            print(f"wrote {output}", file=sys.stderr)

    if not args.quiet:
        print(
            f"status={receipt['status']} stage={receipt['readiness_stage']} "
            f"blockers={len(receipt['deployment_blockers'])}",
            file=sys.stderr,
        )
        print(f"receipt_identity={receipt['receipt_identity']}", file=sys.stderr)
        print(
            f"g213_terminal_bound="
            f"{receipt['acceptance']['g213_terminal_receipt_bound']}",
            file=sys.stderr,
        )
        print(
            f"publication_mode={receipt['publication_mode']} "
            f"publication_bound={receipt['acceptance']['publication_bound']}",
            file=sys.stderr,
        )

    # Exit 0 when the attestation was produced. Blocked/partial readiness is
    # recorded in the receipt, not as a crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
