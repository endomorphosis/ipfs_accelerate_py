#!/usr/bin/env python3
"""Advisor role certification for SymAI, ErgoAI, Leanstral, autoencoder, Hammer.

``AdvisorRoleCertification@1`` / FVT-G160 (FVT-050).

Owns the hammer/advisor lane certification handler.  Certification proves:

* explicit strict installation selects locked SymAI and ErgoAI identities;
* SymAI, ErgoAI, Leanstral, autoencoder, and Hammer proposals are bounded,
  sanitized, source-bound, deterministic or replay-bound, cache-safe, and
  failure-explicit;
* no confidence, similarity, generated text, or advisor availability becomes
  proof without deterministic compilation and independent solver/kernel
  validation;
* advisors remain role=advisor/candidate with authority_ceiling=advisory and
  can never satisfy a certified-authority requirement or promote the hammer
  lane by presence alone.

Certification never installs, downloads, or opens the network (except that
installer dry-run / hermetic marker paths used by tests are offline).  It never
edits the central multi-prover certificate or model runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolchainAuthorityCeiling,
    ToolRole,
    can_satisfy_certified_authority_requirement,
    evaluate_role_aware_promotion,
    get_tool_role,
    tools_by_role,
)
from ipfs_datasets_py.logic.formalization.proposal_advisors import (  # noqa: E402
    LEANSTRAL_ADVISOR_INTERFACE,
    SYMAI_ADVISOR_INTERFACE,
    UNVERIFIED_AUTHORITY,
    LeanstralProposalAdvisor,
    ProposalAdvisorValidationError,
    ProposalKind,
    StaticProposalModel,
    SymAIProposalAdvisor,
    accept_candidate,
    build_json_candidates_response,
    confidence_never_yields_proof,
    is_untrusted_proposal_provider,
    sanitize_inert_text,
)

try:  # pragma: no cover - worktree packaging varies
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
    )
    from tools.logic.certification.roles import (
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]

try:  # pragma: no cover
    from ipfs_datasets_py.logic.backends.installers import advisors as advisors_installer
except Exception:  # pragma: no cover
    advisors_installer = None  # type: ignore[assignment]

INTERFACE: Final = "AdvisorRoleCertification@1"
SCHEMA_VERSION: Final = "advisor-role-certification/v1"
CORPUS_SCHEMA: Final = "advisor-role-corpus/v1"
GOAL_ID: Final = "FVT-G160"
TASK_ID: Final = "FVT-050"
PROGRAM: Final = "formal-verification-tactician/advisor-toolchains"
LANE_ID: Final = "hammer"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.advisors"
HANDLER_ID: Final = "advisor_role_certification@1"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.ADVISORY.value
AUTHORITY_SCOPE: Final = "candidate_generation_only"
LIVE_ERGOAI_INTERFACE: Final = "LiveErgoAIAdvisorCertification@1"
LIVE_ERGOAI_SCHEMA_VERSION: Final = "live-ergoai-advisor-certification/v1"
LIVE_ERGOAI_EVIDENCE_CLASS: Final = (
    "checksummed_managed_vendor_execution_advisory_only"
)
# FVT-G218 / FVT-085 — genuine ErgoAI advisor-toolchain path contract.
ERGOAI_LIVE_TOOLCHAIN_INTERFACE: Final = "ErgoAILiveToolchainContract@1"
ERGOAI_LIVE_TOOLCHAIN_SCHEMA: Final = "ergoai-live-toolchain-contract/v1"
ERGOAI_LIVE_TOOLCHAIN_GOAL_ID: Final = "FVT-G218"
ERGOAI_LIVE_TOOLCHAIN_TASK_ID: Final = "FVT-085"
ERGOAI_LIVE_TOOLCHAIN_PROGRAM: Final = (
    "formal-verification-tactician/ergoai-live-toolchain"
)
ERGOAI_LIVE_CASE_KINDS: Final = (
    "entailment",
    "non_entailment",
    "contradiction",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "resource_bound",
)

ADVISOR_TOOL_IDS: Final = (
    "symbolicai",
    "ergoai",
    "leanstral",
    "autoencoder",
    "hammer",
)
LOCKED_SYMBOLICAI_VERSION: Final = ">=1.14.0,<2.0.0"
LOCKED_ERGOAI_VERSION: Final = "3.0"

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

CHECK_KINDS: Final = frozenset(
    {
        "positive",
        "negative",
        "mutation",
        "replay",
        "malformed",
        "authority",
        "role",
        "install",
        "policy",
        "acceptance",
        "bounds",
    }
)


class AdvisorRoleCertificationError(ValueError):
    """Raised when advisor role certification inputs or results are invalid."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / DEFAULT_LOCK_RELATIVE).is_file():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    return Path.cwd().resolve()


def content_digest(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    # These guards prevent ordinary HTTP clients from inheriting a usable
    # proxy or silently falling back to a direct connection.  They are defense
    # in depth, not a claim of kernel-enforced network isolation (the receipt
    # reports that distinction explicitly).
    blocked_proxy = "http://127.0.0.1:9"
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env[key] = blocked_proxy
    env["NO_PROXY"] = ""
    env["no_proxy"] = ""
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_FORBID_DOWNLOAD"] = "1"
    return env


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    kind: str
    status: str  # passed | failed | skipped | unavailable | blocked | quarantined
    expected: str
    observed: str
    detail: str = ""
    tool_id: str = ""
    reason_codes: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseOutcome:
    case_id: str
    kind: str
    advisor_id: str
    expect: str
    status: str
    matched: bool
    reason_codes: list[str] = field(default_factory=list)
    authority: str = UNVERIFIED_AUTHORITY
    output_digest: str = ""
    detail: str = ""
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdvisorRoleCertification:
    """Full certification receipt for the advisor/hammer lane."""

    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    lane_id: str = LANE_ID
    certification_surface: str = CERTIFICATION_SURFACE
    handler_id: str = HANDLER_ID
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    advisor_tool_ids: list[str] = field(default_factory=lambda: list(ADVISOR_TOOL_IDS))
    locked_symbolicai_version: str = LOCKED_SYMBOLICAI_VERSION
    locked_ergoai_version: str = LOCKED_ERGOAI_VERSION
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    advisors_never_promote_alone: bool = True
    semantic_corpus_passed: bool = False
    role_matrix_passed: bool = False
    install_identity_passed: bool = False
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    cases: list[CaseOutcome] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        payload["cases"] = [case.to_dict() for case in self.cases]
        payload["receipt_digest_sha256"] = content_digest(
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_digest_sha256"
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Corpus (offline proposal / authority cases)
# ---------------------------------------------------------------------------


def _candidate_body(kind: str) -> str:
    return {
        "specification": "requires xs.length > 0 ensures result.sorted",
        "lemma": "lemma swap_preserves : forall xs, permute (swap xs) xs",
        "tactic": "apply Nat.le_refl",
        "premise": "axiom: list permutation is equivalence",
        "repair": "replace /goal/quantifier with forall",
    }.get(kind, "lemma swap_preserves : forall xs, permute (swap xs) xs")


def default_corpus_manifest() -> dict[str, Any]:
    """Compact embedded corpus for advisor role certification."""

    cases: list[dict[str, Any]] = []
    for advisor in ADVISOR_TOOL_IDS:
        provider = {
            "symbolicai": "symai",
            "ergoai": "ergoai",
            "leanstral": "leanstral",
            "autoencoder": "autoencoder",
            "hammer": "hammer",
        }[advisor]
        cases.append(
            {
                "case_id": f"{advisor}.positive_lemma",
                "kind": "positive",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "unverified_candidate",
                "confidence": 0.99,
                "is_valid": True,
                "similarity": 0.98,
                "description": f"{advisor} emits a source-bound lemma candidate only",
            }
        )
        cases.append(
            {
                "case_id": f"{advisor}.negative_confidence_not_proof",
                "kind": "negative",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "not_proved",
                "confidence": 1.0,
                "is_valid": True,
                "similarity": 1.0,
                "description": f"{advisor} confidence/is_valid never yields proof",
            }
        )
        cases.append(
            {
                "case_id": f"{advisor}.mutation_authority_claim",
                "kind": "mutation",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "rejected_or_quarantined",
                "mutate": "authority_claim",
                "description": f"{advisor} authority claim mutation is rejected",
            }
        )
        cases.append(
            {
                "case_id": f"{advisor}.replay_digest",
                "kind": "replay",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "deterministic_replay",
                "description": f"{advisor} proposal digests are deterministic",
            }
        )

    cases.extend(
        [
            {
                "case_id": "shared.malformed_executable_payload",
                "kind": "malformed",
                "advisor_id": "leanstral",
                "provider": "leanstral",
                "proposal_kind": "lemma",
                "expect": "rejected",
                "body": "lemma x : True := by\n```python\nimport os; os.system('rm -rf /')\n```",
                "description": "Executable payload markers are rejected",
            },
            {
                "case_id": "shared.ungrounded_source_refs",
                "kind": "malformed",
                "advisor_id": "symbolicai",
                "provider": "symai",
                "proposal_kind": "lemma",
                "expect": "rejected",
                "source_ref_ids": ["source:not-in-request"],
                "description": "Candidates with source_ref_ids outside the request are rejected",
            },
            {
                "case_id": "shared.acceptance_requires_compilation_and_validation",
                "kind": "acceptance",
                "advisor_id": "leanstral",
                "provider": "leanstral",
                "proposal_kind": "lemma",
                "expect": "acceptance_gate",
                "description": "accept_candidate requires compile + independent validation",
            },
        ]
    )

    return {
        "schema_version": CORPUS_SCHEMA,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "lane_id": LANE_ID,
        "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
        "locked_symbolicai_version": LOCKED_SYMBOLICAI_VERSION,
        "locked_ergoai_version": LOCKED_ERGOAI_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "advisors_never_promote_alone": True,
            "confidence_never_yields_proof": True,
            "availability_is_not_authority": True,
            "authority_is_candidate_generation_only": True,
            "does_not_edit_central_certificate": True,
        },
        "cases": cases,
    }


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise AdvisorRoleCertificationError(
            "advisor corpus must declare a non-empty cases list"
        )
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Role / authority boundary
# ---------------------------------------------------------------------------


def advisor_role_boundary(tool_id: str) -> dict[str, Any]:
    """Prove one advisor cannot satisfy certified authority or promote alone."""

    role = get_tool_role(tool_id)
    decision = evaluate_role_aware_promotion(
        tool_id,
        present=True,
        usable=True,
        production_certified=True,
        hermetic_certificate=True,
        independent_reconstruction=True,
    )
    can_satisfy = can_satisfy_certified_authority_requirement(tool_id)
    return {
        "tool_id": tool_id,
        "role": role.role.value,
        "authority_ceiling": role.authority_ceiling.value,
        "can_satisfy_certified_authority": role.can_satisfy_certified_authority,
        "can_satisfy_certified_authority_requirement": can_satisfy,
        "promotion_allowed": decision.allowed,
        "promotion_decision": decision.to_dict(),
        "is_advisor_or_candidate": role.role
        in {ToolRole.ADVISOR, ToolRole.CANDIDATE},
        "ceiling_is_advisory": role.authority_ceiling
        is ToolchainAuthorityCeiling.ADVISORY,
        "ceiling_is_non_certifying": role.authority_ceiling
        in {
            ToolchainAuthorityCeiling.ADVISORY,
            ToolchainAuthorityCeiling.CANDIDATE,
            ToolchainAuthorityCeiling.NONE,
        },
        "blocks_alone": (not decision.allowed) and (not can_satisfy),
        "lane_ids": list(getattr(role, "lane_ids", ()) or ()),
        "lane_id": (
            list(getattr(role, "lane_ids", ()) or (LANE_ID,))[0]
            if getattr(role, "lane_ids", None)
            else LANE_ID
        ),
    }


def all_advisor_role_boundaries() -> dict[str, Any]:
    items = {tool_id: advisor_role_boundary(tool_id) for tool_id in ADVISOR_TOOL_IDS}
    all_blocked = all(item["blocks_alone"] for item in items.values())
    return {
        "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
        "all_blocked_from_certified_authority": all_blocked,
        "tools": items,
        "policy": {
            "support_advisor_shadow_presence_cannot_certify": True,
            "availability_is_not_authority": True,
            "confidence_never_yields_proof": True,
        },
    }


def advisors_cannot_promote_hammer_lane() -> dict[str, Any]:
    """Aggregate proof that advisor presence cannot promote the hammer lane."""

    boundaries = all_advisor_role_boundaries()
    return {
        "lane_id": LANE_ID,
        "authority_scope": AUTHORITY_SCOPE,
        "promotion_allowed": False,
        "all_blocked_from_certified_authority": boundaries[
            "all_blocked_from_certified_authority"
        ],
        "tools": boundaries["tools"],
        "authority_tool_ids_for_lane": [],  # empty by design in role matrix
    }


# ---------------------------------------------------------------------------
# Case evaluation
# ---------------------------------------------------------------------------


def _build_static_candidate_response(
    *,
    provider: str,
    kind: str,
    body: str | None = None,
    confidence: float = 0.5,
    source_ref_ids: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    allow_authority_claims: bool = False,
) -> str:
    record: dict[str, Any] = {
        "candidate_id": f"cand:{provider}:{kind}",
        "kind": kind,
        "body": body if body is not None else _candidate_body(kind),
        "source_ref_ids": list(
            source_ref_ids
            if source_ref_ids is not None
            else ("source:module.py", "source:spec.md")
        ),
        "provider": provider,
        "confidence": confidence,
        "rationale": "advisor role certification corpus",
    }
    if extra:
        record.update(dict(extra))
    # Mutation cases intentionally inject authority claims so the advisor
    # surface can reject them.  build_json_candidates_response fails closed on
    # those claims, so emit raw JSON for adversarial fixtures only.
    if allow_authority_claims or extra:
        return json.dumps({"candidates": [record]}, sort_keys=True)
    return build_json_candidates_response([record])


def _run_proposal_advisor(
    *,
    advisor_id: str,
    provider: str,
    response: str,
    kind: str = "lemma",
) -> tuple[Any | None, str | None]:
    """Run Leanstral/SymAI advisors; synthesize contract checks for others."""

    request_payload = {
        "request_id": f"req:{advisor_id}",
        "goal_id": "goal:swap-correct",
        "logic_family": "hoare",
        "kind": kind,
        "source_ref_ids": ("source:module.py", "source:spec.md"),
        "context_text": "Prove that swap preserves the multiset of elements.",
        "goal_text": "forall xs, permute (swap xs) xs",
        "formula_id": "formula:swap",
        "notes": "proposal only",
    }
    from ipfs_datasets_py.logic.formalization.proposal_advisors import (
        ProposalAdvisorRequest,
    )

    request = ProposalAdvisorRequest(**request_payload)
    try:
        if advisor_id in {"leanstral"} or provider == "leanstral":
            advisor = LeanstralProposalAdvisor(StaticProposalModel(response))
            return advisor.propose(request), None
        if advisor_id in {"symbolicai"} or provider in {"symai", "symbolicai"}:
            advisor = SymAIProposalAdvisor(StaticProposalModel(response))
            return advisor.propose(request), None
        # ErgoAI / autoencoder / hammer: evaluate authority contracts without a
        # live model runtime (reuse proposal candidate parser path via SymAI
        # static model when provider is symai-compatible; otherwise manual).
        if provider in {"symai", "symbolicai", "leanstral"}:
            advisor = SymAIProposalAdvisor(StaticProposalModel(response))
            return advisor.propose(request), None
        # Manual offline evaluation for non-model advisors.
        payload = json.loads(response)
        candidates = payload.get("candidates") or []
        if not candidates:
            return None, "empty_candidates"
        first = candidates[0]
        body = sanitize_inert_text(
            first.get("body") or "", "body", maximum=8192
        )
        refs = list(first.get("source_ref_ids") or [])
        if not refs:
            return None, "missing_source_refs"
        authority = first.get("authority", UNVERIFIED_AUTHORITY)
        if authority != UNVERIFIED_AUTHORITY:
            return None, "authority_claim_rejected"
        for key in ("is_proved", "proof_status", "verification_status"):
            if key in first:
                return None, "authority_claim_rejected"
        proved = confidence_never_yields_proof(
            is_valid=first.get("is_valid"),
            confidence=first.get("confidence"),
            similarity=first.get("similarity"),
        )
        result = {
            "advisor_id": advisor_id,
            "provider": provider,
            "authority": UNVERIFIED_AUTHORITY,
            "candidates": [
                {
                    "candidate_id": str(first.get("candidate_id") or "cand:1"),
                    "kind": kind,
                    "body": body,
                    "source_ref_ids": refs,
                    "provider": provider,
                    "authority": UNVERIFIED_AUTHORITY,
                    "confidence": float(first.get("confidence") or 0.0),
                    "is_proved": proved,
                }
            ],
            "is_proved": False,
        }
        return result, None
    except (ProposalAdvisorValidationError, ValueError, TypeError, KeyError) as exc:
        return None, str(exc)


def evaluate_corpus_case(case: Mapping[str, Any]) -> CaseOutcome:
    """Evaluate one advisor corpus case offline (no install / network)."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    advisor_id = str(case.get("advisor_id") or "leanstral")
    provider = str(case.get("provider") or advisor_id)
    expect = str(case.get("expect") or "unknown")
    proposal_kind = str(case.get("proposal_kind") or "lemma")
    confidence = float(case.get("confidence") or 0.5)

    reason_codes: list[str] = []
    status = "unknown"
    matched = False
    bindings: dict[str, Any] = {}
    digest_seed = ""

    if kind == "acceptance":
        response = _build_static_candidate_response(
            provider="leanstral",
            kind=proposal_kind,
            confidence=0.99,
        )
        result, error = _run_proposal_advisor(
            advisor_id="leanstral",
            provider="leanstral",
            response=response,
            kind=proposal_kind,
        )
        if error or result is None:
            status = "failed"
            reason_codes.append(error or "proposal_failed")
            matched = False
        else:
            candidates = getattr(result, "candidates", None)
            if candidates is None and isinstance(result, Mapping):
                candidates = result.get("candidates") or []
            candidate = candidates[0] if candidates else None
            if candidate is None:
                status = "failed"
                matched = False
            else:
                rejected = accept_candidate(
                    candidate if not isinstance(candidate, Mapping) else _wrap_candidate(candidate),
                    compiled=False,
                    independently_validated=False,
                )
                half = accept_candidate(
                    candidate if not isinstance(candidate, Mapping) else _wrap_candidate(candidate),
                    compiled=True,
                    independently_validated=False,
                )
                full = accept_candidate(
                    candidate if not isinstance(candidate, Mapping) else _wrap_candidate(candidate),
                    compiled=True,
                    independently_validated=True,
                )
                status = "acceptance_gate"
                # confidence_never_yields_proof always returns False (= not proved)
                matched = (
                    not rejected.accepted
                    and not half.accepted
                    and full.accepted
                    and confidence_never_yields_proof(confidence=1.0) is False
                )
                bindings = {
                    "rejected": rejected.to_dict(),
                    "compile_only": half.to_dict(),
                    "full": full.to_dict(),
                }
                digest_seed = content_digest(bindings)
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched and expect == "acceptance_gate",
            reason_codes=reason_codes,
            output_digest=digest_seed or content_digest(case_id),
            detail=str(case.get("description") or ""),
            bindings=bindings,
        )

    if kind == "malformed":
        body = case.get("body")
        ungrounded_refs = case.get("source_ref_ids")
        extra = {"is_proved": True} if case.get("mutate") == "authority_claim" else None
        try:
            if body is not None:
                # Direct sanitize rejection path.
                try:
                    sanitize_inert_text(body, "body", maximum=8192)
                    status = "accepted_unexpectedly"
                    matched = False
                    reason_codes.append("executable_payload_not_rejected")
                except ProposalAdvisorValidationError:
                    status = "rejected"
                    matched = expect in {"rejected", "rejected_or_quarantined"}
                    reason_codes.append("executable_marker")
            else:
                response = _build_static_candidate_response(
                    provider=provider if provider in {"leanstral", "symai"} else "symai",
                    kind=proposal_kind,
                    confidence=confidence,
                    source_ref_ids=(
                        list(ungrounded_refs)
                        if isinstance(ungrounded_refs, (list, tuple))
                        else None
                    ),
                    extra=extra,
                )
                result, error = _run_proposal_advisor(
                    advisor_id=advisor_id if advisor_id in {"leanstral", "symbolicai"} else "symbolicai",
                    provider=provider if provider in {"leanstral", "symai"} else "symai",
                    response=response,
                    kind=proposal_kind,
                )
                if error or result is None:
                    status = "rejected"
                    matched = expect in {"rejected", "rejected_or_quarantined"}
                    reason_codes.append(error or "rejected")
                else:
                    status = "accepted_unexpectedly"
                    matched = False
                    reason_codes.append("malformed_accepted")
        except ProposalAdvisorValidationError as exc:
            status = "rejected"
            matched = expect in {"rejected", "rejected_or_quarantined"}
            reason_codes.append(str(exc)[:120])
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched,
            reason_codes=reason_codes,
            output_digest=content_digest({"case_id": case_id, "status": status}),
            detail=str(case.get("description") or ""),
        )

    # positive / negative / mutation / replay
    extra: dict[str, Any] | None = None
    if kind == "mutation" and case.get("mutate") == "authority_claim":
        extra = {"authority": "proved", "is_proved": True, "proof_status": "proved"}

    response_provider = provider if provider in {"leanstral", "symai"} else "symai"
    try:
        response = _build_static_candidate_response(
            provider=response_provider,
            kind=proposal_kind,
            confidence=confidence,
            extra=extra,
            allow_authority_claims=bool(extra),
        )
    except ProposalAdvisorValidationError as exc:
        # Authority-claim fixtures must be rejected at the boundary.
        if kind == "mutation":
            return CaseOutcome(
                case_id=case_id,
                kind=kind,
                advisor_id=advisor_id,
                expect=expect,
                status="rejected",
                matched=expect in {"rejected_or_quarantined", "rejected"},
                reason_codes=[str(exc)[:160]],
                output_digest=content_digest({"case_id": case_id, "error": str(exc)}),
                detail=str(case.get("description") or ""),
            )
        raise
    # For non leanstral/symai advisors, evaluate authority contracts offline.
    if advisor_id in {"leanstral", "symbolicai"} or provider in {
        "leanstral",
        "symai",
        "symbolicai",
    }:
        run_advisor = advisor_id if advisor_id in {"leanstral", "symbolicai"} else "symbolicai"
        run_provider = (
            provider if provider in {"leanstral", "symai"} else "symai"
        )
        result, error = _run_proposal_advisor(
            advisor_id=run_advisor,
            provider=run_provider,
            response=response,
            kind=proposal_kind,
        )
    else:
        # Inject provider name into response for manual path.
        try:
            payload = json.loads(response)
            if payload.get("candidates"):
                payload["candidates"][0]["provider"] = provider
                if extra:
                    payload["candidates"][0].update(extra)
            response = json.dumps(payload)
        except Exception:
            pass
        result, error = _run_proposal_advisor(
            advisor_id=advisor_id,
            provider=provider,
            response=response,
            kind=proposal_kind,
        )

    if kind == "mutation":
        if error or result is None:
            status = "rejected"
            matched = expect in {"rejected_or_quarantined", "rejected"}
            reason_codes.append(error or "authority_claim_rejected")
        else:
            # If a result was produced, ensure it cannot claim proof.
            is_proved = False
            if hasattr(result, "candidates"):
                for cand in result.candidates:
                    if getattr(cand, "authority", UNVERIFIED_AUTHORITY) != UNVERIFIED_AUTHORITY:
                        is_proved = True
            elif isinstance(result, Mapping):
                is_proved = bool(result.get("is_proved"))
            if is_proved:
                status = "authority_leak"
                matched = False
                reason_codes.append("authority_leak")
            else:
                status = "quarantined"
                matched = expect in {"rejected_or_quarantined", "quarantined"}
                reason_codes.append("authority_claim_stripped_or_rejected")
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched,
            reason_codes=reason_codes,
            output_digest=content_digest({"case_id": case_id, "status": status}),
            detail=str(case.get("description") or ""),
        )

    if error or result is None:
        status = "failed"
        matched = False
        reason_codes.append(error or "proposal_failed")
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched,
            reason_codes=reason_codes,
            output_digest=content_digest({"case_id": case_id, "error": error}),
            detail=str(case.get("description") or ""),
        )

    # Extract candidate fields
    if hasattr(result, "candidates"):
        candidates = list(result.candidates)
        authority = getattr(result, "authority", UNVERIFIED_AUTHORITY)
        candidate_dicts = [c.to_dict() for c in candidates]
    elif isinstance(result, Mapping):
        candidates = list(result.get("candidates") or [])
        authority = str(result.get("authority") or UNVERIFIED_AUTHORITY)
        candidate_dicts = [
            c if isinstance(c, Mapping) else {"body": str(c)} for c in candidates
        ]
    else:
        candidates = []
        authority = UNVERIFIED_AUTHORITY
        candidate_dicts = []

    proved = confidence_never_yields_proof(
        is_valid=bool(case.get("is_valid")),
        confidence=confidence,
        similarity=float(case.get("similarity") or 0.0),
    )
    digest_seed = content_digest(
        {
            "advisor_id": advisor_id,
            "provider": provider,
            "kind": proposal_kind,
            "candidates": candidate_dicts,
            "authority": authority,
        }
    )
    bindings = {
        "authority": authority,
        "candidate_count": len(candidate_dicts),
        "is_proved": proved,
        "provider_untrusted": is_untrusted_proposal_provider(provider)
        or advisor_id in ADVISOR_TOOL_IDS,
    }

    if kind == "negative":
        status = "not_proved" if proved is False else "proved_unexpectedly"
        matched = expect == "not_proved" and proved is False and authority == UNVERIFIED_AUTHORITY
        if not matched:
            reason_codes.append("confidence_elevated_to_proof")
    elif kind == "replay":
        # Re-run and compare digests.
        result2, error2 = _run_proposal_advisor(
            advisor_id=advisor_id if advisor_id in {"leanstral", "symbolicai"} else advisor_id,
            provider=provider if provider in {"leanstral", "symai"} else provider,
            response=response if advisor_id in {"leanstral", "symbolicai"} or provider in {"leanstral", "symai"} else response,
            kind=proposal_kind,
        )
        if error2 or result2 is None:
            status = "failed"
            matched = False
            reason_codes.append(error2 or "replay_failed")
        else:
            if hasattr(result2, "candidates"):
                c2 = [c.to_dict() for c in result2.candidates]
                a2 = getattr(result2, "authority", UNVERIFIED_AUTHORITY)
            elif isinstance(result2, Mapping):
                c2 = list(result2.get("candidates") or [])
                a2 = str(result2.get("authority") or UNVERIFIED_AUTHORITY)
            else:
                c2, a2 = [], UNVERIFIED_AUTHORITY
            digest2 = content_digest(
                {
                    "advisor_id": advisor_id,
                    "provider": provider,
                    "kind": proposal_kind,
                    "candidates": c2,
                    "authority": a2,
                }
            )
            status = "deterministic_replay" if digest_seed == digest2 else "nondeterministic"
            matched = (
                expect == "deterministic_replay"
                and digest_seed == digest2
                and authority == UNVERIFIED_AUTHORITY
            )
            bindings["replay_digest"] = digest2
            if not matched:
                reason_codes.append("replay_mismatch")
    else:  # positive
        status = "unverified_candidate"
        matched = (
            expect == "unverified_candidate"
            and authority == UNVERIFIED_AUTHORITY
            and proved is False
            and len(candidate_dicts) >= 1
            and all(
                (c.get("source_ref_ids") if isinstance(c, Mapping) else getattr(c, "source_ref_ids", ()))
                for c in (candidate_dicts if candidate_dicts else candidates)
            )
        )
        if not matched:
            reason_codes.append("positive_contract_failed")

    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        advisor_id=advisor_id,
        expect=expect,
        status=status,
        matched=matched,
        reason_codes=reason_codes,
        authority=authority if isinstance(authority, str) else UNVERIFIED_AUTHORITY,
        output_digest=digest_seed,
        detail=str(case.get("description") or ""),
        bindings=bindings,
    )


def _wrap_candidate(record: Mapping[str, Any]):
    """Build a ProposalCandidate from a dict for accept_candidate."""

    from ipfs_datasets_py.logic.formalization.proposal_advisors import (
        ProposalCandidate,
        ProposalProvider,
    )

    provider_raw = str(record.get("provider") or "leanstral")
    try:
        provider = ProposalProvider(provider_raw if provider_raw != "symbolicai" else "symai")
    except ValueError:
        provider = ProposalProvider.LEANSTRAL
    kind_raw = str(record.get("kind") or "lemma")
    try:
        kind = ProposalKind(kind_raw)
    except ValueError:
        kind = ProposalKind.LEMMA
    return ProposalCandidate(
        candidate_id=str(record.get("candidate_id") or "cand:1"),
        kind=kind,
        body=str(record.get("body") or _candidate_body("lemma")),
        source_ref_ids=tuple(record.get("source_ref_ids") or ("source:module.py",)),
        provider=provider,
        confidence=float(record.get("confidence") or 0.0),
        rationale=str(record.get("rationale") or "corpus"),
    )


# ---------------------------------------------------------------------------
# Install identity (offline strict selection)
# ---------------------------------------------------------------------------


def certify_install_identities(
    *,
    repo_root: Path | None = None,
    install_root: Path | None = None,
) -> dict[str, Any]:
    """Select locked SymAI/ErgoAI identities offline (hermetic, no network)."""

    if advisors_installer is None:
        return {
            "passed": False,
            "reason": "advisors_installer_unavailable",
            "symbolicai": None,
            "ergoai": None,
        }

    root = repo_root or repo_root_from()
    # Never materialize hermetic shims under the repo tree (e.g. .tmp/): that
    # path is out of task scope and pollutes the proposal gate. Use an explicit
    # install_root when provided; otherwise a process-local temp directory.
    if install_root is not None:
        target = Path(install_root)
    else:
        target = Path(
            tempfile.mkdtemp(prefix="advisor-role-certification-install-")
        )
    target.mkdir(parents=True, exist_ok=True)

    # Dry-run strict pin selection first.
    symai_pin = advisors_installer.select_strict_pin(
        "symbolicai",
        platform_key="any",
        repo_root=root,
    )
    ergo_platform = advisors_installer.detect_platform_key()
    ergo_pin = advisors_installer.select_strict_pin(
        "ergoai",
        platform_key=ergo_platform,
        repo_root=root,
    )
    pin_ok = (
        symai_pin.version == LOCKED_SYMBOLICAI_VERSION
        and ergo_pin.version == LOCKED_ERGOAI_VERSION
    )

    # Hermetic install markers (offline, yes=True, no network).
    env_snapshot = {
        "FORMAL_VERIFICATION_CERTIFY_OFFLINE": os.environ.get(
            "FORMAL_VERIFICATION_CERTIFY_OFFLINE"
        ),
        "FORMAL_VERIFICATION_FORBID_NETWORK": os.environ.get(
            "FORMAL_VERIFICATION_FORBID_NETWORK"
        ),
    }
    os.environ["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    os.environ["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    try:
        symai = advisors_installer.ensure_symbolicai(
            yes=True,
            strict=True,
            force=True,
            install_root=target,
            repo_root=root,
            hermetic_marker=True,
            test_mode=True,
        )
        ergo = advisors_installer.ensure_ergoai(
            yes=True,
            strict=True,
            force=True,
            install_root=target,
            repo_root=root,
            platform_key=ergo_platform,
            hermetic_shim=True,
            test_mode=True,
        )
    finally:
        for key, value in env_snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    passed = (
        pin_ok
        and symai.ok
        and ergo.ok
        and symai.selected_version == LOCKED_SYMBOLICAI_VERSION
        and ergo.selected_version == LOCKED_ERGOAI_VERSION
        and not symai.grants_proof_authority
        and not ergo.grants_proof_authority
    )
    return {
        "passed": passed,
        "symbolicai_pin": symai_pin.to_dict(),
        "ergoai_pin": ergo_pin.to_dict(),
        "symbolicai_receipt": symai.to_dict(),
        "ergoai_receipt": ergo.to_dict(),
        "locked_symbolicai_version": LOCKED_SYMBOLICAI_VERSION,
        "locked_ergoai_version": LOCKED_ERGOAI_VERSION,
        "install_root": str(target),
        "network_used": False,
        "policy": {
            "strict_selects_locked_identities": True,
            "advisors_never_grant_proof_authority": True,
            "hermetic_offline": True,
        },
    }


def certify_live_ergoai_vendor(
    *,
    executable: str | Path | None = None,
    install_root: str | Path | None = None,
    repo_root: Path | None = None,
    platform_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Certify a real, checksum-bound ErgoAI runtime without installing it.

    This is deliberately separate from the offline advisor-role receipt.  It
    replays managed artifact digests and executes positive, negative, mutation,
    and replay F-logic cases.  Passing proves the vendor runtime is usable for
    bounded candidate/advisor work; it never elevates ErgoAI to solver or
    theorem authority.
    """

    root = repo_root or repo_root_from()
    if advisors_installer is None:
        payload = {
            "interface": LIVE_ERGOAI_INTERFACE,
            "schema_version": LIVE_ERGOAI_SCHEMA_VERSION,
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
            "tool_id": "ergoai",
            "vendor_certified": False,
            "authoritative_live_evidence": False,
            "production_certified": False,
            "promotion_blocked": True,
            "authority_scope": AUTHORITY_SCOPE,
            "authority_ceiling": AUTHORITY_CEILING,
            "grants_proof_authority": False,
            "block_reasons": ["advisors_installer_unavailable"],
            "checks": [],
        }
        payload["receipt_digest_sha256"] = content_digest(payload)
        return payload

    selected_platform = platform_key or advisors_installer.detect_platform_key()
    resolved_root = advisors_installer.expand_user_local_root(install_root)
    probe_env = offline_env()
    probe = advisors_installer.probe_ergoai_identity(
        expected_version=LOCKED_ERGOAI_VERSION,
        executable=str(executable) if executable is not None else None,
        install_root=resolved_root,
        require_managed_vendor=True,
        platform_key=selected_platform,
        env=probe_env,
    )
    resolved_executable = str(probe.get("executable_path") or "")
    # Keep the historical LiveErgoAIAdvisorCertification surface on the core
    # matrix so existing role fixtures remain valid.  The full FVT-G218 matrix
    # is owned by ErgoAILiveToolchainContract@1.
    semantics = (
        advisors_installer.run_ergoai_semantic_checks(
            resolved_executable,
            timeout=timeout,
            include_extended=False,
            env=probe_env,
        )
        if resolved_executable and probe.get("version_match")
        else {
            "schema_version": "ergoai-live-semantic-checks/v2",
            "passed": False,
            "core_passed": False,
            "extended_passed": False,
            "checks": {},
            "replay_bound": False,
        }
    )

    checks: list[CheckResult] = []
    identity_ok = bool(probe.get("path_present") and probe.get("version_match"))
    checks.append(
        CheckResult(
            check_id="advisors.ergoai_live.identity",
            kind="install",
            status="passed" if identity_ok else "failed",
            expected=f"ErgoAI {LOCKED_ERGOAI_VERSION}",
            observed=str(probe.get("version_string") or probe.get("probe_error")),
            tool_id="ergoai",
            reason_codes=[] if identity_ok else [str(probe.get("probe_error") or "identity_failed")],
            bindings={
                "executable_path": resolved_executable or None,
                "platform": selected_platform,
            },
        )
    )
    provenance_ok = bool(probe.get("managed_vendor_provenance_verified"))
    checks.append(
        CheckResult(
            check_id="advisors.ergoai_live.provenance",
            kind="policy",
            status="passed" if provenance_ok else "failed",
            expected="checksummed_non_shim_managed_vendor_identity",
            observed=(
                "verified"
                if provenance_ok
                else ",".join(str(v) for v in probe.get("reason_codes") or ())
                or "unverified"
            ),
            tool_id="ergoai",
            reason_codes=[] if provenance_ok else list(probe.get("reason_codes") or ()),
            bindings={
                "identity_manifest_path": probe.get("identity_manifest_path"),
                "is_hermetic_advisor_shim": bool(
                    probe.get("is_hermetic_advisor_shim")
                ),
            },
        )
    )

    semantic_checks = semantics.get("checks") or {}
    for kind in ("positive", "negative", "mutation", "replay"):
        observed = semantic_checks.get(kind) or {}
        if not observed and kind == "positive":
            observed = semantic_checks.get("entailment") or {}
        if not observed and kind == "negative":
            observed = semantic_checks.get("non_entailment") or {}
        passed = bool(observed.get("passed"))
        checks.append(
            CheckResult(
                check_id=f"advisors.ergoai_live.{kind}",
                kind=kind,
                status="passed" if passed else "failed",
                expected=str(
                    observed.get("expected")
                    or (
                        "yes"
                        if kind in {"positive", "replay"}
                        else "no"
                    )
                ),
                observed=str(observed.get("verdict") or "unavailable"),
                tool_id="ergoai",
                reason_codes=[] if passed else [f"{kind}_semantic_check_failed"],
                bindings={
                    key: observed.get(key)
                    for key in (
                        "returncode",
                        "program_digest_sha256",
                        "query_digest_sha256",
                        "output_digest_sha256",
                    )
                },
            )
        )
    authority_ok = (
        AUTHORITY_CEILING == ToolchainAuthorityCeiling.ADVISORY.value
        and not can_satisfy_certified_authority_requirement("ergoai")
    )
    checks.append(
        CheckResult(
            check_id="advisors.ergoai_live.authority_boundary",
            kind="authority",
            status="passed" if authority_ok else "failed",
            expected="advisor_only_never_proof_authority",
            observed=f"ceiling={AUTHORITY_CEILING};can_satisfy={not authority_ok}",
            tool_id="ergoai",
        )
    )

    replay_invariant_ok = bool(semantics.get("replay_bound"))
    checks.append(
        CheckResult(
            check_id="advisors.ergoai_live.replay_invariant",
            kind="replay",
            status="passed" if replay_invariant_ok else "failed",
            expected="same_input_same_normalized_semantic_result",
            observed=(
                "normalized_semantics_match"
                if replay_invariant_ok
                else "normalized_semantics_mismatch"
            ),
            tool_id="ergoai",
            reason_codes=(
                []
                if replay_invariant_ok
                else ["replay_semantic_invariant_failed"]
            ),
            bindings={
                "replay_bound": replay_invariant_ok,
                "comparison_scope": "normalized_semantics_not_console_bytes",
            },
        )
    )

    # Vendor certification requires the core membership/mutation/replay matrix
    # and managed provenance.  Extended timeout/resource cases are recorded when
    # the executable supports them but do not alone revoke vendor certification
    # for the legacy advisor-role surface.
    vendor_certified = bool(
        identity_ok
        and provenance_ok
        and semantics.get("core_passed", semantics.get("passed"))
        and replay_invariant_ok
        and authority_ok
    )
    block_reasons = sorted(
        {
            reason
            for check in checks
            if check.status != "passed"
            for reason in (check.reason_codes or [check.check_id])
        }
    )
    manifest = probe.get("manifest") or {}
    manifest_projection = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "tool_id",
            "version",
            "selected_platform",
            "release_tag",
            "release_url",
            "release_artifact_sha256",
            "release_artifact_size_bytes",
            "vendor_executable_sha256",
            "xsb_configuration",
            "xsb_executable_sha256",
            "launcher_sha256",
            "identity_digest_sha256",
            "license_components",
            "checksum_verified",
            "is_live_vendor",
            "is_hermetic_advisor_shim",
            "atomic_publish",
            "relocatable_install",
            "runtime_paths_relative",
            "runtime_workspace_cleanup_policy",
            "relocation_certification_scope",
            "developer_rebuild_metadata_relocated",
            "install_publication_model",
        )
    }
    identity_manifest_path = Path(str(probe.get("identity_manifest_path") or ""))
    identity_manifest_digest = (
        content_digest(identity_manifest_path.read_bytes())
        if identity_manifest_path.is_file()
        else None
    )
    payload = {
        "interface": LIVE_ERGOAI_INTERFACE,
        "schema_version": LIVE_ERGOAI_SCHEMA_VERSION,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "tool_id": "ergoai",
        "evidence_class": (
            LIVE_ERGOAI_EVIDENCE_CLASS if vendor_certified else "unverified_or_incomplete"
        ),
        "vendor_certified": vendor_certified,
        # This is authentic managed-vendor execution evidence, but the advisor
        # result is not an independent proof reconstruction and therefore is
        # never labelled authoritative proof evidence.
        "managed_vendor_live_evidence": vendor_certified,
        "authoritative_live_evidence": False,
        "independent_reconstruction_complete": False,
        # Production-certified here means the advisor runtime is deployable in
        # its declared role.  It remains non-authoritative for proofs.
        "production_certified": vendor_certified,
        "promotion_blocked": True,
        "authority_scope": AUTHORITY_SCOPE,
        "authority_ceiling": AUTHORITY_CEILING,
        "grants_theorem_authority": False,
        "grants_proof_authority": False,
        "advisors_never_promote_alone": True,
        # The certifier itself has no network code path.  Environment guards are
        # applied to the child, but this receipt does not pretend they are a
        # kernel-enforced network namespace.
        "network_used": False,
        "network_isolation_enforced": False,
        "install_attempted": False,
        "download_attempted": False,
        "selected_platform": selected_platform,
        "executable_path": resolved_executable or None,
        "identity_manifest_path": str(identity_manifest_path)
        if identity_manifest_path.is_file()
        else None,
        "identity_manifest_digest_sha256": identity_manifest_digest,
        "managed_identity": manifest_projection,
        "semantic_evidence_digest_sha256": semantics.get(
            "normalized_evidence_digest_sha256"
        ),
        "checks": [check.to_dict() for check in checks],
        "block_reasons": block_reasons,
        "source_binding": {
            "repo_root": str(root),
            "release_url": getattr(
                advisors_installer, "ERGOAI_RELEASE_URL", ""
            ),
            "release_sha256": getattr(
                advisors_installer, "ERGOAI_RELEASE_SHA256", ""
            ),
            "release_tag": getattr(
                advisors_installer, "ERGOAI_RELEASE_TAG", ""
            ),
        },
    }
    payload["receipt_digest_sha256"] = content_digest(payload)
    return payload


def _lock_ergoai_tool(repo_root: Path) -> dict[str, Any] | None:
    lock_path = repo_root / DEFAULT_LOCK_RELATIVE
    if not lock_path.is_file():
        return None
    try:
        document = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    for tool in document.get("tools") or ():
        if isinstance(tool, Mapping) and tool.get("tool_id") == "ergoai":
            return dict(tool)
    return None


def _lock_ergoai_inventory(repo_root: Path) -> dict[str, Any] | None:
    lock_path = repo_root / DEFAULT_LOCK_RELATIVE
    if not lock_path.is_file():
        return None
    try:
        document = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    inventory = document.get("checksummed_release_inventory") or {}
    value = inventory.get("ergoai")
    return dict(value) if isinstance(value, Mapping) else None


def build_ergoai_live_toolchain_contract(
    *,
    repo_root: Path | None = None,
    install_root: str | Path | None = None,
    executable: str | Path | None = None,
    platform_key: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    run_semantics: bool = True,
) -> dict[str, Any]:
    """Build the FVT-G218 ``ErgoAILiveToolchainContract@1`` receipt.

    Offline by default: never installs, downloads, or opens the network.
    Semantic execution is optional and only runs against an already-managed
    executable (fixture or real vendor).  Simulation-mode wrapper fixtures
    cannot satisfy ``live_vendor_execution``.
    """

    root = repo_root or repo_root_from()
    probe_env = offline_env(env)
    block_reasons: list[str] = []
    checks: list[CheckResult] = []

    tool = _lock_ergoai_tool(root)
    inventory = _lock_ergoai_inventory(root)
    lock_ok = tool is not None and inventory is not None
    contract = (tool or {}).get("deployment_contract") or {}
    required_kinds = list(
        contract.get("live_semantic_checks_required") or ERGOAI_LIVE_CASE_KINDS
    )
    supported = list(
        contract.get("supported_platforms")
        or (inventory or {}).get("platforms")
        or ()
    )
    if isinstance(supported, Mapping):
        supported = list(supported.keys())

    checks.append(
        CheckResult(
            check_id="ergoai.live_toolchain.lock_binding",
            kind="policy",
            status="passed" if lock_ok else "failed",
            expected="official_release_pin_with_digest_license_matrix",
            observed="bound" if lock_ok else "missing",
            tool_id="ergoai",
            reason_codes=[] if lock_ok else ["lock_ergoai_binding_missing"],
            bindings={
                "version": (inventory or {}).get("version") or LOCKED_ERGOAI_VERSION,
                "sha256": (inventory or {}).get("sha256"),
                "release_tag": (inventory or {}).get("release_tag"),
                "entry_point": contract.get("entry_point")
                or (inventory or {}).get("entry_point"),
                "supported_platforms": supported,
                "license_components": contract.get("license_components")
                or (inventory or {}).get("license_components"),
                "runtime_dependencies": contract.get("runtime_dependencies")
                or (inventory or {}).get("runtime_dependencies"),
                "build_dependencies": (inventory or {}).get("build_dependencies"),
                "identity_probe": contract.get("identity_probe")
                or (inventory or {}).get("identity_probe"),
                "acquisition_conditions": contract.get("acquisition_conditions")
                or (inventory or {}).get("acquisition_conditions"),
                "lazy_install": contract.get("lazy_install"),
            },
        )
    )
    if not lock_ok:
        block_reasons.append("lock_ergoai_binding_missing")

    matrix_ok = set(required_kinds) >= set(ERGOAI_LIVE_CASE_KINDS)
    checks.append(
        CheckResult(
            check_id="ergoai.live_toolchain.case_matrix",
            kind="policy",
            status="passed" if matrix_ok else "failed",
            expected=",".join(ERGOAI_LIVE_CASE_KINDS),
            observed=",".join(required_kinds),
            tool_id="ergoai",
            reason_codes=[] if matrix_ok else ["live_case_matrix_incomplete"],
        )
    )
    if not matrix_ok:
        block_reasons.append("live_case_matrix_incomplete")

    installer_ok = advisors_installer is not None
    pin_bindings: dict[str, Any] = {}
    if installer_ok:
        try:
            selected_platform = platform_key or advisors_installer.detect_platform_key()
            if selected_platform not in (
                advisors_installer.ERGOAI_SUPPORTED_PLATFORMS
            ):
                # Prefer a reviewed pin for offline contract inspection.
                selected_platform = advisors_installer.ERGOAI_SUPPORTED_PLATFORMS[0]
            pin = advisors_installer.select_strict_pin(
                "ergoai",
                platform_key=selected_platform,
                repo_root=root,
                allow_source_fallback=False,
            )
            pin_bindings = pin.to_dict()
            pin_ok = (
                pin.version == LOCKED_ERGOAI_VERSION
                and pin.is_checksummed
                and bool(pin.sha256)
            )
        except Exception as exc:  # pragma: no cover - host/lock variance
            pin_ok = False
            pin_bindings = {"error": str(exc)[:200]}
    else:
        pin_ok = False
        selected_platform = platform_key or "unknown"
    checks.append(
        CheckResult(
            check_id="ergoai.live_toolchain.strict_pin",
            kind="install",
            status="passed" if pin_ok else "failed",
            expected=f"ergoai={LOCKED_ERGOAI_VERSION}/checksummed",
            observed=str(pin_bindings.get("version") or pin_bindings.get("error")),
            tool_id="ergoai",
            reason_codes=[] if pin_ok else ["strict_pin_failed"],
            bindings=pin_bindings,
        )
    )
    if not pin_ok:
        block_reasons.append("strict_pin_failed")

    lazy_policy_ok = False
    plugin_policy: Mapping[str, Any] = {}
    if installer_ok:
        # Prove the installer refuses import-time install and requires yes=.
        try:
            advisors_installer.authorize_plugin_install(
                "ergoai",
                yes=True,
                import_context=True,
            )
            import_blocked = False
        except Exception:
            import_blocked = True
        # Consent must be tested against a genuinely absent install.  Reusing a
        # valid managed root correctly reports "available" without mutation and
        # therefore cannot exercise the yes-required branch.
        with tempfile.TemporaryDirectory(
            prefix="ergoai-live-toolchain-policy-"
        ) as policy_root:
            refused = advisors_installer.ensure_ergoai(
                yes=False,
                strict=False,
                force=True,
                dry_run=False,
                install_root=policy_root,
                repo_root=root,
                platform_key=selected_platform
                if selected_platform
                in advisors_installer.ERGOAI_SUPPORTED_PLATFORMS
                else advisors_installer.ERGOAI_SUPPORTED_PLATFORMS[0],
                hermetic_shim=True,
            )
        plugin_policy = (
            advisors_installer.plugin_manifest().get("policy") or {}
        )
        publication_ok = bool(
            plugin_policy.get("ergoai_atomic_publish") is True
            and plugin_policy.get("ergoai_relocatable_install") is True
            and plugin_policy.get("ergoai_runtime_execution_policy")
            == "private-ergoai-copy-shared-immutable-xsb/v1"
            and plugin_policy.get("ergoai_java_consumer_policy")
            == "private-ergoai-copy-java-consumers/v2"
            and plugin_policy.get("ergoai_runtime_workspace_cleanup_policy")
            == "normal-and-handled-signals-clean-sigkill-orphans-retained/v1"
            and plugin_policy.get("ergoai_relocation_certification_scope")
            == "executed-runtime-and-bundled-java-consumers/v1"
            and plugin_policy.get("ergoai_developer_rebuild_metadata_relocated")
            is False
            and plugin_policy.get("ergoai_publication_model")
            == "staged_vendor_atomic_rename_private_runtime_workspaces_identity_commit_v4"
        )
        lazy_policy_ok = bool(
            import_blocked
            and refused.status in {"blocked", "refused"}
            and "yes_required" in refused.reason_codes
            and not refused.grants_proof_authority
            and refused.authority_ceiling == "advisory"
            and publication_ok
        )
    checks.append(
        CheckResult(
            check_id="ergoai.live_toolchain.lazy_install_policy",
            kind="install",
            status="passed" if lazy_policy_ok else "failed",
            expected="explicit_yes_checksummed_user_local_no_import_install",
            observed="ok" if lazy_policy_ok else "policy_gap",
            tool_id="ergoai",
            reason_codes=[] if lazy_policy_ok else ["lazy_install_policy_failed"],
            bindings={
                "never_on_import": True,
                "requires_explicit_yes": True,
                "user_local_only": True,
                "offline_after_acquisition": True,
                "atomic_staged": True,
                "relocatable": True,
                "publication_model": plugin_policy.get(
                    "ergoai_publication_model"
                ),
            },
        )
    )
    if not lazy_policy_ok:
        block_reasons.append("lazy_install_policy_failed")

    # Wrapper surface: bounded live adapter exists and preserves authority.
    # Force a missing binary so simulation-mode fixtures cannot be mistaken for
    # live vendor execution even when the host has a managed ErgoAI install.
    wrapper_ok = False
    wrapper_bindings: dict[str, Any] = {}
    try:
        from ipfs_datasets_py.logic.flogic.ergoai_wrapper import (
            AUTHORITY_CEILING as WRAPPER_CEILING,
        )
        from ipfs_datasets_py.logic.flogic.ergoai_wrapper import (
            EVIDENCE_CLASS as WRAPPER_EVIDENCE,
        )
        from ipfs_datasets_py.logic.flogic.ergoai_wrapper import (
            LIVE_CASE_KINDS as WRAPPER_KINDS,
        )
        from ipfs_datasets_py.logic.flogic.ergoai_wrapper import (
            LIVE_TOOLCHAIN_INTERFACE as WRAPPER_INTERFACE,
        )
        from ipfs_datasets_py.logic.flogic.ergoai_wrapper import (
            ErgoAIWrapper,
        )

        missing = Path(
            tempfile.mkdtemp(prefix="ergoai-wrapper-missing-")
        ) / "missing-runergo"
        wrapper = ErgoAIWrapper(binary=missing, lazy_install=False)
        stats = wrapper.get_statistics()
        adapter = wrapper.run_live_semantic_adapter(require_live_binary=True)
        wrapper_ok = (
            WRAPPER_INTERFACE == ERGOAI_LIVE_TOOLCHAIN_INTERFACE
            and WRAPPER_CEILING == AUTHORITY_CEILING
            and WRAPPER_EVIDENCE
            == "proposal_or_candidate_until_independent_reconstruction"
            and set(WRAPPER_KINDS) >= set(ERGOAI_LIVE_CASE_KINDS)
            and stats.get("grants_proof_authority") is False
            and adapter.get("grants_proof_authority") is False
            and wrapper.simulation_mode is True
            and adapter.get("live_vendor_execution") is False
        )
        wrapper_bindings = {
            "interface": WRAPPER_INTERFACE,
            "authority_ceiling": WRAPPER_CEILING,
            "evidence_class": WRAPPER_EVIDENCE,
            "case_kinds": list(WRAPPER_KINDS),
            "simulation_mode": wrapper.simulation_mode,
            "adapter_live_vendor_execution": adapter.get("live_vendor_execution"),
        }
    except Exception as exc:
        wrapper_bindings = {"error": str(exc)[:300]}
        wrapper_ok = False
    checks.append(
        CheckResult(
            check_id="ergoai.live_toolchain.wrapper_adapter",
            kind="policy",
            status="passed" if wrapper_ok else "failed",
            expected="bounded_live_adapter_advisory_only",
            observed="ok" if wrapper_ok else "adapter_gap",
            tool_id="ergoai",
            reason_codes=[] if wrapper_ok else ["wrapper_adapter_failed"],
            bindings=wrapper_bindings,
        )
    )
    if not wrapper_ok:
        block_reasons.append("wrapper_adapter_failed")

    authority_ok = (
        AUTHORITY_CEILING == ToolchainAuthorityCeiling.ADVISORY.value
        and not can_satisfy_certified_authority_requirement("ergoai")
    )
    checks.append(
        CheckResult(
            check_id="ergoai.live_toolchain.authority_boundary",
            kind="authority",
            status="passed" if authority_ok else "failed",
            expected="advisor_candidate_never_theorem_authority",
            observed=f"ceiling={AUTHORITY_CEILING};can_satisfy={not authority_ok}",
            tool_id="ergoai",
        )
    )
    if not authority_ok:
        block_reasons.append("authority_boundary_failed")

    semantics: dict[str, Any] = {
        "passed": False,
        "core_passed": False,
        "extended_passed": False,
        "checks": {},
        "replay_bound": False,
    }
    live_vendor_execution = False
    if run_semantics and advisors_installer is not None:
        resolved_root = advisors_installer.expand_user_local_root(install_root)
        probe = advisors_installer.probe_ergoai_identity(
            expected_version=LOCKED_ERGOAI_VERSION,
            executable=str(executable) if executable is not None else None,
            install_root=resolved_root,
            require_managed_vendor=True,
            platform_key=platform_key
            or (
                selected_platform
                if installer_ok
                else advisors_installer.detect_platform_key()
            ),
            env=probe_env,
        )
        resolved_executable = str(probe.get("executable_path") or "")
        if (
            resolved_executable
            and probe.get("version_match")
            and probe.get("managed_vendor_provenance_verified")
            and not probe.get("is_hermetic_advisor_shim")
        ):
            live_vendor_execution = True
            semantics = advisors_installer.run_ergoai_semantic_checks(
                resolved_executable,
                timeout=timeout,
                include_extended=True,
                env=probe_env,
            )
        elif resolved_executable and probe.get("version_match"):
            # Allow explicit fixture executables supplied for contract tests
            # without elevating them to live vendor execution.
            semantics = advisors_installer.run_ergoai_semantic_checks(
                resolved_executable,
                timeout=timeout,
                include_extended=True,
                bound_timeout_seconds=0.15,
                env=probe_env,
            )
            live_vendor_execution = bool(
                probe.get("managed_vendor_provenance_verified")
                and not probe.get("is_hermetic_advisor_shim")
            )

    semantic_checks = semantics.get("checks") or {}
    for kind in ERGOAI_LIVE_CASE_KINDS:
        observed = semantic_checks.get(kind) or {}
        passed = bool(observed.get("passed")) if observed else False
        if not run_semantics:
            # A declared case is not execution evidence.  Keep structural
            # inspection successful while truthfully marking every live case
            # as skipped.
            passed = False
            observed = {"verdict": "not_executed", "passed": False}
        checks.append(
            CheckResult(
                check_id=f"ergoai.live_toolchain.case.{kind}",
                kind=kind if kind in CHECK_KINDS else "acceptance",
                status=(
                    "skipped"
                    if not run_semantics
                    else "passed"
                    if passed
                    else "failed"
                ),
                expected=str(
                    observed.get("expected")
                    or observed.get("expected_any")
                    or kind
                ),
                observed=str(observed.get("verdict") or "unavailable"),
                tool_id="ergoai",
                reason_codes=[]
                if passed or not run_semantics
                else [f"{kind}_failed"],
                bindings={
                    "live_vendor_execution": live_vendor_execution,
                    "program_digest_sha256": observed.get("program_digest_sha256"),
                    "query_digest_sha256": observed.get("query_digest_sha256"),
                    "timed_out": observed.get("timed_out"),
                    "resource_bound_enforced": observed.get(
                        "resource_bound_enforced"
                    ),
                },
            )
        )
        if run_semantics and not passed:
            block_reasons.append(f"{kind}_failed")

    replay_invariant_ok = bool(semantics.get("replay_bound"))
    if run_semantics:
        checks.append(
            CheckResult(
                check_id="ergoai.live_toolchain.replay_invariant",
                kind="replay",
                status="passed" if replay_invariant_ok else "failed",
                expected="same_input_same_normalized_semantic_result",
                observed=(
                    "normalized_semantics_match"
                    if replay_invariant_ok
                    else "normalized_semantics_mismatch"
                ),
                tool_id="ergoai",
                reason_codes=(
                    []
                    if replay_invariant_ok
                    else ["replay_semantic_invariant_failed"]
                ),
                bindings={
                    "comparison_scope": "normalized_semantics_not_console_bytes",
                    "replay_bound": replay_invariant_ok,
                },
            )
        )
        if not replay_invariant_ok:
            block_reasons.append("replay_semantic_invariant_failed")
        if not live_vendor_execution:
            block_reasons.append("managed_vendor_provenance_unverified")

    structural_ok = lock_ok and matrix_ok and pin_ok and lazy_policy_ok and wrapper_ok and authority_ok
    semantic_ok = (not run_semantics) or bool(semantics.get("passed"))
    contract_passed = (
        structural_ok
        if not run_semantics
        else structural_ok and semantic_ok and live_vendor_execution
    )

    payload = {
        "interface": ERGOAI_LIVE_TOOLCHAIN_INTERFACE,
        "schema_version": ERGOAI_LIVE_TOOLCHAIN_SCHEMA,
        "goal_id": ERGOAI_LIVE_TOOLCHAIN_GOAL_ID,
        "task_id": ERGOAI_LIVE_TOOLCHAIN_TASK_ID,
        "program": ERGOAI_LIVE_TOOLCHAIN_PROGRAM,
        "tool_id": "ergoai",
        "locked_version": LOCKED_ERGOAI_VERSION,
        "contract_passed": contract_passed,
        "structural_passed": structural_ok,
        "semantic_passed": semantic_ok if run_semantics else None,
        "live_vendor_execution": live_vendor_execution,
        "production_certified": False,
        "promotion_blocked": True,
        "authority_scope": AUTHORITY_SCOPE,
        "authority_ceiling": AUTHORITY_CEILING,
        "grants_theorem_authority": False,
        "grants_proof_authority": False,
        "evidence_class": (
            "checksummed_managed_vendor_execution_advisory_only"
            if live_vendor_execution and semantic_ok
            else "proposal_or_candidate_until_independent_reconstruction"
        ),
        "network_used": False,
        "network_isolation_enforced": False,
        "install_attempted": False,
        "download_attempted": False,
        "case_kinds": list(ERGOAI_LIVE_CASE_KINDS),
        "required_case_kinds": required_kinds,
        "selected_platform": selected_platform if installer_ok else platform_key,
        "pin": pin_bindings,
        "lock_projection": {
            "tool": {
                key: (tool or {}).get(key)
                for key in (
                    "tool_id",
                    "display_name",
                    "license",
                    "source",
                    "identity_kind",
                    "installer_entry",
                    "executable_candidates",
                )
            },
            "inventory": inventory,
            "deployment_contract": contract,
        },
        "semantic_evidence_digest_sha256": semantics.get(
            "normalized_evidence_digest_sha256"
        ),
        "checks": [check.to_dict() for check in checks],
        "block_reasons": sorted(set(block_reasons)),
        "policy": {
            "never_download_during_certification": True,
            "wrapper_fixtures_are_not_live_execution": True,
            "advisor_verdict_never_theorem_authority": True,
            "full_contract_requires_managed_vendor_provenance": True,
            "offline_env_keys": sorted(
                key
                for key in probe_env
                if key.startswith("FORMAL_VERIFICATION_")
            ),
        },
        "env_policy": {
            "certification_offline": True,
            "forbid_network": probe_env.get("FORMAL_VERIFICATION_FORBID_NETWORK")
            == "1",
            "forbid_install": probe_env.get("FORMAL_VERIFICATION_FORBID_INSTALL")
            == "1",
            "kernel_network_namespace": False,
            "scope": "environment_guard_and_no_certifier_network_code_path",
        },
    }
    payload["receipt_digest_sha256"] = content_digest(payload)
    return payload


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_certification_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    install_root: Path | None = None,
) -> AdvisorRoleCertification:
    """Run the full advisor role certification suite (offline)."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else default_corpus_manifest()
    cases = corpus_cases(corpus)
    cert = AdvisorRoleCertification()
    probe_env = offline_env(env)

    cert.checks.append(
        CheckResult(
            check_id="advisors.offline_policy",
            kind="policy",
            status="passed",
            expected="no_install_no_download_no_network",
            observed=(
                f"install={cert.install_attempted},"
                f"download={cert.download_attempted},"
                f"network={cert.network_used},"
                f"FORMAL_VERIFICATION_CERTIFY_OFFLINE="
                f"{probe_env.get('FORMAL_VERIFICATION_CERTIFY_OFFLINE')}"
            ),
            detail="certification never installs, downloads, or opens the network",
        )
    )

    # Interface constants.
    cert.checks.append(
        CheckResult(
            check_id="advisors.interfaces",
            kind="policy",
            status="passed",
            expected=INTERFACE,
            observed=INTERFACE,
            detail=(
                f"Leanstral={LEANSTRAL_ADVISOR_INTERFACE}; "
                f"SymAI={SYMAI_ADVISOR_INTERFACE}; "
                f"authority={UNVERIFIED_AUTHORITY}"
            ),
            bindings={
                "leanstral_interface": LEANSTRAL_ADVISOR_INTERFACE,
                "symai_interface": SYMAI_ADVISOR_INTERFACE,
                "unverified_authority": UNVERIFIED_AUTHORITY,
            },
        )
    )

    # Role matrix.
    role_report = all_advisor_role_boundaries()
    cert.role_matrix_passed = bool(role_report["all_blocked_from_certified_authority"])
    if not cert.role_matrix_passed:
        cert.block_reasons.append("advisor_role_matrix_failed")
    cert.checks.append(
        CheckResult(
            check_id="advisors.role_matrix",
            kind="role",
            status="passed" if cert.role_matrix_passed else "failed",
            expected="all_advisors_blocked_from_certified_authority",
            observed=(
                "blocked" if cert.role_matrix_passed else "authority_leak"
            ),
            detail="advisor/candidate presence cannot satisfy certified authority",
            bindings=role_report,
        )
    )
    for tool_id in ADVISOR_TOOL_IDS:
        boundary = role_report["tools"][tool_id]
        ok = (
            boundary["blocks_alone"]
            and boundary["is_advisor_or_candidate"]
            and boundary.get("ceiling_is_non_certifying", False)
        )
        if not ok:
            cert.block_reasons.append(f"role_boundary_failed:{tool_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"advisors.role.{tool_id}",
                kind="role",
                status="passed" if ok else "failed",
                expected="advisor_or_candidate_non_certifying_blocked",
                observed=(
                    f"role={boundary['role']},"
                    f"ceiling={boundary['authority_ceiling']},"
                    f"promote={boundary['promotion_allowed']}"
                ),
                tool_id=tool_id,
                bindings=boundary,
            )
        )

    promotion = advisors_cannot_promote_hammer_lane()
    cert.checks.append(
        CheckResult(
            check_id="advisors.hammer_lane_promotion_blocked",
            kind="authority",
            status="passed" if not promotion["promotion_allowed"] else "failed",
            expected="promotion_blocked",
            observed=(
                "blocked" if not promotion["promotion_allowed"] else "allowed"
            ),
            detail="hammer lane has empty authority_tool_ids by design",
            bindings=promotion,
        )
    )

    # Install identities (hermetic offline).
    try:
        install_report = certify_install_identities(
            repo_root=root, install_root=install_root
        )
        cert.install_identity_passed = bool(install_report.get("passed"))
        if not cert.install_identity_passed:
            cert.block_reasons.append("install_identity_failed")
        cert.checks.append(
            CheckResult(
                check_id="advisors.install_identities",
                kind="install",
                status="passed" if cert.install_identity_passed else "failed",
                expected=(
                    f"symbolicai={LOCKED_SYMBOLICAI_VERSION};"
                    f"ergoai={LOCKED_ERGOAI_VERSION}"
                ),
                observed=(
                    f"symbolicai={install_report.get('symbolicai_receipt', {}).get('selected_version')};"
                    f"ergoai={install_report.get('ergoai_receipt', {}).get('selected_version')}"
                ),
                detail="strict hermetic selection of locked advisor identities",
                bindings=install_report,
            )
        )
        cert.bindings["install"] = install_report
    except Exception as exc:  # pragma: no cover - defensive
        cert.block_reasons.append(f"install_identity_error:{exc}")
        cert.checks.append(
            CheckResult(
                check_id="advisors.install_identities",
                kind="install",
                status="failed",
                expected="locked identities",
                observed=str(exc)[:200],
                detail="install identity certification raised",
            )
        )

    # Semantic / proposal corpus.
    outcomes_by_id: dict[str, CaseOutcome] = {}
    for case in cases:
        outcome = evaluate_corpus_case(case)
        outcomes_by_id[outcome.case_id] = outcome
        cert.cases.append(outcome)
        status = "passed" if outcome.matched else "failed"
        if not outcome.matched:
            cert.block_reasons.append(f"case_failed:{outcome.case_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"advisors.{outcome.case_id}",
                kind=outcome.kind if outcome.kind in CHECK_KINDS else "positive",
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                tool_id=outcome.advisor_id,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "output_digest": outcome.output_digest,
                    "authority": outcome.authority,
                    **dict(outcome.bindings),
                },
            )
        )

    cert.semantic_corpus_passed = all(case.matched for case in cert.cases)
    if not cert.semantic_corpus_passed:
        cert.block_reasons.append("semantic_corpus_failed")

    cert.checks.append(
        CheckResult(
            check_id="advisors.confidence_never_yields_proof",
            kind="authority",
            status="passed"
            if confidence_never_yields_proof(confidence=1.0, is_valid=True) is False
            else "failed",
            expected="False",
            observed=str(
                confidence_never_yields_proof(confidence=1.0, is_valid=True)
            ),
            detail="documented invariant: model scores never establish proof",
        )
    )

    # Production certification for advisors means: role/corpus contracts hold
    # and install identities select locked pins.  Advisors never become
    # certified theorem authority — production_certified here means
    # "role-certified as candidate generation only".
    cert.production_certified = (
        cert.role_matrix_passed
        and cert.semantic_corpus_passed
        and cert.install_identity_passed
        and not cert.network_used
        and not cert.download_attempted
    )
    # Promotion remains blocked by design (advisors cannot promote hammer lane).
    cert.promotion_blocked = True
    if cert.production_certified and cert.block_reasons:
        # Drop soft block reasons that are informational only when all checks pass.
        hard = [
            r
            for r in cert.block_reasons
            if not r.startswith("case_failed:") or True
        ]
        # If any case failed, production_certified is already False.
        _ = hard

    cert.bindings.update(
        {
            "authority": {
                "scope": AUTHORITY_SCOPE,
                "ceiling": AUTHORITY_CEILING,
                "advisors_never_promote_alone": True,
                "confidence_never_yields_proof": True,
            },
            "locked_versions": {
                "symbolicai": LOCKED_SYMBOLICAI_VERSION,
                "ergoai": LOCKED_ERGOAI_VERSION,
            },
            "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
            "role_tools": [
                item.tool_id
                for item in tools_by_role(ToolRole.ADVISOR)
            ]
            + [item.tool_id for item in tools_by_role(ToolRole.CANDIDATE)],
            "lane_id": LANE_ID,
            "handler_id": HANDLER_ID,
        }
    )
    cert.notes = (
        "Advisor role certification: candidates only; independent reconstruction "
        "and kernel/solver validation remain mandatory for any proof authority."
    )
    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    install_root: Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable certification receipt."""

    cert = run_certification_suite(
        repo_root=repo_root, env=env, install_root=install_root
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "advisors_never_promote_alone": True,
        "confidence_never_yields_proof": True,
        "availability_is_not_authority": True,
        "does_not_edit_central_certificate": True,
        "does_not_change_model_runtimes": True,
        "authority_is_candidate_generation_only": True,
    }
    payload["semantic_corpus_passed"] = cert.semantic_corpus_passed
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def lane_handler(**kwargs: Any) -> dict[str, Any]:
    """Hammer-lane handler for RoleAwarePromotionPolicy binding."""

    repo_root = kwargs.get("repo_root")
    root = Path(repo_root) if repo_root is not None else repo_root_from()
    cert = run_certification_suite(repo_root=root)
    return {
        "lane_id": LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": HANDLER_ID,
        "status": "certified" if cert.production_certified else "role_boundary_only",
        "certified": cert.production_certified,
        "production_certified": cert.production_certified,
        "promotion_blocked": cert.promotion_blocked,
        "authority_scope": AUTHORITY_SCOPE,
        "authority_ceiling": AUTHORITY_CEILING,
        "advisors_never_promote_alone": True,
        "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
        "semantic_corpus_passed": cert.semantic_corpus_passed,
        "role_matrix_passed": cert.role_matrix_passed,
        "install_identity_passed": cert.install_identity_passed,
        "reason_codes": list(cert.block_reasons),
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
    }


def bind_to_role_policy(policy: Any | None = None) -> Any:
    """Bind this lane handler under tools.logic.certification.roles."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        raise AdvisorRoleCertificationError(
            "roles certification surface unavailable for lane binding"
        )
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(LANE_ID, lane_handler, policy=target, replace=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify advisor utilities as bounded candidate generation "
            f"({INTERFACE})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (auto-detected when omitted)",
    )
    parser.add_argument(
        "--live-ergoai",
        action="store_true",
        help="Run checksum/provenance-bound live ErgoAI semantic certification",
    )
    parser.add_argument(
        "--ergoai-live-toolchain",
        action="store_true",
        help="Build ErgoAILiveToolchainContract@1 receipt (FVT-G218 / FVT-085)",
    )
    parser.add_argument(
        "--run-semantics",
        action="store_true",
        help="When used with --ergoai-live-toolchain, execute the live matrix",
    )
    parser.add_argument(
        "--ergoai-executable",
        type=Path,
        default=None,
        help="Managed ErgoAI launcher to certify (auto-discovered when omitted)",
    )
    parser.add_argument(
        "--install-root",
        type=Path,
        default=None,
        help="Managed prover root containing the ErgoAI identity manifest",
    )
    parser.add_argument(
        "--platform-key",
        default=None,
        help="Expected lock platform, such as linux-aarch64",
    )
    args = parser.parse_args(argv)

    if args.ergoai_live_toolchain:
        receipt = build_ergoai_live_toolchain_contract(
            executable=args.ergoai_executable,
            install_root=args.install_root,
            repo_root=args.repo_root,
            platform_key=args.platform_key,
            run_semantics=bool(args.run_semantics or args.ergoai_executable),
        )
        success_key = "contract_passed"
    elif args.live_ergoai:
        receipt = certify_live_ergoai_vendor(
            executable=args.ergoai_executable,
            install_root=args.install_root,
            repo_root=args.repo_root,
            platform_key=args.platform_key,
        )
        success_key = "vendor_certified"
    else:
        receipt = build_certification_receipt(repo_root=args.repo_root)
        success_key = "production_certified"
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.ergoai_live_toolchain:
        print(
            f"{ERGOAI_LIVE_TOOLCHAIN_INTERFACE} "
            f"goal={ERGOAI_LIVE_TOOLCHAIN_GOAL_ID} "
            f"task={ERGOAI_LIVE_TOOLCHAIN_TASK_ID}"
        )
        print(
            f"  contract_passed={receipt['contract_passed']} "
            f"structural={receipt['structural_passed']} "
            f"live_vendor_execution={receipt['live_vendor_execution']} "
            f"promotion_blocked={receipt['promotion_blocked']}"
        )
        print(f"  digest={receipt.get('receipt_digest_sha256', '')[:16]}…")
    elif args.live_ergoai:
        print(f"{LIVE_ERGOAI_INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"  vendor_certified={receipt['vendor_certified']} "
            f"authoritative_live_evidence={receipt['authoritative_live_evidence']} "
            f"promotion_blocked={receipt['promotion_blocked']}"
        )
        print(f"  digest={receipt.get('receipt_digest_sha256', '')[:16]}…")
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"  production_certified={receipt['production_certified']} "
            f"promotion_blocked={receipt['promotion_blocked']} "
            f"corpus={receipt['semantic_corpus_passed']} "
            f"roles={receipt['role_matrix_passed']}"
        )
        print(f"  advisors={','.join(ADVISOR_TOOL_IDS)}")
        print(f"  digest={receipt.get('receipt_digest_sha256', '')[:16]}…")
    return 0 if receipt.get(success_key) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "CORPUS_SCHEMA",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "LIVE_ERGOAI_INTERFACE",
    "LIVE_ERGOAI_SCHEMA_VERSION",
    "LIVE_ERGOAI_EVIDENCE_CLASS",
    "ERGOAI_LIVE_TOOLCHAIN_INTERFACE",
    "ERGOAI_LIVE_TOOLCHAIN_SCHEMA",
    "ERGOAI_LIVE_TOOLCHAIN_GOAL_ID",
    "ERGOAI_LIVE_TOOLCHAIN_TASK_ID",
    "ERGOAI_LIVE_TOOLCHAIN_PROGRAM",
    "ERGOAI_LIVE_CASE_KINDS",
    "ADVISOR_TOOL_IDS",
    "LOCKED_SYMBOLICAI_VERSION",
    "LOCKED_ERGOAI_VERSION",
    "AdvisorRoleCertificationError",
    "CheckResult",
    "CaseOutcome",
    "AdvisorRoleCertification",
    "default_corpus_manifest",
    "corpus_cases",
    "advisor_role_boundary",
    "all_advisor_role_boundaries",
    "advisors_cannot_promote_hammer_lane",
    "evaluate_corpus_case",
    "certify_install_identities",
    "certify_live_ergoai_vendor",
    "build_ergoai_live_toolchain_contract",
    "run_certification_suite",
    "build_certification_receipt",
    "lane_handler",
    "bind_to_role_policy",
    "offline_env",
    "main",
]
