"""Evidence-checked documentation claims (FACP-057).

Controlled claim parser, ClaimIR requirement mapping, and narrowing renderer
for release-qualification documentation statements. Unsupported strong claims
fail closed or render a narrower evidence-qualified statement. Every accepted
or narrowed claim links current exact evidence and freshness. Human and
heuristic conclusions remain labeled and are never rewritten as proof.

Prohibited (by design):
- auto-upgrading prose into stronger claims
- treating Markdown, git history, or free prose as evidence
- rewriting subjective human conclusions as machine proof
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Optional, Sequence, Union

SCHEMA: Final[str] = "facp/docs-claims@1"
EVIDENCE_SCHEMA: Final[str] = "facp/docs-claims@1"
VOCAB_SCHEMA: Final[str] = "facp/formal-claim-algebra-v1@1"
TASK_ID: Final[str] = "FACP-057"
GOAL_ID: Final[str] = "FACP-G810"
BUNDLE: Final[str] = "facp/release/documentation"
MODULE_VERSION: Final[str] = "evidence-checked-documentation/v1"

# Normative strong documentation claim tokens (FACP-057 evidence subset).
STRONG_CLAIM_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "supports",
        "production-ready",
        "formally verified",
        "live",
        "current",
        "complete",
        "authenticated",
        "content-addressed",
        "filing-ready",
        "zero-knowledge",
        "cryptographically proven",
    }
)

# Evidence kinds that must never satisfy a documentation claim.
FORBIDDEN_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "markdown",
        "history",
        "git_history",
        "prose",
        "readme",
        "documentation",
        "free_text",
        "narrative",
    }
)

_TOKEN_ALIASES: Final[Mapping[str, str]] = {
    "support": "supports",
    "supported": "supports",
    "production_ready": "production-ready",
    "production ready": "production-ready",
    "formally_verified": "formally verified",
    "formally-verified": "formally verified",
    "cryptographically_proven": "cryptographically proven",
    "cryptographically-proven": "cryptographically proven",
    "content_addressed": "content-addressed",
    "content addressed": "content-addressed",
    "filing_ready": "filing-ready",
    "filing ready": "filing-ready",
    "zero_knowledge": "zero-knowledge",
    "zero knowledge": "zero-knowledge",
    "zk": "zero-knowledge",
}

_HUMAN_HEURISTIC_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "human",
        "heuristic",
        "subjective",
        "opinion",
        "judgment",
        "judgement",
        "human_conclusion",
        "heuristic_conclusion",
        "human-reviewed",
        "human_reviewed",
        "operator_judgment",
        "manual_review",
    }
)

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b("
    + "|".join(
        re.escape(token)
        for token in sorted(STRONG_CLAIM_TOKENS, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)


class DocsClaimsError(ValueError):
    """Fail-closed rejection for malformed documentation-claim input."""

    def __init__(self, message: str, *, code: str = "docs_claims_error") -> None:
        self.code = str(code)
        super().__init__(message)


class DocumentationClaimKind(str, Enum):
    STRONG = "strong"
    QUALIFIED = "qualified"
    HUMAN_HEURISTIC = "human_heuristic"
    NOT_A_CLAIM = "not_a_claim"


class ClaimCheckDisposition(str, Enum):
    ACCEPTED = "accepted"
    NARROWED = "narrowed"
    REJECTED = "rejected"
    HUMAN_LABELED = "human_labeled"


class ClaimMode(str, Enum):
    """How unsupported strong claims are handled."""

    FAIL = "fail"
    NARROW = "narrow"


@dataclass(frozen=True)
class ClaimIRRequirement:
    """Machine requirements that must hold for one strong claim token."""

    token: str
    required_predicates: tuple[str, ...]
    required_dimensions: Mapping[str, tuple[str, ...]]
    narrower_statement_template: str
    evidence_kinds_allowed: tuple[str, ...] = (
        "receipt",
        "digest",
        "proof",
        "attestation",
        "capability",
        "lock",
        "artifact",
        "envelope",
    )
    requires_current_freshness: bool = True
    release_conjuncts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "required_predicates": list(self.required_predicates),
            "required_dimensions": {
                key: list(values) for key, values in self.required_dimensions.items()
            },
            "narrower_statement_template": self.narrower_statement_template,
            "evidence_kinds_allowed": list(self.evidence_kinds_allowed),
            "requires_current_freshness": self.requires_current_freshness,
            "release_conjuncts": list(self.release_conjuncts),
        }


# ClaimIR: each strong token maps onto FCA dimensions / release conjuncts.
CLAIM_IR_REQUIREMENTS: Final[Mapping[str, ClaimIRRequirement]] = {
    "supports": ClaimIRRequirement(
        token="supports",
        required_predicates=("effect.observed", "freshness.current"),
        required_dimensions={
            "effect": ("observed",),
            "freshness": ("current",),
        },
        narrower_statement_template=(
            "Observed under declared evidence {evidence_id} "
            "(freshness={freshness}); not an unqualified support claim."
        ),
        release_conjuncts=("current_proofs_and_tests",),
    ),
    "production-ready": ClaimIRRequirement(
        token="production-ready",
        required_predicates=(
            "production_supported",
            "environment.live",
            "origin.live_observed",
            "freshness.current",
        ),
        required_dimensions={
            "environment": ("live",),
            "origin": ("live_observed",),
            "freshness": ("current",),
            "policy": ("allowed", "allowed_with_obligations"),
        },
        narrower_statement_template=(
            "Hermetic/candidate readiness only under evidence {evidence_id} "
            "(freshness={freshness}); not production-ready."
        ),
        release_conjuncts=("live_capabilities", "rights_resolution"),
    ),
    "formally verified": ClaimIRRequirement(
        token="formally verified",
        required_predicates=("proof.verified", "freshness.current"),
        required_dimensions={
            "proof": ("verified",),
            "freshness": ("current",),
        },
        narrower_statement_template=(
            "Proof status is {proof} under evidence {evidence_id} "
            "(freshness={freshness}); not formally verified."
        ),
        evidence_kinds_allowed=("proof", "attestation", "receipt", "envelope"),
        release_conjuncts=("current_proofs_and_tests",),
    ),
    "live": ClaimIRRequirement(
        token="live",
        required_predicates=(
            "environment.live",
            "origin.live_observed",
            "freshness.current",
        ),
        required_dimensions={
            "environment": ("live",),
            "origin": ("live_observed",),
            "freshness": ("current",),
        },
        narrower_statement_template=(
            "Environment/origin are {environment}/{origin} under evidence "
            "{evidence_id} (freshness={freshness}); not live-qualified."
        ),
        release_conjuncts=("live_capabilities",),
    ),
    "current": ClaimIRRequirement(
        token="current",
        required_predicates=("freshness.current",),
        required_dimensions={"freshness": ("current",)},
        narrower_statement_template=(
            "Evidence {evidence_id} freshness is {freshness}; not current."
        ),
        release_conjuncts=("current_proofs_and_tests",),
    ),
    "complete": ClaimIRRequirement(
        token="complete",
        required_predicates=("effect.observed", "freshness.current"),
        required_dimensions={
            "effect": ("observed",),
            "freshness": ("current",),
            "integrity": ("digest_valid", "signature_valid", "structurally_valid"),
        },
        narrower_statement_template=(
            "Partial/incomplete under evidence {evidence_id} "
            "(freshness={freshness}); not complete."
        ),
    ),
    "authenticated": ClaimIRRequirement(
        token="authenticated",
        required_predicates=("integrity.signature_valid", "freshness.current"),
        required_dimensions={
            "integrity": ("signature_valid",),
            "authority": ("valid",),
            "freshness": ("current",),
        },
        narrower_statement_template=(
            "Integrity/authority are {integrity}/{authority} under evidence "
            "{evidence_id} (freshness={freshness}); not authenticated."
        ),
        evidence_kinds_allowed=("attestation", "receipt", "artifact", "envelope"),
    ),
    "content-addressed": ClaimIRRequirement(
        token="content-addressed",
        required_predicates=("integrity.digest_valid",),
        required_dimensions={
            "integrity": ("digest_valid", "signature_valid"),
        },
        narrower_statement_template=(
            "Integrity is {integrity} under evidence {evidence_id}; "
            "not content-addressed."
        ),
        requires_current_freshness=False,
        evidence_kinds_allowed=("digest", "artifact", "lock", "receipt", "envelope"),
        release_conjuncts=("immutable_dependency_closure", "exact_source_binding"),
    ),
    "filing-ready": ClaimIRRequirement(
        token="filing-ready",
        required_predicates=(
            "integrity.digest_valid",
            "freshness.current",
            "review.human_reviewed",
        ),
        required_dimensions={
            "integrity": ("digest_valid", "signature_valid"),
            "freshness": ("current",),
            "review": ("human_reviewed",),
        },
        narrower_statement_template=(
            "Draft/not filing-ready under evidence {evidence_id} "
            "(freshness={freshness}, review={review})."
        ),
        release_conjuncts=("rights_resolution",),
    ),
    "zero-knowledge": ClaimIRRequirement(
        token="zero-knowledge",
        required_predicates=("proof.verified", "freshness.current"),
        required_dimensions={
            "proof": ("verified",),
            "freshness": ("current",),
        },
        narrower_statement_template=(
            "No verified zero-knowledge attestation under evidence "
            "{evidence_id} (proof={proof}, freshness={freshness})."
        ),
        evidence_kinds_allowed=("attestation", "proof", "receipt", "envelope"),
    ),
    "cryptographically proven": ClaimIRRequirement(
        token="cryptographically proven",
        required_predicates=(
            "proof.verified",
            "integrity.signature_valid",
            "freshness.current",
        ),
        required_dimensions={
            "proof": ("verified",),
            "integrity": ("signature_valid",),
            "freshness": ("current",),
        },
        narrower_statement_template=(
            "Cryptographic proof status is {proof}/{integrity} under evidence "
            "{evidence_id} (freshness={freshness}); not cryptographically proven."
        ),
        evidence_kinds_allowed=("attestation", "proof", "receipt", "envelope"),
    ),
}


def normalize_claim_token(value: Any) -> str:
    """Return the canonical strong-claim token, or ``\"\"`` when unrecognized."""

    if not isinstance(value, str):
        return ""
    raw = value.strip().casefold()
    if not raw:
        return ""
    if raw in STRONG_CLAIM_TOKENS:
        return raw
    aliased = _TOKEN_ALIASES.get(raw)
    if aliased is not None:
        return aliased
    collapsed = re.sub(r"[\s_]+", "-", raw)
    if collapsed in STRONG_CLAIM_TOKENS:
        return collapsed
    spaced = collapsed.replace("-", " ")
    if spaced in STRONG_CLAIM_TOKENS:
        return spaced
    return _TOKEN_ALIASES.get(spaced, "")


def requirements_for_token(token: Any) -> ClaimIRRequirement:
    """Return ClaimIR requirements for one strong claim token."""

    canonical = normalize_claim_token(token)
    if not canonical or canonical not in CLAIM_IR_REQUIREMENTS:
        raise DocsClaimsError(
            f"unknown strong claim token: {token!r}",
            code="unknown_claim_token",
        )
    return CLAIM_IR_REQUIREMENTS[canonical]


@dataclass(frozen=True)
class ExactEvidenceRef:
    """Exact, non-prose evidence binding for one documentation claim."""

    evidence_id: str
    kind: str
    digest: str = ""
    artifact_path: str = ""
    tree_id: str = ""
    observed_at: str = ""
    freshness: str = "stale"
    envelope: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", str(self.evidence_id or "").strip())
        object.__setattr__(self, "kind", str(self.kind or "").strip().casefold())
        object.__setattr__(self, "digest", str(self.digest or "").strip())
        object.__setattr__(
            self, "artifact_path", str(self.artifact_path or "").strip()
        )
        object.__setattr__(self, "tree_id", str(self.tree_id or "").strip())
        object.__setattr__(self, "observed_at", str(self.observed_at or "").strip())
        object.__setattr__(
            self, "freshness", str(self.freshness or "stale").strip().casefold()
        )
        object.__setattr__(self, "envelope", dict(self.envelope or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if not self.evidence_id:
            raise DocsClaimsError(
                "exact evidence requires evidence_id",
                code="missing_evidence_id",
            )
        if not self.kind:
            raise DocsClaimsError(
                "exact evidence requires kind",
                code="missing_evidence_kind",
            )

    @property
    def is_forbidden_kind(self) -> bool:
        return self.kind in FORBIDDEN_EVIDENCE_KINDS

    @property
    def is_current(self) -> bool:
        return self.freshness == "current"

    def dimension(self, name: str) -> str:
        if name == "freshness" and "freshness" not in self.envelope:
            return self.freshness
        value = self.envelope.get(name, "")
        return str(value).strip().casefold()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "digest": self.digest,
            "artifact_path": self.artifact_path,
            "tree_id": self.tree_id,
            "observed_at": self.observed_at,
            "freshness": self.freshness,
            "envelope": dict(self.envelope),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ClaimIR:
    """Intermediate representation for one controlled documentation claim."""

    claim_id: str
    token: str
    raw_text: str
    kind: DocumentationClaimKind
    requirements: Optional[ClaimIRRequirement]
    evidence_refs: tuple[ExactEvidenceRef, ...] = ()
    source_path: str = ""
    source_line: int = 0
    labels: tuple[str, ...] = ()
    subject: str = ""
    mode: ClaimMode = ClaimMode.NARROW

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "token": self.token,
            "raw_text": self.raw_text,
            "kind": self.kind.value,
            "requirements": (
                None if self.requirements is None else self.requirements.to_dict()
            ),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "source_path": self.source_path,
            "source_line": self.source_line,
            "labels": list(self.labels),
            "subject": self.subject,
            "mode": self.mode.value,
        }


@dataclass(frozen=True)
class NarrowedStatement:
    """Rendered documentation statement after evidence checking."""

    claim_id: str
    disposition: ClaimCheckDisposition
    original_text: str
    rendered_text: str
    token: str
    evidence_links: tuple[dict[str, str], ...]
    freshness: str
    labels: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    satisfied_predicates: tuple[str, ...] = ()
    missing_predicates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "disposition": self.disposition.value,
            "original_text": self.original_text,
            "rendered_text": self.rendered_text,
            "token": self.token,
            "evidence_links": [dict(item) for item in self.evidence_links],
            "freshness": self.freshness,
            "labels": list(self.labels),
            "reason_codes": list(self.reason_codes),
            "satisfied_predicates": list(self.satisfied_predicates),
            "missing_predicates": list(self.missing_predicates),
        }


@dataclass(frozen=True)
class DocsClaimsReport:
    """Deterministic report for a documentation-claims check."""

    schema: str = SCHEMA
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    bundle: str = BUNDLE
    vocab_schema: str = VOCAB_SCHEMA
    module_version: str = MODULE_VERSION
    evidence_subset: tuple[str, ...] = tuple(sorted(STRONG_CLAIM_TOKENS))
    claims: tuple[ClaimIR, ...] = ()
    results: tuple[NarrowedStatement, ...] = ()

    @property
    def accepted(self) -> tuple[NarrowedStatement, ...]:
        return tuple(
            item
            for item in self.results
            if item.disposition is ClaimCheckDisposition.ACCEPTED
        )

    @property
    def narrowed(self) -> tuple[NarrowedStatement, ...]:
        return tuple(
            item
            for item in self.results
            if item.disposition is ClaimCheckDisposition.NARROWED
        )

    @property
    def rejected(self) -> tuple[NarrowedStatement, ...]:
        return tuple(
            item
            for item in self.results
            if item.disposition is ClaimCheckDisposition.REJECTED
        )

    @property
    def human_labeled(self) -> tuple[NarrowedStatement, ...]:
        return tuple(
            item
            for item in self.results
            if item.disposition is ClaimCheckDisposition.HUMAN_LABELED
        )

    @property
    def ok(self) -> bool:
        return not self.rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_schema": EVIDENCE_SCHEMA,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "bundle": self.bundle,
            "vocab_schema": self.vocab_schema,
            "module_version": self.module_version,
            "evidence_subset": list(self.evidence_subset),
            "ok": self.ok,
            "claims": [claim.to_dict() for claim in self.claims],
            "results": [result.to_dict() for result in self.results],
            "counts": {
                "claims": len(self.claims),
                "accepted": len(self.accepted),
                "narrowed": len(self.narrowed),
                "rejected": len(self.rejected),
                "human_labeled": len(self.human_labeled),
            },
        }


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise DocsClaimsError(
        f"expected mapping, got {type(value).__name__}",
        code="invalid_mapping",
    )


def _normalize_labels(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        items = [values]
    elif isinstance(values, Iterable):
        items = list(values)
    else:
        raise DocsClaimsError("labels must be a string or sequence", code="invalid_labels")
    out: list[str] = []
    for item in items:
        text = str(item or "").strip().casefold()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _is_human_heuristic(
    *,
    kind: Any = None,
    labels: Sequence[str] = (),
    conclusion_type: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    markers = set(labels)
    for raw in (kind, conclusion_type):
        if isinstance(raw, str) and raw.strip():
            markers.add(raw.strip().casefold())
    if metadata:
        for key in ("conclusion_type", "claim_kind", "authority"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                markers.add(value.strip().casefold())
        if metadata.get("human") is True or metadata.get("heuristic") is True:
            return True
    return bool(markers.intersection(_HUMAN_HEURISTIC_MARKERS))


def parse_exact_evidence_ref(value: Any) -> ExactEvidenceRef:
    """Parse one exact evidence reference; reject Markdown/history kinds."""

    payload = _as_mapping(value)
    envelope_raw = payload.get("envelope") or payload.get("dimensions") or {}
    envelope: dict[str, str] = {}
    if isinstance(envelope_raw, Mapping):
        for key, item in envelope_raw.items():
            if hasattr(item, "value"):
                envelope[str(key)] = str(item.value)
            else:
                envelope[str(key)] = str(item)
    freshness = payload.get("freshness")
    if freshness is None and "freshness" in envelope:
        freshness = envelope["freshness"]
    return ExactEvidenceRef(
        evidence_id=str(
            payload.get("evidence_id")
            or payload.get("id")
            or payload.get("artifact_id")
            or ""
        ),
        kind=str(payload.get("kind") or payload.get("evidence_kind") or ""),
        digest=str(payload.get("digest") or payload.get("content_digest") or ""),
        artifact_path=str(
            payload.get("artifact_path") or payload.get("path") or ""
        ),
        tree_id=str(payload.get("tree_id") or payload.get("repository_tree_id") or ""),
        observed_at=str(payload.get("observed_at") or payload.get("timestamp") or ""),
        freshness=str(freshness or "stale"),
        envelope=envelope,
        metadata=_as_mapping(payload.get("metadata")),
    )


def _detect_token_from_text(text: str) -> str:
    match = _TOKEN_PATTERN.search(text.casefold())
    if not match:
        return ""
    return normalize_claim_token(match.group(1))


def parse_controlled_claim(
    record: Any,
    *,
    default_mode: ClaimMode = ClaimMode.NARROW,
) -> ClaimIR:
    """Parse one controlled claim record into ClaimIR.

    Accepts structured records only. Free Markdown is not evidence and is not
    auto-upgraded into a stronger claim.
    """

    payload = _as_mapping(record)
    claim_id = str(payload.get("claim_id") or payload.get("id") or "").strip()
    if not claim_id:
        raise DocsClaimsError("claim_id is required", code="missing_claim_id")

    raw_text = str(payload.get("raw_text") or payload.get("text") or "").strip()
    subject = str(payload.get("subject") or "").strip()
    labels = _normalize_labels(payload.get("labels"))
    mode_raw = payload.get("mode", default_mode)
    if isinstance(mode_raw, ClaimMode):
        mode = mode_raw
    else:
        mode = ClaimMode(str(mode_raw).strip().casefold())

    token = normalize_claim_token(payload.get("token") or payload.get("claim"))
    if not token and raw_text:
        token = _detect_token_from_text(raw_text)
    if not raw_text:
        raw_text = token

    evidence_values = payload.get("evidence") or payload.get("evidence_refs") or ()
    if isinstance(evidence_values, Mapping):
        evidence_values = (evidence_values,)
    evidence_refs = tuple(parse_exact_evidence_ref(item) for item in evidence_values)

    human = _is_human_heuristic(
        kind=payload.get("kind") or payload.get("claim_kind"),
        labels=labels,
        conclusion_type=payload.get("conclusion_type"),
        metadata=_as_mapping(payload.get("metadata")),
    )
    if human:
        if "human_conclusion" not in labels and "heuristic" not in labels:
            labels = labels + ("human_conclusion",)
        return ClaimIR(
            claim_id=claim_id,
            token=token,
            raw_text=raw_text or "human/heuristic conclusion",
            kind=DocumentationClaimKind.HUMAN_HEURISTIC,
            requirements=None,
            evidence_refs=evidence_refs,
            source_path=str(payload.get("source_path") or ""),
            source_line=int(payload.get("source_line") or 0),
            labels=labels,
            subject=subject,
            mode=mode,
        )

    if not token:
        return ClaimIR(
            claim_id=claim_id,
            token="",
            raw_text=raw_text,
            kind=DocumentationClaimKind.NOT_A_CLAIM,
            requirements=None,
            evidence_refs=evidence_refs,
            source_path=str(payload.get("source_path") or ""),
            source_line=int(payload.get("source_line") or 0),
            labels=labels,
            subject=subject,
            mode=mode,
        )

    requirements = requirements_for_token(token)
    kind = DocumentationClaimKind.STRONG
    if payload.get("qualified") is True or "qualified" in labels:
        kind = DocumentationClaimKind.QUALIFIED
    return ClaimIR(
        claim_id=claim_id,
        token=token,
        raw_text=raw_text,
        kind=kind,
        requirements=requirements,
        evidence_refs=evidence_refs,
        source_path=str(payload.get("source_path") or ""),
        source_line=int(payload.get("source_line") or 0),
        labels=labels,
        subject=subject,
        mode=mode,
    )


def parse_controlled_claims(
    records: Iterable[Any],
    *,
    default_mode: ClaimMode = ClaimMode.NARROW,
) -> tuple[ClaimIR, ...]:
    """Parse a sequence of controlled claim records."""

    return tuple(
        parse_controlled_claim(record, default_mode=default_mode) for record in records
    )


def _predicate_holds(predicate: str, evidence: ExactEvidenceRef) -> bool:
    if "." not in predicate:
        # Bare alias predicates derived from conjuncts / typed spellings.
        if predicate == "production_supported":
            return (
                evidence.dimension("environment") == "live"
                and evidence.dimension("origin") == "live_observed"
                and evidence.is_current
                and evidence.dimension("policy")
                in {"allowed", "allowed_with_obligations"}
            )
        return False
    dimension, expected = predicate.split(".", 1)
    actual = evidence.dimension(dimension)
    return actual == expected.casefold()


def _evaluate_against_evidence(
    claim: ClaimIR,
    evidence: ExactEvidenceRef,
) -> tuple[bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (ok, satisfied, missing, reason_codes)."""

    assert claim.requirements is not None
    req = claim.requirements
    reasons: list[str] = []
    if evidence.is_forbidden_kind:
        return False, (), ("forbidden_evidence_kind",), ("forbidden_evidence_kind",)
    if evidence.kind not in req.evidence_kinds_allowed:
        reasons.append("evidence_kind_not_allowed")
    if req.requires_current_freshness and not evidence.is_current:
        reasons.append("stale_evidence")

    satisfied: list[str] = []
    missing: list[str] = []
    for predicate in req.required_predicates:
        if _predicate_holds(predicate, evidence):
            satisfied.append(predicate)
        else:
            missing.append(predicate)

    for dimension, allowed in req.required_dimensions.items():
        actual = evidence.dimension(dimension)
        if actual not in {item.casefold() for item in allowed}:
            # Dimensions already covered by required_predicates still contribute
            # a typed reason when absent or mismatched.
            code = f"dimension_mismatch:{dimension}"
            if code not in reasons:
                reasons.append(code)

    # FACP-055 non-implication: simulation/hermetic cannot satisfy live.
    if claim.token in {"live", "production-ready"}:
        if evidence.dimension("origin") in {"simulated", "fixture", "declared", "absent"}:
            reasons.append("simulation_as_live")
        if evidence.dimension("environment") in {"hermetic", "conditional"}:
            reasons.append("hermetic_not_live")

    ok = not missing and not reasons
    return ok, tuple(satisfied), tuple(missing), tuple(reasons)


def _evidence_links(refs: Sequence[ExactEvidenceRef]) -> tuple[dict[str, str], ...]:
    links: list[dict[str, str]] = []
    for ref in refs:
        link = {
            "evidence_id": ref.evidence_id,
            "kind": ref.kind,
            "freshness": ref.freshness,
        }
        if ref.digest:
            link["digest"] = ref.digest
        if ref.artifact_path:
            link["artifact_path"] = ref.artifact_path
        if ref.tree_id:
            link["tree_id"] = ref.tree_id
        links.append(link)
    return tuple(links)


def _format_narrower(
    template: str,
    *,
    evidence: Optional[ExactEvidenceRef],
    claim: ClaimIR,
) -> str:
    values = {
        "evidence_id": evidence.evidence_id if evidence else "none",
        "freshness": evidence.freshness if evidence else "absent",
        "proof": evidence.dimension("proof") if evidence else "none",
        "integrity": evidence.dimension("integrity") if evidence else "unchecked",
        "authority": evidence.dimension("authority") if evidence else "unchecked",
        "environment": evidence.dimension("environment") if evidence else "hermetic",
        "origin": evidence.dimension("origin") if evidence else "absent",
        "review": evidence.dimension("review") if evidence else "unreviewed",
        "subject": claim.subject or "subject",
        "token": claim.token,
    }
    try:
        body = template.format(**values)
    except (KeyError, ValueError):
        body = template
    if claim.subject:
        return f"{claim.subject}: {body}"
    return body


def render_narrowed_statement(
    claim: ClaimIR,
    *,
    disposition: ClaimCheckDisposition,
    evidence: Optional[ExactEvidenceRef] = None,
    reason_codes: Sequence[str] = (),
    satisfied_predicates: Sequence[str] = (),
    missing_predicates: Sequence[str] = (),
    evidence_refs: Sequence[ExactEvidenceRef] = (),
) -> NarrowedStatement:
    """Render an evidence-qualified statement. Never strengthens the claim."""

    refs = tuple(evidence_refs) or ((evidence,) if evidence is not None else ())
    freshness = evidence.freshness if evidence is not None else (
        refs[0].freshness if refs else "absent"
    )
    links = _evidence_links(refs)

    if disposition is ClaimCheckDisposition.HUMAN_LABELED:
        labels = tuple(
            dict.fromkeys(
                (
                    *claim.labels,
                    "human_conclusion",
                    *(
                        ("heuristic",)
                        if "heuristic" in claim.labels
                        or "heuristic_conclusion" in claim.labels
                        else ()
                    ),
                )
            )
        )
        rendered = (
            f"[human/heuristic] {claim.raw_text}"
            if not claim.raw_text.lower().startswith("[human")
            else claim.raw_text
        )
        return NarrowedStatement(
            claim_id=claim.claim_id,
            disposition=disposition,
            original_text=claim.raw_text,
            rendered_text=rendered,
            token=claim.token,
            evidence_links=links,
            freshness=freshness,
            labels=labels,
            reason_codes=tuple(reason_codes) or ("human_or_heuristic_conclusion",),
            satisfied_predicates=tuple(satisfied_predicates),
            missing_predicates=tuple(missing_predicates),
        )

    if disposition is ClaimCheckDisposition.ACCEPTED:
        if not links:
            raise DocsClaimsError(
                "accepted claims must link exact evidence",
                code="accepted_without_evidence",
            )
        if freshness != "current" and (
            claim.requirements is None or claim.requirements.requires_current_freshness
        ):
            raise DocsClaimsError(
                "accepted claims require current freshness",
                code="accepted_without_current_freshness",
            )
        subject = f"{claim.subject}: " if claim.subject else ""
        evidence_id = links[0]["evidence_id"]
        rendered = (
            f"{subject}{claim.token} under exact evidence {evidence_id} "
            f"(freshness={freshness})."
        )
        return NarrowedStatement(
            claim_id=claim.claim_id,
            disposition=disposition,
            original_text=claim.raw_text,
            rendered_text=rendered,
            token=claim.token,
            evidence_links=links,
            freshness=freshness,
            labels=claim.labels,
            reason_codes=tuple(reason_codes),
            satisfied_predicates=tuple(satisfied_predicates),
            missing_predicates=tuple(missing_predicates),
        )

    if disposition is ClaimCheckDisposition.NARROWED:
        template = (
            claim.requirements.narrower_statement_template
            if claim.requirements is not None
            else (
                "Evidence-qualified only under {evidence_id} "
                "(freshness={freshness}); strong claim withdrawn."
            )
        )
        rendered = _format_narrower(
            template,
            evidence=evidence or (refs[0] if refs else None),
            claim=claim,
        )
        # Narrowed text must stay evidence-qualified and must not read as an
        # unqualified strong assertion.
        if "evidence" not in rendered.casefold():
            rendered = f"{rendered} [evidence-qualified]"
        return NarrowedStatement(
            claim_id=claim.claim_id,
            disposition=disposition,
            original_text=claim.raw_text,
            rendered_text=rendered,
            token=claim.token,
            evidence_links=links,
            freshness=freshness,
            labels=claim.labels,
            reason_codes=tuple(reason_codes) or ("unsupported_strong_claim",),
            satisfied_predicates=tuple(satisfied_predicates),
            missing_predicates=tuple(missing_predicates),
        )

    # REJECTED
    reasons = tuple(reason_codes) or ("unsupported_strong_claim",)
    rendered = (
        f"REJECTED strong claim {claim.token!r}: "
        + ", ".join(reasons)
        + (
            f" (evidence={links[0]['evidence_id']}, freshness={freshness})"
            if links
            else " (no exact current evidence)"
        )
    )
    return NarrowedStatement(
        claim_id=claim.claim_id,
        disposition=ClaimCheckDisposition.REJECTED,
        original_text=claim.raw_text,
        rendered_text=rendered,
        token=claim.token,
        evidence_links=links,
        freshness=freshness,
        labels=claim.labels,
        reason_codes=reasons,
        satisfied_predicates=tuple(satisfied_predicates),
        missing_predicates=tuple(missing_predicates),
    )


def evaluate_claim(claim: ClaimIR) -> NarrowedStatement:
    """Evaluate one ClaimIR against its linked exact evidence."""

    if claim.kind is DocumentationClaimKind.HUMAN_HEURISTIC:
        return render_narrowed_statement(
            claim,
            disposition=ClaimCheckDisposition.HUMAN_LABELED,
            evidence_refs=claim.evidence_refs,
        )

    if claim.kind is DocumentationClaimKind.NOT_A_CLAIM:
        return NarrowedStatement(
            claim_id=claim.claim_id,
            disposition=ClaimCheckDisposition.ACCEPTED,
            original_text=claim.raw_text,
            rendered_text=claim.raw_text or "[not a strong claim]",
            token="",
            evidence_links=(),
            freshness="n/a",
            labels=claim.labels,
            reason_codes=("not_a_claim",),
        )

    if claim.requirements is None:
        return render_narrowed_statement(
            claim,
            disposition=ClaimCheckDisposition.REJECTED,
            reason_codes=("missing_claim_ir_requirements",),
        )

    if not claim.evidence_refs:
        if claim.mode is ClaimMode.FAIL:
            return render_narrowed_statement(
                claim,
                disposition=ClaimCheckDisposition.REJECTED,
                reason_codes=("missing_exact_evidence",),
            )
        return render_narrowed_statement(
            claim,
            disposition=ClaimCheckDisposition.NARROWED,
            reason_codes=("missing_exact_evidence",),
            missing_predicates=claim.requirements.required_predicates,
        )

    # Prefer the first evidence ref that fully satisfies requirements.
    best: Optional[tuple[ExactEvidenceRef, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = None
    for ref in claim.evidence_refs:
        ok, satisfied, missing, reasons = _evaluate_against_evidence(claim, ref)
        if ok:
            return render_narrowed_statement(
                claim,
                disposition=ClaimCheckDisposition.ACCEPTED,
                evidence=ref,
                evidence_refs=claim.evidence_refs,
                satisfied_predicates=satisfied,
                missing_predicates=missing,
                reason_codes=(),
            )
        if best is None or len(satisfied) > len(best[1]):
            best = (ref, satisfied, missing, reasons)

    assert best is not None
    ref, satisfied, missing, reasons = best
    if claim.mode is ClaimMode.FAIL:
        return render_narrowed_statement(
            claim,
            disposition=ClaimCheckDisposition.REJECTED,
            evidence=ref,
            evidence_refs=claim.evidence_refs,
            satisfied_predicates=satisfied,
            missing_predicates=missing,
            reason_codes=reasons or ("unsupported_strong_claim",),
        )
    return render_narrowed_statement(
        claim,
        disposition=ClaimCheckDisposition.NARROWED,
        evidence=ref,
        evidence_refs=claim.evidence_refs,
        satisfied_predicates=satisfied,
        missing_predicates=missing,
        reason_codes=reasons or ("unsupported_strong_claim",),
    )


def check_documentation_claims(
    records: Iterable[Any],
    *,
    default_mode: ClaimMode = ClaimMode.NARROW,
) -> DocsClaimsReport:
    """Parse and evaluate controlled documentation claims."""

    claims = parse_controlled_claims(records, default_mode=default_mode)
    results = tuple(evaluate_claim(claim) for claim in claims)
    return DocsClaimsReport(claims=claims, results=results)


def claim_ir_coverage() -> dict[str, ClaimIRRequirement]:
    """Return the full ClaimIR mapping; used by tests and fixtures."""

    missing = STRONG_CLAIM_TOKENS.difference(CLAIM_IR_REQUIREMENTS)
    if missing:
        raise DocsClaimsError(
            f"ClaimIR missing tokens: {sorted(missing)}",
            code="incomplete_claim_ir",
        )
    return dict(CLAIM_IR_REQUIREMENTS)


def documentation_claim_fixtures() -> tuple[dict[str, Any], ...]:
    """Compact in-module fixtures covering acceptance paths."""

    current_live = {
        "origin": "live_observed",
        "integrity": "signature_valid",
        "authority": "valid",
        "policy": "allowed",
        "proof": "verified",
        "freshness": "current",
        "effect": "observed",
        "environment": "live",
        "review": "human_reviewed",
    }
    hermetic_candidate = {
        "origin": "hermetic_observed",
        "integrity": "structurally_valid",
        "authority": "unchecked",
        "policy": "unchecked",
        "proof": "candidate",
        "freshness": "stale",
        "effect": "started",
        "environment": "hermetic",
        "review": "unreviewed",
    }
    return (
        {
            "claim_id": "fixture:formally-verified-supported",
            "token": "formally verified",
            "raw_text": "The gate is formally verified.",
            "subject": "release gate",
            "evidence": [
                {
                    "evidence_id": "evidence:proof-receipt-1",
                    "kind": "proof",
                    "digest": "sha256:abc",
                    "freshness": "current",
                    "envelope": current_live,
                }
            ],
        },
        {
            "claim_id": "fixture:formally-verified-unsupported",
            "token": "formally verified",
            "raw_text": "The gate is formally verified.",
            "mode": "narrow",
            "evidence": [
                {
                    "evidence_id": "evidence:proof-candidate-1",
                    "kind": "proof",
                    "digest": "sha256:def",
                    "freshness": "stale",
                    "envelope": hermetic_candidate,
                }
            ],
        },
        {
            "claim_id": "fixture:live-simulation",
            "token": "live",
            "raw_text": "Backend is live.",
            "mode": "narrow",
            "evidence": [
                {
                    "evidence_id": "evidence:sim-1",
                    "kind": "capability",
                    "freshness": "current",
                    "envelope": {
                        **hermetic_candidate,
                        "origin": "simulated",
                        "environment": "hermetic",
                        "freshness": "current",
                    },
                }
            ],
        },
        {
            "claim_id": "fixture:missing-evidence-fail",
            "token": "production-ready",
            "raw_text": "Service is production-ready.",
            "mode": "fail",
        },
        {
            "claim_id": "fixture:human-heuristic",
            "token": "supports",
            "raw_text": "Operators believe this supports the release.",
            "kind": "human_heuristic",
            "labels": ["heuristic"],
            "conclusion_type": "human",
        },
        {
            "claim_id": "fixture:markdown-evidence-rejected",
            "token": "current",
            "raw_text": "Status is current.",
            "mode": "fail",
            "evidence": [
                {
                    "evidence_id": "evidence:readme",
                    "kind": "markdown",
                    "freshness": "current",
                }
            ],
        },
        {
            "claim_id": "fixture:content-addressed",
            "token": "content-addressed",
            "raw_text": "Artifact is content-addressed.",
            "evidence": [
                {
                    "evidence_id": "evidence:digest-1",
                    "kind": "digest",
                    "digest": "sha256:fff",
                    "freshness": "stale",
                    "envelope": {
                        "integrity": "digest_valid",
                        "freshness": "stale",
                    },
                }
            ],
        },
    )


__all__ = [
    "BUNDLE",
    "CLAIM_IR_REQUIREMENTS",
    "ClaimCheckDisposition",
    "ClaimIR",
    "ClaimIRRequirement",
    "ClaimMode",
    "DocsClaimsError",
    "DocsClaimsReport",
    "DocumentationClaimKind",
    "EVIDENCE_SCHEMA",
    "ExactEvidenceRef",
    "FORBIDDEN_EVIDENCE_KINDS",
    "GOAL_ID",
    "MODULE_VERSION",
    "NarrowedStatement",
    "SCHEMA",
    "STRONG_CLAIM_TOKENS",
    "TASK_ID",
    "VOCAB_SCHEMA",
    "check_documentation_claims",
    "claim_ir_coverage",
    "documentation_claim_fixtures",
    "evaluate_claim",
    "normalize_claim_token",
    "parse_controlled_claim",
    "parse_controlled_claims",
    "parse_exact_evidence_ref",
    "render_narrowed_statement",
    "requirements_for_token",
]
