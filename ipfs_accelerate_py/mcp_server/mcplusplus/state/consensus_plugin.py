"""ConsensusPlugin@1 — honest guarantee labels and plugin evidence for consensus mode.

MCP++ 1.0 does not treat "consensus" as a single boolean. Plugins and Profile G
neighborhood coordination MUST declare exactly one of four guarantee labels
(ADR-0004 §4 / plan KD-11):

* ``coordination`` — best-effort ordering/scheduling; no crash or Byzantine
  safety claim.
* ``majority_approval`` — threshold approval among a declared peer set under an
  honest-majority assumption for that set only.
* ``crash_consensus`` — classic CFT (crash/recover) agreement under a declared
  membership.
* ``bft`` — Byzantine fault tolerance under a declared fault bound — only when a
  real BFT engine is present and tested.

Profile G neighborhood agreement is **coordination and/or majority_approval
only**. Labeling a neighborhood result as ``bft`` (or ``crash_consensus``) is a
fail-closed error. This module supplies the plugin contract, the evidence
format, Profile G wiring, and a deterministic test adapter that never claims
BFT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

CONSENSUS_PLUGIN_INTERFACE = "ConsensusPlugin@1"
CONSENSUS_EVIDENCE_SCHEMA = "mcp++/state/consensus-evidence@1"
CONSENSUS_RESULT_SCHEMA = "mcp++/state/consensus-result@1"
CONSENSUS_MODE = "consensus"

# Profile G neighborhood plugin id used when wiring G records into evidence.
PROFILE_G_NEIGHBORHOOD_PLUGIN_ID = "mcp++/consensus/profile-g-neighborhood@1"
DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID = "mcp++/consensus/deterministic-test@1"

# ---------------------------------------------------------------------------
# Guarantee labels (closed set — ADR-0004 §4 / KD-11)
# ---------------------------------------------------------------------------


class GuaranteeLabel(str, Enum):
    """Honest consensus-class guarantee labels (wire form is the enum value)."""

    COORDINATION = "coordination"
    MAJORITY_APPROVAL = "majority_approval"
    CRASH_CONSENSUS = "crash_consensus"
    BFT = "bft"


GUARANTEE_COORDINATION = GuaranteeLabel.COORDINATION.value
GUARANTEE_MAJORITY_APPROVAL = GuaranteeLabel.MAJORITY_APPROVAL.value
GUARANTEE_CRASH_CONSENSUS = GuaranteeLabel.CRASH_CONSENSUS.value
GUARANTEE_BFT = GuaranteeLabel.BFT.value

GUARANTEE_LABELS: frozenset[str] = frozenset(label.value for label in GuaranteeLabel)

# Profile G neighborhood may only claim these two classes.
PROFILE_G_ALLOWED_GUARANTEES: frozenset[str] = frozenset(
    {
        GUARANTEE_COORDINATION,
        GUARANTEE_MAJORITY_APPROVAL,
    }
)

# Evidence kind discriminators.
EVIDENCE_KIND_NEIGHBORHOOD = "neighborhood"
EVIDENCE_KIND_PLUGIN = "plugin"
EVIDENCE_KIND_TEST = "test"

EVIDENCE_KINDS: frozenset[str] = frozenset(
    {
        EVIDENCE_KIND_NEIGHBORHOOD,
        EVIDENCE_KIND_PLUGIN,
        EVIDENCE_KIND_TEST,
    }
)

# Verdicts for neighborhood / majority ballots.
VERDICT_SUPPORT = "support"
VERDICT_CHALLENGE = "challenge"
VERDICT_ABSTAIN = "abstain"
ALLOWED_VERDICTS: frozenset[str] = frozenset(
    {VERDICT_SUPPORT, VERDICT_CHALLENGE, VERDICT_ABSTAIN}
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConsensusPluginError(Exception):
    """Base error for ConsensusPlugin@1 operations."""


class InvalidGuaranteeError(ConsensusPluginError):
    """Raised when a guarantee label is missing, unknown, or misapplied."""


class NeighborhoodGuaranteeError(InvalidGuaranteeError):
    """Raised when Profile G neighborhood is labeled beyond its real guarantee.

    Acceptance (MCPP-039): tests fail if a neighborhood result is labeled BFT.
    The same fail-closed rule applies to ``crash_consensus`` for neighborhood
    paths — neighborhood is coordination / majority approval only.
    """


class ConsensusEvidenceError(ConsensusPluginError):
    """Raised when plugin evidence is missing, malformed, or inconsistent."""


class ConsensusRejectedError(ConsensusPluginError):
    """Raised when evidence is well-formed but does not meet the threshold."""


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


def normalize_guarantee_label(value: object) -> str:
    """Return a canonical guarantee label string or raise :class:`InvalidGuaranteeError`."""

    if isinstance(value, GuaranteeLabel):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise InvalidGuaranteeError("guarantee label is required")
    label = value.strip()
    if label not in GUARANTEE_LABELS:
        raise InvalidGuaranteeError(
            f"guarantee label {label!r} is not one of {sorted(GUARANTEE_LABELS)}"
        )
    return label


def is_profile_g_allowed_guarantee(guarantee: object) -> bool:
    """Return True when ``guarantee`` is allowed for Profile G neighborhood."""

    try:
        return normalize_guarantee_label(guarantee) in PROFILE_G_ALLOWED_GUARANTEES
    except InvalidGuaranteeError:
        return False


def require_profile_g_guarantee(guarantee: object) -> str:
    """Validate a guarantee for Profile G neighborhood; fail closed on BFT.

    Profile G neighborhood agreement is coordination and/or majority_approval
    only (ADR-0004 §4; plan §11; REQ-G-03). Labeling a neighborhood result as
    ``bft`` is a fail-closed error.
    """

    label = normalize_guarantee_label(guarantee)
    if label == GUARANTEE_BFT:
        raise NeighborhoodGuaranteeError(
            "Profile G neighborhood results must not be labeled bft; "
            "neighborhood agreement is coordination / majority_approval only "
            "(ADR-0004 §4 / plan KD-11)"
        )
    if label not in PROFILE_G_ALLOWED_GUARANTEES:
        raise NeighborhoodGuaranteeError(
            f"Profile G neighborhood results must not be labeled {label!r}; "
            f"allowed guarantees are {sorted(PROFILE_G_ALLOWED_GUARANTEES)}"
        )
    return label


def require_plugin_guarantee(
    guarantee: object,
    *,
    implements_bft: bool = False,
) -> str:
    """Validate a plugin's declared guarantee.

    A plugin may declare ``bft`` only when it actually implements a BFT engine
    (``implements_bft=True``). The deterministic test adapter never sets that
    flag.
    """

    label = normalize_guarantee_label(guarantee)
    if label == GUARANTEE_BFT and not implements_bft:
        raise InvalidGuaranteeError(
            "guarantee 'bft' requires a real BFT engine (implements_bft=True); "
            "do not claim BFT without tested Byzantine fault tolerance"
        )
    return label


# ---------------------------------------------------------------------------
# Evidence and result records
# ---------------------------------------------------------------------------


def _require_non_empty_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConsensusEvidenceError(f"{field} must be a non-empty string")
    return value.strip()


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConsensusEvidenceError(f"{field} must be a mapping")
    return value


def _sorted_unique_strings(values: Iterable[object], *, field: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ConsensusEvidenceError(f"{field} entries must be non-empty strings")
        text = item.strip()
        if text not in seen:
            seen.add(text)
            out.append(text)
    out.sort()
    return tuple(out)


@dataclass(frozen=True)
class ConsensusEvidence:
    """Plugin evidence format for ``mode: consensus`` state transitions.

    Wire schema marker: ``mcp++/state/consensus-evidence@1``.

    Attributes:
        schema: Evidence schema marker.
        plugin_id: Declaring plugin identifier.
        guarantee: Honest guarantee label.
        state_id: Logical StateRef id the evidence applies to.
        proposal_cid: Content id of the proposed value / transition.
        evidence_kind: ``neighborhood`` | ``plugin`` | ``test``.
        members: Declared peer / principal set (sorted unique).
        approvals: Principals that approved / supported the proposal.
        rejections: Principals that challenged / rejected.
        abstentions: Principals that abstained.
        threshold: Required approval count for majority-class guarantees.
        round_id: Round / epoch identifier (deterministic string).
        source: Provenance tag (e.g. ``profile_g_neighborhood``).
        metadata: Non-authoritative annotations (must not redeclare guarantee).
    """

    plugin_id: str
    guarantee: str
    state_id: str
    proposal_cid: str
    evidence_kind: str
    members: tuple[str, ...] = ()
    approvals: tuple[str, ...] = ()
    rejections: tuple[str, ...] = ()
    abstentions: tuple[str, ...] = ()
    threshold: int = 0
    round_id: str = "0"
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CONSENSUS_EVIDENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready mapping."""

        return {
            "schema": self.schema,
            "plugin_id": self.plugin_id,
            "guarantee": self.guarantee,
            "state_id": self.state_id,
            "proposal_cid": self.proposal_cid,
            "evidence_kind": self.evidence_kind,
            "members": list(self.members),
            "approvals": list(self.approvals),
            "rejections": list(self.rejections),
            "abstentions": list(self.abstentions),
            "threshold": self.threshold,
            "round_id": self.round_id,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConsensusResult:
    """Accepted or rejected consensus outcome with an honest guarantee label.

    Wire schema marker: ``mcp++/state/consensus-result@1``.
    """

    plugin_id: str
    guarantee: str
    state_id: str
    proposal_cid: str
    accepted: bool
    evidence_kind: str
    approval_count: int
    threshold: int
    members: tuple[str, ...] = ()
    reason: str = ""
    round_id: str = "0"
    source: str = ""
    evidence: Optional[Mapping[str, Any]] = None
    schema: str = CONSENSUS_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready mapping."""

        out: dict[str, Any] = {
            "schema": self.schema,
            "plugin_id": self.plugin_id,
            "guarantee": self.guarantee,
            "state_id": self.state_id,
            "proposal_cid": self.proposal_cid,
            "accepted": self.accepted,
            "evidence_kind": self.evidence_kind,
            "approval_count": self.approval_count,
            "threshold": self.threshold,
            "members": list(self.members),
            "reason": self.reason,
            "round_id": self.round_id,
            "source": self.source,
        }
        if self.evidence is not None:
            out["evidence"] = dict(self.evidence)
        return out


def build_plugin_evidence(
    *,
    plugin_id: str,
    guarantee: object,
    state_id: str,
    proposal_cid: str,
    evidence_kind: str = EVIDENCE_KIND_PLUGIN,
    members: Optional[Sequence[str]] = None,
    approvals: Optional[Sequence[str]] = None,
    rejections: Optional[Sequence[str]] = None,
    abstentions: Optional[Sequence[str]] = None,
    threshold: Optional[int] = None,
    round_id: object = "0",
    source: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
    implements_bft: bool = False,
    profile_g_neighborhood: bool = False,
) -> ConsensusEvidence:
    """Build and validate a :class:`ConsensusEvidence` record.

    When ``profile_g_neighborhood`` is True (or ``evidence_kind`` is
    ``neighborhood``), the guarantee is constrained by
    :func:`require_profile_g_guarantee` so neighborhood results cannot be
    labeled BFT.
    """

    plugin = _require_non_empty_str(plugin_id, field="plugin_id")
    state = _require_non_empty_str(state_id, field="state_id")
    proposal = _require_non_empty_str(proposal_cid, field="proposal_cid")

    kind = _require_non_empty_str(evidence_kind, field="evidence_kind")
    if kind not in EVIDENCE_KINDS:
        raise ConsensusEvidenceError(
            f"evidence_kind {kind!r} is not one of {sorted(EVIDENCE_KINDS)}"
        )

    is_neighborhood = profile_g_neighborhood or kind == EVIDENCE_KIND_NEIGHBORHOOD
    if is_neighborhood:
        label = require_profile_g_guarantee(guarantee)
        kind = EVIDENCE_KIND_NEIGHBORHOOD
    else:
        label = require_plugin_guarantee(guarantee, implements_bft=implements_bft)

    member_t = _sorted_unique_strings(members or (), field="members")
    approval_t = _sorted_unique_strings(approvals or (), field="approvals")
    rejection_t = _sorted_unique_strings(rejections or (), field="rejections")
    abstention_t = _sorted_unique_strings(abstentions or (), field="abstentions")

    if threshold is None:
        if label in (GUARANTEE_MAJORITY_APPROVAL, GUARANTEE_CRASH_CONSENSUS, GUARANTEE_BFT):
            # Simple majority of declared members (ceil((n+1)/2) via integer math).
            n = len(member_t)
            computed = (n // 2) + 1 if n else 0
        else:
            computed = 0
        thr = computed
    else:
        if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
            raise ConsensusEvidenceError("threshold must be a non-negative integer")
        thr = threshold

    round_text = "0" if round_id is None else str(round_id).strip()
    if not round_text:
        raise ConsensusEvidenceError("round_id must be a non-empty string when provided")

    meta = dict(metadata or {})
    if "guarantee" in meta and meta["guarantee"] != label:
        raise ConsensusEvidenceError(
            "metadata must not redeclare a conflicting guarantee label"
        )

    return ConsensusEvidence(
        plugin_id=plugin,
        guarantee=label,
        state_id=state,
        proposal_cid=proposal,
        evidence_kind=kind,
        members=member_t,
        approvals=approval_t,
        rejections=rejection_t,
        abstentions=abstention_t,
        threshold=thr,
        round_id=round_text,
        source=str(source or ""),
        metadata=meta,
    )


def validate_plugin_evidence(evidence: Mapping[str, Any] | ConsensusEvidence) -> ConsensusEvidence:
    """Validate a mapping or evidence object and return a :class:`ConsensusEvidence`."""

    if isinstance(evidence, ConsensusEvidence):
        # Re-run through builder for fail-closed checks.
        return build_plugin_evidence(
            plugin_id=evidence.plugin_id,
            guarantee=evidence.guarantee,
            state_id=evidence.state_id,
            proposal_cid=evidence.proposal_cid,
            evidence_kind=evidence.evidence_kind,
            members=evidence.members,
            approvals=evidence.approvals,
            rejections=evidence.rejections,
            abstentions=evidence.abstentions,
            threshold=evidence.threshold,
            round_id=evidence.round_id,
            source=evidence.source,
            metadata=evidence.metadata,
            implements_bft=(evidence.guarantee == GUARANTEE_BFT),
            profile_g_neighborhood=(
                evidence.evidence_kind == EVIDENCE_KIND_NEIGHBORHOOD
            ),
        )

    mapping = _require_mapping(evidence, field="evidence")
    schema = mapping.get("schema", CONSENSUS_EVIDENCE_SCHEMA)
    if schema != CONSENSUS_EVIDENCE_SCHEMA:
        raise ConsensusEvidenceError(
            f"evidence.schema must be {CONSENSUS_EVIDENCE_SCHEMA!r}, got {schema!r}"
        )

    return build_plugin_evidence(
        plugin_id=mapping.get("plugin_id", ""),
        guarantee=mapping.get("guarantee"),
        state_id=mapping.get("state_id", ""),
        proposal_cid=mapping.get("proposal_cid", ""),
        evidence_kind=mapping.get("evidence_kind", EVIDENCE_KIND_PLUGIN),
        members=mapping.get("members") or (),
        approvals=mapping.get("approvals") or (),
        rejections=mapping.get("rejections") or (),
        abstentions=mapping.get("abstentions") or (),
        threshold=mapping.get("threshold"),
        round_id=mapping.get("round_id", "0"),
        source=str(mapping.get("source") or ""),
        metadata=mapping.get("metadata") or {},
        implements_bft=(mapping.get("guarantee") == GUARANTEE_BFT),
        profile_g_neighborhood=(
            mapping.get("evidence_kind") == EVIDENCE_KIND_NEIGHBORHOOD
            or bool(mapping.get("profile_g_neighborhood"))
            or str(mapping.get("source") or "") == "profile_g_neighborhood"
        ),
    )


def evaluate_majority(
    *,
    members: Sequence[str],
    approvals: Sequence[str],
    threshold: Optional[int] = None,
) -> tuple[bool, int, int]:
    """Evaluate a simple majority over a declared member set.

    Returns ``(accepted, approval_count, threshold_used)``. Approvals from
    non-members are ignored (fail-closed membership).
    """

    member_set = set(_sorted_unique_strings(members, field="members"))
    approval_set = {
        a
        for a in _sorted_unique_strings(approvals, field="approvals")
        if a in member_set
    }
    n = len(member_set)
    if threshold is None:
        thr = (n // 2) + 1 if n else 0
    else:
        if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
            raise ConsensusEvidenceError("threshold must be a non-negative integer")
        thr = threshold
    count = len(approval_set)
    accepted = thr > 0 and count >= thr
    return accepted, count, thr


# ---------------------------------------------------------------------------
# Profile G neighborhood wiring
# ---------------------------------------------------------------------------


def _ballot_from_attestation(attestation: Mapping[str, Any]) -> tuple[str, str]:
    """Extract ``(attester_did, verdict)`` from a Profile G-style attestation."""

    attester = attestation.get("attester_did") or attestation.get("peer_did") or attestation.get("principal")
    if not isinstance(attester, str) or not attester.strip():
        raise ConsensusEvidenceError(
            "neighborhood attestation requires attester_did (or peer_did/principal)"
        )
    verdict_raw = attestation.get("verdict", VERDICT_SUPPORT)
    if not isinstance(verdict_raw, str) or not verdict_raw.strip():
        raise ConsensusEvidenceError("neighborhood attestation verdict is required")
    verdict = verdict_raw.strip().lower()
    if verdict not in ALLOWED_VERDICTS:
        raise ConsensusEvidenceError(
            f"neighborhood verdict {verdict!r} is not one of {sorted(ALLOWED_VERDICTS)}"
        )
    return attester.strip(), verdict


def wire_neighborhood_result(
    *,
    state_id: str,
    proposal_cid: str,
    attestations: Sequence[Mapping[str, Any]],
    members: Optional[Sequence[str]] = None,
    guarantee: object = GUARANTEE_MAJORITY_APPROVAL,
    threshold: Optional[int] = None,
    round_id: object = "0",
    plugin_id: str = PROFILE_G_NEIGHBORHOOD_PLUGIN_ID,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    accept_on_coordination: bool = True,
) -> ConsensusResult:
    """Wire Profile G neighborhood records into plugin evidence and a result.

    The ``guarantee`` argument is validated with :func:`require_profile_g_guarantee`.
    Labeling the neighborhood result as ``bft`` raises
    :class:`NeighborhoodGuaranteeError`.

    For ``majority_approval``, acceptance requires meeting the majority
    threshold among the declared member set. For ``coordination``, acceptance
    is best-effort: if ``accept_on_coordination`` is True (default), any
    non-empty support set accepts; otherwise the same threshold math is used
    for observation only without claiming crash or Byzantine safety.
    """

    label = require_profile_g_guarantee(guarantee)

    approvals: list[str] = []
    rejections: list[str] = []
    abstentions: list[str] = []
    for raw in attestations:
        mapping = _require_mapping(raw, field="attestation")
        attester, verdict = _ballot_from_attestation(mapping)
        if verdict == VERDICT_SUPPORT:
            approvals.append(attester)
        elif verdict == VERDICT_CHALLENGE:
            rejections.append(attester)
        else:
            abstentions.append(attester)

    if members is None:
        # Derive membership from ballots + optional NeighborhoodRecord peer_dids.
        derived: list[str] = list(approvals) + list(rejections) + list(abstentions)
        if records:
            for rec in records:
                rec_map = _require_mapping(rec, field="neighborhood_record")
                peer = rec_map.get("peer_did") or rec_map.get("signer_did")
                if isinstance(peer, str) and peer.strip():
                    derived.append(peer.strip())
        member_list: Sequence[str] = derived
    else:
        member_list = members

    evidence = build_plugin_evidence(
        plugin_id=plugin_id,
        guarantee=label,
        state_id=state_id,
        proposal_cid=proposal_cid,
        evidence_kind=EVIDENCE_KIND_NEIGHBORHOOD,
        members=member_list,
        approvals=approvals,
        rejections=rejections,
        abstentions=abstentions,
        threshold=threshold,
        round_id=round_id,
        source="profile_g_neighborhood",
        metadata={
            "profile_g": True,
            "attestation_count": len(attestations),
            "record_count": len(records or ()),
        },
        profile_g_neighborhood=True,
    )

    if label == GUARANTEE_COORDINATION:
        # Coordination: no crash/Byzantine safety claim. Best-effort alignment.
        if accept_on_coordination:
            accepted = len(evidence.approvals) > 0
            thr_used = 0
            count = len(evidence.approvals)
            reason = (
                "coordination support observed"
                if accepted
                else "coordination: no supporting attestations"
            )
        else:
            accepted, count, thr_used = evaluate_majority(
                members=evidence.members,
                approvals=evidence.approvals,
                threshold=evidence.threshold or None,
            )
            reason = (
                "coordination majority observed (no safety claim)"
                if accepted
                else "coordination threshold not met (no safety claim)"
            )
    else:
        # majority_approval
        accepted, count, thr_used = evaluate_majority(
            members=evidence.members,
            approvals=evidence.approvals,
            threshold=evidence.threshold or None,
        )
        reason = (
            f"majority_approval: {count}/{thr_used} approvals among "
            f"{len(evidence.members)} members"
            if accepted
            else f"majority_approval: insufficient approvals "
            f"({count} < {thr_used}) among {len(evidence.members)} members"
        )

    return ConsensusResult(
        plugin_id=evidence.plugin_id,
        guarantee=evidence.guarantee,
        state_id=evidence.state_id,
        proposal_cid=evidence.proposal_cid,
        accepted=accepted,
        evidence_kind=evidence.evidence_kind,
        approval_count=count,
        threshold=thr_used if label != GUARANTEE_COORDINATION or not accept_on_coordination else 0,
        members=evidence.members,
        reason=reason,
        round_id=evidence.round_id,
        source=evidence.source,
        evidence=evidence.to_dict(),
    )


def label_neighborhood_result(
    guarantee: object,
    *,
    state_id: str = "state:label-check",
    proposal_cid: str = "bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e",
    supporters: Optional[Sequence[str]] = None,
) -> ConsensusResult:
    """Convenience helper: attempt to label a neighborhood result.

    Raises :class:`NeighborhoodGuaranteeError` when ``guarantee`` is ``bft``
    (or any non-Profile-G label). Used by tests to assert fail-closed labeling.
    """

    peers = list(supporters or ("did:key:peer-a", "did:key:peer-b", "did:key:peer-c"))
    attestations = [
        {"attester_did": peer, "verdict": VERDICT_SUPPORT} for peer in peers
    ]
    return wire_neighborhood_result(
        state_id=state_id,
        proposal_cid=proposal_cid,
        attestations=attestations,
        members=peers,
        guarantee=guarantee,
    )


# ---------------------------------------------------------------------------
# Plugin contract
# ---------------------------------------------------------------------------


class ConsensusPlugin(ABC):
    """Abstract ConsensusPlugin@1 contract for ``mode: consensus`` providers.

    Implementations MUST:

    * declare a stable ``plugin_id``;
    * declare exactly one honest ``guarantee`` from :data:`GUARANTEE_LABELS`;
    * refuse state transitions without valid plugin evidence;
    * never upgrade Profile G neighborhood evidence to ``bft``.
    """

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Stable plugin identifier (e.g. ``mcp++/consensus/raft-local@1``)."""

    @property
    @abstractmethod
    def guarantee(self) -> str:
        """Honest guarantee label this plugin actually provides."""

    @property
    def interface(self) -> str:
        """Wire interface label."""

        return CONSENSUS_PLUGIN_INTERFACE

    @property
    def implements_bft(self) -> bool:
        """True only when this plugin implements a real BFT engine."""

        return False

    @property
    def mode(self) -> str:
        """Consistency mode this plugin serves (always ``consensus``)."""

        return CONSENSUS_MODE

    def describe(self) -> dict[str, Any]:
        """Return a deterministic descriptor for diagnostics and tests."""

        return {
            "interface": self.interface,
            "plugin_id": self.plugin_id,
            "guarantee": self.guarantee,
            "mode": self.mode,
            "implements_bft": self.implements_bft,
        }

    @abstractmethod
    def propose(
        self,
        *,
        state_id: str,
        proposal_cid: str,
        members: Sequence[str],
        round_id: object = "0",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ConsensusEvidence:
        """Open a proposal round and return initial evidence (pre-ballots)."""

    @abstractmethod
    def record_ballot(
        self,
        evidence: Mapping[str, Any] | ConsensusEvidence,
        *,
        principal: str,
        verdict: str,
    ) -> ConsensusEvidence:
        """Record one ballot against existing evidence and return updated evidence."""

    @abstractmethod
    def evaluate(
        self,
        evidence: Mapping[str, Any] | ConsensusEvidence,
    ) -> ConsensusResult:
        """Evaluate evidence under this plugin's guarantee; never silently upgrades labels."""

    def accept(
        self,
        evidence: Mapping[str, Any] | ConsensusEvidence,
    ) -> ConsensusResult:
        """Evaluate evidence and raise :class:`ConsensusRejectedError` if not accepted."""

        result = self.evaluate(evidence)
        if not result.accepted:
            raise ConsensusRejectedError(
                result.reason or "consensus evidence was not accepted"
            )
        return result


# ---------------------------------------------------------------------------
# Deterministic test adapter
# ---------------------------------------------------------------------------


class DeterministicTestAdapter(ConsensusPlugin):
    """Deterministic consensus adapter for tests (never claims BFT).

    Supports ``coordination`` and ``majority_approval`` only. Ballots and
    membership are processed in sorted order so results are independent of
    input interleaving. Profile G neighborhood results are wired through
    :func:`wire_neighborhood_result` and fail closed if labeled ``bft``.
    """

    def __init__(
        self,
        *,
        guarantee: object = GUARANTEE_MAJORITY_APPROVAL,
        plugin_id: str = DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID,
        default_threshold: Optional[int] = None,
    ) -> None:
        label = normalize_guarantee_label(guarantee)
        if label not in PROFILE_G_ALLOWED_GUARANTEES:
            # Test adapter intentionally does not implement crash_consensus or bft.
            raise InvalidGuaranteeError(
                f"DeterministicTestAdapter supports only "
                f"{sorted(PROFILE_G_ALLOWED_GUARANTEES)}; got {label!r}. "
                "Do not claim BFT (or crash consensus) from the test adapter."
            )
        self._guarantee = label
        self._plugin_id = _require_non_empty_str(plugin_id, field="plugin_id")
        self._default_threshold = default_threshold
        self._rounds: dict[tuple[str, str, str], ConsensusEvidence] = {}

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    @property
    def guarantee(self) -> str:
        return self._guarantee

    @property
    def implements_bft(self) -> bool:
        return False

    def propose(
        self,
        *,
        state_id: str,
        proposal_cid: str,
        members: Sequence[str],
        round_id: object = "0",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ConsensusEvidence:
        evidence = build_plugin_evidence(
            plugin_id=self.plugin_id,
            guarantee=self.guarantee,
            state_id=state_id,
            proposal_cid=proposal_cid,
            evidence_kind=EVIDENCE_KIND_TEST,
            members=members,
            approvals=(),
            rejections=(),
            abstentions=(),
            threshold=self._default_threshold,
            round_id=round_id,
            source="deterministic_test_adapter",
            metadata=metadata,
            implements_bft=False,
        )
        key = (evidence.state_id, evidence.proposal_cid, evidence.round_id)
        self._rounds[key] = evidence
        return evidence

    def record_ballot(
        self,
        evidence: Mapping[str, Any] | ConsensusEvidence,
        *,
        principal: str,
        verdict: str,
    ) -> ConsensusEvidence:
        current = validate_plugin_evidence(evidence)
        if current.guarantee != self.guarantee:
            raise InvalidGuaranteeError(
                f"evidence guarantee {current.guarantee!r} does not match "
                f"adapter guarantee {self.guarantee!r}"
            )
        peer = _require_non_empty_str(principal, field="principal")
        if peer not in current.members:
            raise ConsensusEvidenceError(
                f"principal {peer!r} is not in the declared member set"
            )
        v = _require_non_empty_str(verdict, field="verdict").lower()
        if v not in ALLOWED_VERDICTS:
            raise ConsensusEvidenceError(
                f"verdict {v!r} is not one of {sorted(ALLOWED_VERDICTS)}"
            )

        approvals = set(current.approvals)
        rejections = set(current.rejections)
        abstentions = set(current.abstentions)
        # Remove prior ballot from this principal (idempotent re-vote).
        approvals.discard(peer)
        rejections.discard(peer)
        abstentions.discard(peer)
        if v == VERDICT_SUPPORT:
            approvals.add(peer)
        elif v == VERDICT_CHALLENGE:
            rejections.add(peer)
        else:
            abstentions.add(peer)

        updated = build_plugin_evidence(
            plugin_id=current.plugin_id,
            guarantee=current.guarantee,
            state_id=current.state_id,
            proposal_cid=current.proposal_cid,
            evidence_kind=current.evidence_kind,
            members=current.members,
            approvals=sorted(approvals),
            rejections=sorted(rejections),
            abstentions=sorted(abstentions),
            threshold=current.threshold,
            round_id=current.round_id,
            source=current.source or "deterministic_test_adapter",
            metadata=current.metadata,
            implements_bft=False,
            profile_g_neighborhood=(
                current.evidence_kind == EVIDENCE_KIND_NEIGHBORHOOD
            ),
        )
        key = (updated.state_id, updated.proposal_cid, updated.round_id)
        self._rounds[key] = updated
        return updated

    def evaluate(
        self,
        evidence: Mapping[str, Any] | ConsensusEvidence,
    ) -> ConsensusResult:
        current = validate_plugin_evidence(evidence)
        if current.guarantee != self.guarantee:
            raise InvalidGuaranteeError(
                f"evidence guarantee {current.guarantee!r} does not match "
                f"adapter guarantee {self.guarantee!r}"
            )
        if current.evidence_kind == EVIDENCE_KIND_NEIGHBORHOOD:
            # Re-validate neighborhood constraint (fail closed on BFT).
            require_profile_g_guarantee(current.guarantee)

        if current.guarantee == GUARANTEE_COORDINATION:
            accepted = len(current.approvals) > 0
            count = len(current.approvals)
            thr = 0
            reason = (
                "coordination support observed"
                if accepted
                else "coordination: no supporting ballots"
            )
        else:
            accepted, count, thr = evaluate_majority(
                members=current.members,
                approvals=current.approvals,
                threshold=current.threshold or None,
            )
            reason = (
                f"majority_approval met ({count} >= {thr})"
                if accepted
                else f"majority_approval not met ({count} < {thr})"
            )

        return ConsensusResult(
            plugin_id=self.plugin_id,
            guarantee=current.guarantee,
            state_id=current.state_id,
            proposal_cid=current.proposal_cid,
            accepted=accepted,
            evidence_kind=current.evidence_kind,
            approval_count=count,
            threshold=thr,
            members=current.members,
            reason=reason,
            round_id=current.round_id,
            source=current.source or "deterministic_test_adapter",
            evidence=current.to_dict(),
        )

    def evaluate_neighborhood(
        self,
        *,
        state_id: str,
        proposal_cid: str,
        attestations: Sequence[Mapping[str, Any]],
        members: Optional[Sequence[str]] = None,
        guarantee: Optional[object] = None,
        threshold: Optional[int] = None,
        round_id: object = "0",
        records: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> ConsensusResult:
        """Evaluate Profile G neighborhood ballots under an honest label.

        Defaults to this adapter's guarantee. Passing ``guarantee='bft'``
        raises :class:`NeighborhoodGuaranteeError`.
        """

        label = self.guarantee if guarantee is None else guarantee
        # Fail closed before wiring if the caller attempts a BFT label.
        require_profile_g_guarantee(label)
        return wire_neighborhood_result(
            state_id=state_id,
            proposal_cid=proposal_cid,
            attestations=attestations,
            members=members,
            guarantee=label,
            threshold=threshold if threshold is not None else self._default_threshold,
            round_id=round_id,
            plugin_id=PROFILE_G_NEIGHBORHOOD_PLUGIN_ID,
            records=records,
        )

    def stats(self) -> MutableMapping[str, Any]:
        """Return deterministic adapter diagnostics."""

        return {
            "interface": self.interface,
            "plugin_id": self.plugin_id,
            "guarantee": self.guarantee,
            "implements_bft": self.implements_bft,
            "open_rounds": len(self._rounds),
            "supported_guarantees": sorted(PROFILE_G_ALLOWED_GUARANTEES),
        }


__all__ = [
    "ALLOWED_VERDICTS",
    "CONSENSUS_EVIDENCE_SCHEMA",
    "CONSENSUS_MODE",
    "CONSENSUS_PLUGIN_INTERFACE",
    "CONSENSUS_RESULT_SCHEMA",
    "DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID",
    "EVIDENCE_KIND_NEIGHBORHOOD",
    "EVIDENCE_KIND_PLUGIN",
    "EVIDENCE_KIND_TEST",
    "EVIDENCE_KINDS",
    "GUARANTEE_BFT",
    "GUARANTEE_COORDINATION",
    "GUARANTEE_CRASH_CONSENSUS",
    "GUARANTEE_LABELS",
    "GUARANTEE_MAJORITY_APPROVAL",
    "PROFILE_G_ALLOWED_GUARANTEES",
    "PROFILE_G_NEIGHBORHOOD_PLUGIN_ID",
    "VERDICT_ABSTAIN",
    "VERDICT_CHALLENGE",
    "VERDICT_SUPPORT",
    "ConsensusEvidence",
    "ConsensusEvidenceError",
    "ConsensusPlugin",
    "ConsensusPluginError",
    "ConsensusRejectedError",
    "ConsensusResult",
    "DeterministicTestAdapter",
    "GuaranteeLabel",
    "InvalidGuaranteeError",
    "NeighborhoodGuaranteeError",
    "build_plugin_evidence",
    "evaluate_majority",
    "is_profile_g_allowed_guarantee",
    "label_neighborhood_result",
    "normalize_guarantee_label",
    "require_plugin_guarantee",
    "require_profile_g_guarantee",
    "validate_plugin_evidence",
    "wire_neighborhood_result",
]
