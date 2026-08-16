"""DCR-074 non-mutating merge/commit/gitlink publication proposals.

The module receives only observations and typed prior evidence.  It does not
run Git, discover a checkout, write bytes, contact providers, or create a
publication receipt.  A proposal is deliberately not DCR-002 PUBLISHED
authority and remains integration-pending until a separate sealed executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..planning.proof_carrying_repair_dag import (
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanDagResult,
    RepairPlanNodeKind,
)
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .contracts import AuthorityStage, RepairEvidenceEnvelope
from .root_ownership import RootBinding
from .validation import PostRepairDisposition, RepairProofTransition

DCR074_PUBLICATION_PROPOSAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-074-publication-proposal@1"
)
DCR074_ACTIVATION: Final = "integration_pending_external_merge_executor_sealed_dcr073"
_GIT_ID: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40,64}$")


class PublicationProposalDisposition(str, Enum):  # noqa: UP042 - Python 3.8
    INTEGRATION_PENDING = "integration_pending"
    STALE = "stale"
    REPLAN = "replan"
    REJECTED = "rejected"


class PublicationProposalError(ValueError):
    """An observed publication record is stale, synthetic, or unordered."""


@dataclass(frozen=True)
class ObservedCommitRecord:
    """Externally observed Git state only; this is not a commit instruction."""

    root_id: str
    predecessor_head: str
    successor_head: str
    successor_tree: str
    parent_heads: tuple[str, ...]
    diff_digest: str
    operator_cid: str
    validation_cid: str
    proof_cid: str
    observed: bool = True
    provenance: str = "observed_git"

    def __post_init__(self) -> None:
        for field in ("predecessor_head", "successor_head", "successor_tree"):
            if not _GIT_ID.fullmatch(getattr(self, field)):
                raise PublicationProposalError(f"{field} must be an observed Git object id")
        if not self.root_id or not all(
            isinstance(item, str) and item for item in (self.diff_digest, self.operator_cid, self.validation_cid, self.proof_cid)
        ):
            raise PublicationProposalError("commit provenance is incomplete")
        if self.observed is not True or self.provenance != "observed_git":
            raise PublicationProposalError("synthetic or expected commits are not publication evidence")
        if not self.parent_heads or any(not _GIT_ID.fullmatch(item) for item in self.parent_heads):
            raise PublicationProposalError("observed commit parents are required")

    @property
    def record_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR074_PUBLICATION_PROPOSAL_SCHEMA,
            "root_id": self.root_id,
            "predecessor_head": self.predecessor_head,
            "successor_head": self.successor_head,
            "successor_tree": self.successor_tree,
            "parent_heads": list(self.parent_heads),
            "diff_digest": self.diff_digest,
            "operator_cid": self.operator_cid,
            "validation_cid": self.validation_cid,
            "proof_cid": self.proof_cid,
            "observed": self.observed,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ObservedGitlinkRecord:
    """Observed parent gitlink predecessor/successor, not a pin operation."""

    consumer_root_id: str
    provider_root_id: str
    pin_path: str
    predecessor: str
    successor: str
    observed: bool = True
    provenance: str = "observed_git"

    def __post_init__(self) -> None:
        if not self.consumer_root_id or not self.provider_root_id or not self.pin_path:
            raise PublicationProposalError("gitlink owner and path are required")
        if self.pin_path.startswith("/") or ".." in self.pin_path.split("/"):
            raise PublicationProposalError("gitlink path escapes consumer root")
        if not _GIT_ID.fullmatch(self.predecessor) or not _GIT_ID.fullmatch(self.successor):
            raise PublicationProposalError("gitlink revisions must be observed Git object ids")
        if self.observed is not True or self.provenance != "observed_git":
            raise PublicationProposalError("synthetic gitlink is not publication evidence")


@dataclass(frozen=True)
class PublicationProposal:
    disposition: PublicationProposalDisposition
    reason_codes: tuple[str, ...]
    validation_cid: str = ""
    reproof_cid: str = ""
    commit_record_cids: tuple[str, ...] = ()
    gitlink: ObservedGitlinkRecord | None = None

    @property
    def proposal_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR074_PUBLICATION_PROPOSAL_SCHEMA,
            "authoritative": False,
            "activation_status": DCR074_ACTIVATION,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "validation_cid": self.validation_cid,
            "reproof_cid": self.reproof_cid,
            "commit_record_cids": list(self.commit_record_cids),
            "gitlink": None if self.gitlink is None else self.gitlink.__dict__,
            "execution_authorized": False,
            "publication_authorized": False,
            "completion_authorized": False,
            "git_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
        }


def _ordered(before: str, after: str, plan: ProofCarryingRepairPlan) -> bool:
    nodes = {node.node_id: node for node in plan.nodes}
    pending = list(nodes[after].dependencies)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == before:
            return True
        if current not in seen:
            seen.add(current)
            pending.extend(nodes[current].dependencies)
    return False


def _result(disposition: PublicationProposalDisposition, *reasons: str) -> PublicationProposal:
    return PublicationProposal(disposition=disposition, reason_codes=tuple(sorted(set(reasons))))


def propose_publication(
    *,
    validation: RepairProofTransition,
    reproved: RepairEvidenceEnvelope,
    immediate_predecessor: RepairEvidenceEnvelope,
    root_bindings: tuple[RootBinding, ...],
    plan: ProofCarryingRepairPlan,
    dag_result: RepairPlanDagResult,
    commits: tuple[ObservedCommitRecord, ...],
    gitlink: ObservedGitlinkRecord,
) -> PublicationProposal:
    """Check publication provenance and return an intentionally pending proposal."""

    try:
        if not isinstance(validation, RepairProofTransition) or validation.transition_cid != content_identity(validation.to_dict()):
            raise PublicationProposalError("DCR-073 validation result is missing or forged")
        if validation.disposition is not PostRepairDisposition.INTEGRATION_PENDING:
            raise PublicationProposalError("DCR-073 validation is stale or non-passing")
        if not isinstance(reproved, RepairEvidenceEnvelope) or not isinstance(immediate_predecessor, RepairEvidenceEnvelope):
            raise PublicationProposalError("typed DCR-002 reproof chain is required")
        if reproved.authority_stage is not AuthorityStage.REPROVED or immediate_predecessor.authority_stage is not AuthorityStage.POST_EDIT_VALIDATED:
            raise PublicationProposalError("DCR-002 predecessor must be immediate post-edit validation")
        if reproved.previous_envelope_cid != immediate_predecessor.content_id:
            raise PublicationProposalError("DCR-002 reproof predecessor chain is forged")
        reproved.require_advances(immediate_predecessor)
        reproved.require_typed_authority()
        if not isinstance(plan, ProofCarryingRepairPlan) or not isinstance(dag_result, RepairPlanDagResult):
            raise PublicationProposalError("typed DCR-061 plan and result required")
        if dag_result.disposition is not RepairPlanDagDisposition.INTEGRATION_PENDING or dag_result.plan_cid != plan.content_id:
            raise PublicationProposalError("DCR-061 DAG result is stale or rejected")
        bindings = {item.root_id: item for item in root_bindings}
        if not bindings or len(bindings) != len(root_bindings) or any(item.dirty for item in bindings.values()):
            raise PublicationProposalError("DCR-003 roots have unrelated dirty overlay")
        records = {item.root_id: item for item in commits}
        if len(records) != len(commits) or not records:
            raise PublicationProposalError("commit observations are missing or duplicate")
        for root_id, record in records.items():
            binding = bindings.get(root_id)
            if binding is None:
                raise PublicationProposalError("commit root is outside DCR-003 owner bindings")
            if record.predecessor_head != binding.head:
                raise PublicationProposalError("target head drifted before proposal")
            if record.predecessor_head not in record.parent_heads:
                raise PublicationProposalError("observed successor does not name its predecessor parent")
            if not all((record.diff_digest, record.operator_cid, record.validation_cid, record.proof_cid)):
                raise PublicationProposalError("commit lacks exact diff/operator/validation/proof provenance")
            if record.validation_cid != validation.transition_cid or record.proof_cid != reproved.reproof_cid:
                raise PublicationProposalError("commit validation or proof provenance drifted")
        provider_nodes = [node.node_id for node in plan.nodes if node.kind is RepairPlanNodeKind.PROVIDER_COMMIT]
        consumer_nodes = [node.node_id for node in plan.nodes if node.kind is RepairPlanNodeKind.CONSUMER_VALIDATION]
        pin_nodes = [node.node_id for node in plan.nodes if node.kind is RepairPlanNodeKind.OUTER_GITLINK_PIN]
        if not provider_nodes or not consumer_nodes or not pin_nodes:
            raise PublicationProposalError("DCR-061 provider/consumer/pin ordering nodes required")
        if not all(_ordered(provider, consumer, plan) for provider in provider_nodes for consumer in consumer_nodes):
            raise PublicationProposalError("provider commit is not ordered before consumer validation")
        if not all(_ordered(consumer, pin, plan) for consumer in consumer_nodes for pin in pin_nodes):
            raise PublicationProposalError("premature gitlink pin lacks consumer validation")
        provider_record = records.get(gitlink.provider_root_id)
        consumer_binding = bindings.get(gitlink.consumer_root_id)
        if provider_record is None or consumer_binding is None:
            raise PublicationProposalError("gitlink provider/consumer ownership is absent")
        if gitlink.predecessor != provider_record.predecessor_head or gitlink.successor != provider_record.successor_head:
            raise PublicationProposalError("gitlink predecessor/successor does not bind provider commit")
        return PublicationProposal(
            disposition=PublicationProposalDisposition.INTEGRATION_PENDING,
            reason_codes=("external_merge_executor_and_sealed_dcr073_required",),
            validation_cid=validation.transition_cid,
            reproof_cid=reproved.reproof_cid,
            commit_record_cids=tuple(sorted(item.record_cid for item in records.values())),
            gitlink=gitlink,
        )
    except PublicationProposalError as exc:
        text = str(exc)
        disposition = (
            PublicationProposalDisposition.STALE
            if "drift" in text or "stale" in text
            else PublicationProposalDisposition.REPLAN
            if "ordered" in text or "premature" in text
            else PublicationProposalDisposition.REJECTED
        )
        return _result(disposition, text)


def canonical_publication_proposal_bytes(value: PublicationProposal) -> bytes:
    if not isinstance(value, PublicationProposal):
        raise PublicationProposalError("proposal must be typed")
    return canonical_json_bytes(value.to_dict())


__all__ = [
    "DCR074_ACTIVATION",
    "DCR074_PUBLICATION_PROPOSAL_SCHEMA",
    "ObservedCommitRecord",
    "ObservedGitlinkRecord",
    "PublicationProposal",
    "PublicationProposalDisposition",
    "PublicationProposalError",
    "canonical_publication_proposal_bytes",
    "propose_publication",
]
