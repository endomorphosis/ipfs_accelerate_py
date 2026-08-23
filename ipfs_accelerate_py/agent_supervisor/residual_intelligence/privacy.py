"""Hard information-flow gates across residual intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .contracts import PrivacyClass, ResidualIntelligenceError, required_text
from .rights import TrainingCorpusAdmission

POLICY_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-information-flow-policy@1"
WITHDRAWAL_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-corpus-withdrawal@1"
NEVER_TRAIN: Final[frozenset[PrivacyClass]] = frozenset(
    {PrivacyClass.CREDENTIAL, PrivacyClass.PROOF_WITNESS}
)
REMOTE_DENIED_DEFAULT: Final[frozenset[PrivacyClass]] = frozenset(
    {
        PrivacyClass.CREDENTIAL,
        PrivacyClass.PROOF_WITNESS,
        PrivacyClass.PERSONAL_DATA,
        PrivacyClass.HEALTH_DATA,
        PrivacyClass.LEGAL_PRIVILEGED,
        PrivacyClass.MATTER_CONFIDENTIAL,
        PrivacyClass.TENANT_PRIVATE,
    }
)


@dataclass(frozen=True)
class InformationFlowPolicy:
    schema: str = POLICY_SCHEMA

    def may_train(self, privacy: PrivacyClass) -> bool:
        return PrivacyClass(privacy) not in NEVER_TRAIN

    def may_publish(self, privacy: PrivacyClass) -> bool:
        return PrivacyClass(privacy) not in {
            PrivacyClass.PROOF_WITNESS,
            PrivacyClass.CREDENTIAL,
            PrivacyClass.PERSONAL_DATA,
            PrivacyClass.HEALTH_DATA,
            PrivacyClass.LEGAL_PRIVILEGED,
        }

    def may_send_remote(self, privacy: PrivacyClass, *, authorized: bool) -> bool:
        privacy = PrivacyClass(privacy)
        if privacy in REMOTE_DENIED_DEFAULT and not authorized:
            return False
        if privacy is PrivacyClass.CREDENTIAL:
            return False
        return True


@dataclass(frozen=True)
class PrivacyRouteDecision:
    privacy: PrivacyClass
    train: bool
    publish: bool
    remote: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class CorpusWithdrawal:
    admission_id: str
    reason: str
    schema: str = WITHDRAWAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "admission_id", required_text(self.admission_id, "admission_id"))
        object.__setattr__(self, "reason", required_text(self.reason, "reason"))


@dataclass(frozen=True)
class DeclassificationAuthority:
    scope_id: str
    allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", required_text(self.scope_id, "scope_id"))
        if type(self.allowed) is not bool:
            raise ResidualIntelligenceError("declassification allowed flag must be boolean")


def route_privacy(
    privacy: PrivacyClass,
    *,
    remote_authorized: bool = False,
    declassification: DeclassificationAuthority | None = None,
) -> PrivacyRouteDecision:
    policy = InformationFlowPolicy()
    publish = policy.may_publish(privacy)
    if declassification is not None and declassification.allowed and publish is False:
        # Explicit scoped declassification still cannot publish credentials/proof witnesses.
        if privacy not in {PrivacyClass.CREDENTIAL, PrivacyClass.PROOF_WITNESS}:
            publish = True
    reasons = []
    if not policy.may_train(privacy):
        reasons.append("credentials_or_witness_never_train")
    if not policy.may_send_remote(privacy, authorized=remote_authorized):
        reasons.append("remote_denied")
    if privacy is PrivacyClass.PROOF_WITNESS:
        reasons.append("proof_witness_no_public")
    return PrivacyRouteDecision(
        privacy=PrivacyClass(privacy),
        train=policy.may_train(privacy),
        publish=publish,
        remote=policy.may_send_remote(privacy, authorized=remote_authorized),
        reason_codes=tuple(reasons),
    )


def withdrawal_invalidates(admission: TrainingCorpusAdmission, withdrawal: CorpusWithdrawal) -> bool:
    return admission.admission_id == withdrawal.admission_id
