from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import PrivacyClass
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.privacy import (
    CorpusWithdrawal,
    route_privacy,
    withdrawal_invalidates,
)
from .helpers import admission


def test_credentials_and_proof_witnesses_never_train_or_publish() -> None:
    cred = route_privacy(PrivacyClass.CREDENTIAL)
    assert cred.train is False
    assert cred.publish is False
    assert cred.remote is False
    witness = route_privacy(PrivacyClass.PROOF_WITNESS)
    assert witness.train is False
    assert witness.publish is False
    tenant = route_privacy(PrivacyClass.TENANT_PRIVATE, remote_authorized=False)
    assert tenant.remote is False
    tenant_ok = route_privacy(PrivacyClass.TENANT_PRIVATE, remote_authorized=True)
    assert tenant_ok.remote is True


def test_withdrawal_propagates_to_admission() -> None:
    record, _ = admission()
    withdrawal = CorpusWithdrawal(admission_id=record.admission_id, reason="privacy-withdraw")
    assert withdrawal_invalidates(record, withdrawal) is True
