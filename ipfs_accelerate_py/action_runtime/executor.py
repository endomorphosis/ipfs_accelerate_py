"""Orchestrate proposal -> policy decision -> admitted adapter invocation."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters.cli import CLIActionAdapter
from .catalog import ActionCatalog
from .contracts import ActionDecision, ActionProposal, ActionReceipt, ActionStatus
from .policy import ActionPolicyEngine


@dataclass
class ActionExecutor:
    """Fail-closed executor that never runs adapters without a permit."""

    catalog: ActionCatalog
    policy: ActionPolicyEngine
    cli_adapter: CLIActionAdapter | None = None

    def evaluate(self, proposal: ActionProposal) -> ActionDecision:
        return self.policy.decide(proposal)

    def execute(self, proposal: ActionProposal) -> tuple[ActionDecision, ActionReceipt]:
        decision = self.evaluate(proposal)
        if not decision.permits_execution:
            receipt = ActionReceipt(
                receipt_id=f"rcpt-denied-{decision.decision_id[-12:]}",
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="none",
                interface_identity="none",
                error=decision.reason,
            )
            return decision, receipt

        descriptor = self.catalog.require(proposal.descriptor_id)
        if descriptor.adapter == "cli":
            if self.cli_adapter is None:
                receipt = ActionReceipt(
                    receipt_id=f"rcpt-missing-cli-{decision.decision_id[-12:]}",
                    status=ActionStatus.FAILED,
                    proposal_id=proposal.proposal_id,
                    decision_id=decision.decision_id,
                    descriptor_id=proposal.descriptor_id,
                    adapter="cli",
                    interface_identity="cli:unconfigured",
                    error="cli_adapter_not_configured",
                )
                return decision, receipt
            return decision, self.cli_adapter.invoke(proposal=proposal, decision=decision)

        receipt = ActionReceipt(
            receipt_id=f"rcpt-unsupported-{decision.decision_id[-12:]}",
            status=ActionStatus.FAILED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter=descriptor.adapter,
            interface_identity=f"{descriptor.adapter}:unsupported",
            error="adapter_not_implemented",
        )
        return decision, receipt
