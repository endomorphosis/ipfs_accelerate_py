"""Fail-closed voice/customer-care action runtime.

GraphRAG and response templates may *propose* logical actions.  Only an
operator-owned catalog, policy decision, and admitted adapter may execute.
Importing this package never starts processes or loads credentials.
"""

from __future__ import annotations

from .catalog import ActionCatalog, ActionDescriptor
from .contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    ActionReceipt,
    ActionStatus,
    RiskClass,
    SideEffectClass,
)
from .executor import ActionExecutor
from .policy import ActionPolicyEngine, FailClosedPolicy
from .voice_bridge import VoiceActionBridge, propose_from_voice_route

__all__ = [
    "ActionCatalog",
    "ActionDecision",
    "ActionDecisionKind",
    "ActionDescriptor",
    "ActionExecutor",
    "ActionPolicyEngine",
    "ActionProposal",
    "ActionReceipt",
    "ActionStatus",
    "FailClosedPolicy",
    "RiskClass",
    "SideEffectClass",
    "VoiceActionBridge",
    "propose_from_voice_route",
]
