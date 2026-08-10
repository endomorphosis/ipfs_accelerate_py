"""Agent-supervisor evaluation corpora and adversarial suites."""

from .dcr_adversarial import (
    ADVERSARIAL_CONFORMANCE_INTERFACE,
    MUTATION_SCORE_INTERFACE,
    AuthorityMutationSuite,
    ContractRepairAdversary,
    DcrAdversarialReport,
    evaluate_dcr_adversarial,
    materialize_adversarial_report,
)

__all__ = [
    "ADVERSARIAL_CONFORMANCE_INTERFACE",
    "MUTATION_SCORE_INTERFACE",
    "AuthorityMutationSuite",
    "ContractRepairAdversary",
    "DcrAdversarialReport",
    "evaluate_dcr_adversarial",
    "materialize_adversarial_report",
]
