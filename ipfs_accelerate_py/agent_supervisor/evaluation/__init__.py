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
from .dcr_benchmark import (
    DETERMINISTIC_REPAIR_BENCHMARK_INTERFACE,
    DeterministicRepairBenchmark,
    RepairSafetyMetrics,
    materialize_benchmark,
    run_deterministic_repair_benchmark,
)
from .dcr_fixed_point import (
    CONTRACT_REPAIR_FIXED_POINT_INTERFACE,
    ContractRepairFixedPoint,
    materialize_fixed_point,
    reach_contract_repair_fixed_point,
    supersede_legacy_repairs,
)

__all__ = [
    "ADVERSARIAL_CONFORMANCE_INTERFACE",
    "CONTRACT_REPAIR_FIXED_POINT_INTERFACE",
    "DETERMINISTIC_REPAIR_BENCHMARK_INTERFACE",
    "MUTATION_SCORE_INTERFACE",
    "AuthorityMutationSuite",
    "ContractRepairAdversary",
    "ContractRepairFixedPoint",
    "DcrAdversarialReport",
    "DeterministicRepairBenchmark",
    "RepairSafetyMetrics",
    "evaluate_dcr_adversarial",
    "materialize_adversarial_report",
    "materialize_benchmark",
    "materialize_fixed_point",
    "reach_contract_repair_fixed_point",
    "run_deterministic_repair_benchmark",
    "supersede_legacy_repairs",
]
