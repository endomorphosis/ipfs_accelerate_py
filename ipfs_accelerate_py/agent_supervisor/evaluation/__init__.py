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
from .dcr_canary import (
    AUTO_SAFE_ADMISSION_INTERFACE,
    DETERMINISTIC_REPAIR_POLICY_INTERFACE,
    AutoSafeAdmission,
    RepairExecutionMode,
    materialize_policy_and_canary,
    run_fixture_apply_canary,
)
from .dcr_shadow import (
    REPAIR_SHADOW_REPORT_INTERFACE,
    DeterministicRepairShadowRun,
    compare_shadow_to_truth,
    materialize_shadow_report,
    run_deterministic_repair_shadow,
)

__all__ = [
    "ADVERSARIAL_CONFORMANCE_INTERFACE",
    "CONTRACT_REPAIR_FIXED_POINT_INTERFACE",
    "DETERMINISTIC_REPAIR_BENCHMARK_INTERFACE",
    "MUTATION_SCORE_INTERFACE",
    "REPAIR_SHADOW_REPORT_INTERFACE",
    "AuthorityMutationSuite",
    "ContractRepairAdversary",
    "ContractRepairFixedPoint",
    "DcrAdversarialReport",
    "DeterministicRepairBenchmark",
    "DeterministicRepairShadowRun",
    "AutoSafeAdmission",
    "RepairExecutionMode",
    "AUTO_SAFE_ADMISSION_INTERFACE",
    "DETERMINISTIC_REPAIR_POLICY_INTERFACE",
    "materialize_policy_and_canary",
    "run_fixture_apply_canary",
    "RepairSafetyMetrics",
    "compare_shadow_to_truth",
    "evaluate_dcr_adversarial",
    "materialize_adversarial_report",
    "materialize_benchmark",
    "materialize_fixed_point",
    "materialize_shadow_report",
    "reach_contract_repair_fixed_point",
    "run_deterministic_repair_benchmark",
    "run_deterministic_repair_shadow",
    "supersede_legacy_repairs",
]
