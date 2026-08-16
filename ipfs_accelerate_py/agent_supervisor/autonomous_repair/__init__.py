"""Domain-agnostic autonomous repair (no language model).

This package orchestrates deterministic supervisor repair work:

* interface / ORB / MCP-IDL name aliases
* MCP package surface resolution (register_tool anchors)
* IR logic application (intent/legal/security/ui + AST/KG/vector)
* doctor transform receipts (``model_call_count == 0``)
* single-path vs multi-path repair disposition

SCA, SwissKnife GUI, contract-repair, and other programs are **consumers** that
supply work items and optional surface roots — they do not own the engine.
"""

from .capabilities import (
    CAPABILITY_EVIDENCE_RECEIPT_SCHEMA,
    DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE,
    SOLVER_READINESS_INTERFACE,
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    DeterministicRepairCapabilities,
    DeterministicRepairCapabilityProbe,
    LogicModuleRequirement,
    NetworkMode,
    SolverReadiness,
    ToolchainRequirement,
    probe_deterministic_repair_capabilities,
)
from .contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AuthorityStage,
    AutonomousRepairPolicy,
    AutonomousRepairReport,
    DeterministicRepairAuthorityError,
    DeterministicRepairDisposition,
    PostEditValidationReceipt,
    PublicationReceipt,
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
    RepairDisposition,
    RepairEvidenceEnvelope,
    RepairWorkItem,
    ReproofReceipt,
    verify_repair_evidence_envelope,
)
from .edit_plan import (
    AdmittedEditPlan,
    materialize_admitted_edit_plan,
    write_edit_plans,
)
from .engine import (
    AutonomousRepairEngine,
    run_autonomous_repair,
)
from .interface_alias_registry import (
    InterfaceAliasRegistry,
    default_mcp_idl_alias_registry,
)
from .materialize import (
    AutonomousRepairMaterializer,
    MaterializePolicy,
    load_edit_plans,
    materialize_edit_plan_dir,
)
from .mcp_surface_resolution import (
    SurfaceResolutionResult,
    resolve_mcp_surfaces,
)
from .no_llm_policy import (
    DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE,
    DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA,
    NO_LLM_EXECUTION_GUARD_INTERFACE,
    TARGET_REPAIR_RUNTIME,
    DeterministicRepairAuthorityDenied,
    DeterministicRepairAuthorityPolicy,
    NoLlmExecutionDenied,
    NoLlmExecutionGuard,
    RepairAuthorityDecision,
    RepairAuthorityDisposition,
    RepairExecutionRoute,
)
from .root_ownership import (
    REPAIR_ROOT_OWNERSHIP_INTERFACE,
    REPAIR_ROOTS_SCHEMA,
    ROOT_OWNERSHIP_RECEIPT_SCHEMA,
    SUBMODULE_PIN_ADMISSION_INTERFACE,
    SUBMODULE_PIN_RECEIPT_SCHEMA,
    RepairRoot,
    RepairRootOwnership,
    RootBinding,
    RootOwnershipDenied,
    RootOwnershipReceipt,
    SubmodulePinAdmission,
    SubmodulePinReceipt,
)

__all__ = [
    "AUTONOMOUS_REPAIR_INTERFACE",
    "AdmittedEditPlan",
    "AuthorityStage",
    "AutonomousRepairEngine",
    "AutonomousRepairMaterializer",
    "AutonomousRepairPolicy",
    "AutonomousRepairReport",
    "CapabilityReceipt",
    "CapabilityStatus",
    "CapabilityEvidenceReceipt",
    "CAPABILITY_EVIDENCE_RECEIPT_SCHEMA",
    "DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE",
    "DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE",
    "DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA",
    "DeterministicRepairAuthorityDenied",
    "DeterministicRepairAuthorityPolicy",
    "DeterministicRepairCapabilities",
    "DeterministicRepairCapabilityProbe",
    "DeterministicRepairAuthorityError",
    "DeterministicRepairDisposition",
    "InterfaceAliasRegistry",
    "MaterializePolicy",
    "LogicModuleRequirement",
    "NetworkMode",
    "NO_LLM_EXECUTION_GUARD_INTERFACE",
    "NoLlmExecutionDenied",
    "NoLlmExecutionGuard",
    "PostEditValidationReceipt",
    "PublicationReceipt",
    "RepairAdmissionReceipt",
    "RepairAuthorityDecision",
    "RepairAuthorityDisposition",
    "RepairAuthorityRoots",
    "RepairExecutionRoute",
    "RepairDisposition",
    "RepairEvidenceEnvelope",
    "REPAIR_ROOT_OWNERSHIP_INTERFACE",
    "REPAIR_ROOTS_SCHEMA",
    "ROOT_OWNERSHIP_RECEIPT_SCHEMA",
    "SUBMODULE_PIN_ADMISSION_INTERFACE",
    "SUBMODULE_PIN_RECEIPT_SCHEMA",
    "RepairRoot",
    "RepairRootOwnership",
    "RootBinding",
    "RootOwnershipDenied",
    "RootOwnershipReceipt",
    "RepairWorkItem",
    "ReproofReceipt",
    "SOLVER_READINESS_INTERFACE",
    "SolverReadiness",
    "SubmodulePinAdmission",
    "SubmodulePinReceipt",
    "SurfaceResolutionResult",
    "TARGET_REPAIR_RUNTIME",
    "ToolchainRequirement",
    "default_mcp_idl_alias_registry",
    "load_edit_plans",
    "materialize_admitted_edit_plan",
    "materialize_edit_plan_dir",
    "probe_deterministic_repair_capabilities",
    "resolve_mcp_surfaces",
    "run_autonomous_repair",
    "write_edit_plans",
    "verify_repair_evidence_envelope",
]
