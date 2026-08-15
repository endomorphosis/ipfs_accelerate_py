# LPC-006: Supervisor semantic types and datasets imports

**Task:** LPC-006  
**Goal:** LPC-G010  
**Package scanned:** `ipfs_accelerate_py/agent_supervisor`  
**Datasets package:** `ipfs_datasets_py`  
**Machine-readable companion:** `supervisor_semantics.json`

## Classification vocabulary

Every direct supervisor→datasets import and every duplicate supervisor-side
semantic type is classified as exactly one of:

| Classification | Meaning |
| --- | --- |
| **operational** | Supervisor-owned scheduling, isolation, resource, workflow, storage, or daemon concern. Datasets may be a backend, but logic-family meaning is not redefined. |
| **compatibility** | Alias, spelling, or retained name that preserves an older public surface without independent semantic authority. |
| **duplicate** | Supervisor-local type that redefines or parallel-owns a logic semantic vocabulary datasets is intended to own. |
| **canonical projection** | Import or adapter that loads datasets contracts and projects supervisor values onto them, or consumes datasets types as the authority. |

## Scope

- **Direct imports** include AST-level `from/import ipfs_datasets_py...`
  statements and explicit lazy `importlib` targets that load datasets modules
  at call time.
- String references used only for path scope, documentation provenance,
  `git ls-tree` probes, or environment variable names are **not** imports.
- Homonymous enums that share a name with a logic semantic type but live in a
  different domain (goal quality, prompt workflow, repository surface) are
  inventoried so they are not silently confused with proof-receipt axes.
- In this worktree `ipfs_datasets_py/` is empty; classifications use supervisor
  source and the plan ownership split (datasets owns semantics; supervisor owns
  operations).

## Summary

| Slice | Count |
| --- | ---: |
| Direct imports | 60 |
| Duplicate semantic types | 38 |
| Unclassified | 0 |

| Classification | Imports | Semantic types | Combined |
| --- | ---: | ---: | ---: |
| operational | 8 | 14 | 22 |
| compatibility | 0 | 9 | 9 |
| duplicate | 0 | 13 | 13 |
| canonical projection | 52 | 2 | 54 |

## Direct supervisor→datasets imports

### Static top-level and function-local imports

| ID | Supervisor path | Datasets module | Style | Classification |
| --- | --- | --- | --- | --- |
| imp-001 | `contract_analysis/cache_adapter.py` | `logic.software_contracts.cache` | static top-level | canonical projection |
| imp-002 | `proof/code_contract_logic.py` | `logic.ir_core.claims` | static top-level | canonical projection |
| imp-003 | `proof/code_contract_prover.py` | `logic.backends.cvc5.compiler` | static top-level | canonical projection |
| imp-004 | `proof/code_contract_prover.py` | `logic.backends.registry` | static top-level | canonical projection |
| imp-005 | `proof/code_contract_prover.py` | `logic.backends.z3.compiler` | static top-level | canonical projection |
| imp-006 | `proof/code_contract_prover.py` | `logic.ir_core.claims` | static top-level | canonical projection |
| imp-007 | `proof/code_contract_prover.py` | `logic.ir_core.protocols` | static top-level | canonical projection |
| imp-008 | `proof/code_contract_prover.py` | `logic.backends.registry` | function-local | canonical projection |
| imp-009 | `proof/end_goal_development.py` | `logic.software_verification.tactician.contracts` | try-optional | canonical projection |
| imp-010 | `proof/kernel_verification.py` | `logic.hammers.reconstruction` | function-local | canonical projection |
| imp-011 | `proof/runtime_contract_obligations.py` | `logic.ir_core.claims` | function-local | canonical projection |
| imp-012 | `proof/mcp_contract_obligations.py` | `logic.ir_core.claims` | function-local | canonical projection |
| imp-013 | `proof/program_analysis_zkp.py` | `logic.zkp.ceremony` | function-local | canonical projection |
| imp-014 | `integrations/ipfs_datasets_logic_provider.py` | `utils.symai_config` | function-local | operational |
| imp-015 | `merge/lease_coordination.py` | `logic.profile_g` | function-local | canonical projection |
| imp-016 | `task_sources/dataset_store.py` | `ipfs_datasets` | function-local | operational |
| imp-017 | `task_sources/dataset_store.py` | `dataset_manager` | function-local | operational |
| imp-018 | `todo_daemon/legal_parser_daemon.py` | `logic.deontic.metrics` | static top-level | canonical projection |
| imp-019 | `todo_daemon/legal_parser_daemon.py` | `logic.deontic.exports` | static top-level | canonical projection |
| imp-020 | `todo_daemon/legal_parser_daemon.py` | `logic.deontic.utils.deontic_parser` | static top-level | canonical projection |
| imp-021 | `todo_daemon/legal_parser_daemon.py` | `optimizers.agentic.base` | static top-level | operational |
| imp-022 | `todo_daemon/legal_parser_daemon.py` | `optimizers.common.base_optimizer` | static top-level | operational |
| imp-023 | `todo_daemon/legal_parser_daemon.py` | `llm_router` | function-local | operational |
| imp-024 | `todo_daemon/logic_port.py` | `optimizers.logic_port_daemon` | function-local | operational |

### Lazy importlib boundaries (explicit module targets)

These modules never cold-import datasets; they load pinned module paths only
on an explicit probe, conversion, or invocation.

| ID | Supervisor path | Datasets module | Classification |
| --- | --- | --- | --- |
| imp-025 | `proof/logic_provider_contract.py` | `logic.backends.provider` | canonical projection |
| imp-026 | `proof/canonical_logic_adapter.py` | `logic.families.registry` | canonical projection |
| imp-027 | `proof/canonical_logic_adapter.py` | `logic.families.models` | canonical projection |
| imp-028 | `proof/canonical_logic_adapter.py` | `logic.software_verification.properties` | canonical projection |
| imp-029 | `proof/canonical_logic_adapter.py` | `logic.software_verification.translations` | canonical projection |
| imp-030 | `proof/canonical_logic_adapter.py` | `logic.software_verification.receipts` | canonical projection |
| imp-031 | `proof/canonical_logic_adapter.py` | `logic.backends.cache_protocol` | canonical projection |
| imp-032 | `proof/canonical_logic_adapter.py` | `logic.verification_api` | canonical projection |
| imp-033 | `proof/admissibility_bridge.py` | `logic.admissibility.gate` | canonical projection |
| imp-034 | `proof/admissibility_bridge.py` | `logic.admissibility.profiles` | canonical projection |
| imp-035 | `proof/admissibility_bridge.py` | `logic.admissibility.reasons` | canonical projection |
| imp-036 | `proof/admissibility_bridge.py` | `logic.proof_corpus.store` | canonical projection |
| imp-037 | `proof/admissibility_bridge.py` | `logic.proof_corpus.schemas` | canonical projection |
| imp-038 | `proof/admissibility_bridge.py` | `logic.formalization.compiler` | canonical projection |
| imp-039 | `proof/admissibility_enforcement.py` | `logic.admissibility.receipt` | canonical projection |
| imp-040 | `proof/admissibility_enforcement.py` | `logic.admissibility.service` | canonical projection |
| imp-041 | `proof/admissibility_enforcement.py` | `logic.admissibility.compose` | canonical projection |
| imp-042 | `proof/mcp_contract_attestation.py` | `logic.bridge.zkp_attestation` | canonical projection |
| imp-043 | `integrations/ipfs_datasets_logic_provider.py` | `logic.hammers` | canonical projection |
| imp-044 | `integrations/ipfs_datasets_logic_provider.py` | `logic.ir_core.identity` | canonical projection |
| imp-045 | `integrations/ipfs_datasets_logic_provider.py` | `logic.TDFOL` | canonical projection |
| imp-046 | `integrations/ipfs_datasets_logic_provider.py` | `logic.CEC.native` | canonical projection |
| imp-047 | `integrations/ipfs_datasets_logic_provider.py` | `logic.external_provers.smt` | canonical projection |
| imp-048 | `integrations/ipfs_datasets_tactician_provider.py` | `logic.tactician` | canonical projection |
| imp-049 | `integrations/ipfs_datasets_doctor_logic.py` | `logic.tactician` | canonical projection |
| imp-050 | `integrations/ipfs_datasets_doctor_logic.py` | `logic.hammers` | canonical projection |
| imp-051 | `integrations/ipfs_datasets_doctor_logic.py` | `logic.proof_corpus.store` | canonical projection |
| imp-052 | `integrations/ipfs_datasets_doctor_logic.py` | `logic.ir_core.identity` | canonical projection |
| imp-053 | `integrations/ipfs_datasets_analysis_provider.py` | `logic.intent_ir.graphrag.retrieval` | canonical projection |
| imp-054 | `integrations/ipfs_datasets_analysis_provider.py` | `knowledge_graphs.cypher.ast` | canonical projection |
| imp-055 | `integrations/ipfs_datasets_analysis_provider.py` | `logic.ir_core.identity` | canonical projection |
| imp-056 | `integrations/ipfs_datasets_embedding_provider.py` | `ml.embeddings.embeddings_engine` | operational |
| imp-057 | `integrations/ipfs_datasets_test_certificate_provider.py` | `logic.zkp.test_execution_certificate` | canonical projection |
| imp-058 | `integrations/tactician_hammer_capabilities.py` | `logic.tactician` | canonical projection |
| imp-059 | `integrations/tactician_hammer_capabilities.py` | `logic.hammers.premise_selection` | canonical projection |
| imp-060 | `integrations/tactician_hammer_capabilities.py` | `logic.hammers.reconstruction` | canonical projection |

### Import notes

- **Canonical projection majority:** proof and integration adapters load
  datasets IR, backends, hammers, tactician, admissibility, and verification
  APIs as authority. They do not invent second registries.
- **Operational minority:** dataset store, legal-parser optimizer bases,
  logic-port daemon lifecycle, embedding backend, and symai import isolation
  use datasets as storage/routing/runtime support.
- **No import classified as duplicate or compatibility:** imports either
  consume datasets authority or serve operational wiring. Duplication appears
  in supervisor-local *types*, not in the import edges themselves.

## Duplicate supervisor-side semantic types

### True semantic duplicates (datasets should own meaning)

| ID | Name | Supervisor path | Classification | Notes |
| --- | --- | --- | --- | --- |
| sem-001 | `LogicFamily` | `analysis/analysis_operation_registry.py` | duplicate | Parallel family enum; projected via `_ANALYSIS_FAMILY_TO_CANONICAL` |
| sem-002 | `PropertyKind` | `proof/multi_prover_router.py` | duplicate | Parallel property families; projected via `_PROPERTY_KIND_TO_CANONICAL` |
| sem-005 | `LogicForm` | `proof/logic_translation_validation.py` | duplicate | Parallel form labels; projected via `_LOGIC_FORM_TO_CANONICAL` |
| sem-006 | `TranslationClass` | `proof/logic_translation_validation.py` | duplicate | Parallel preservation kinds; projected via `_TRANSLATION_CLASS_TO_PRESERVATION` |
| sem-008 | `EvidenceKind` (proof) | `proof/formal_verification_contracts.py` | duplicate | Parallel evidence-kind axis for receipts |
| sem-011 | `EvidenceAuthority` (proof) | `proof/formal_verification_contracts.py` | duplicate | Parallel evidence-authority axis for receipts |
| sem-016 | `AssuranceLevel` | `proof/formal_verification_contracts.py` | duplicate | Ordered assurance parallel to datasets receipt ceilings |
| sem-019 | `ProofVerdict` | `proof/formal_verification_contracts.py` | duplicate | Parallel semantic verdict vocabulary |
| sem-022 | `CacheScope` | `analysis/analysis_operation_registry.py` | duplicate | Parallel cache scopes; projected via `_CACHE_SCOPE_TO_CANONICAL` |
| sem-023 | `ProofProviderOperation` | `proof/formal_verification_capabilities.py` | duplicate | Parallel provider operations; facade converts to datasets wire types |
| sem-025 | `ProviderRequest` / `ProviderResponse` / `ResourceBudget` | `proof/formal_verification_provider.py` | duplicate | Parallel provider envelope; converted by `SupervisorLogicProviderFacade` |
| sem-028 | DCEC/TDFOL plan vocabulary | `proof/formal_logic_vocabulary.py` | duplicate | Reviewed plan-check operators parallel datasets DCEC/TDFOL names |
| sem-037 | Translation support enums | `proof/logic_translation_validation.py` | duplicate | Approximation/dimension/issue codes parallel datasets translation contracts |

### Compatibility aliases

| ID | Name | Alias of | Path |
| --- | --- | --- | --- |
| sem-003 | `PropertyType` | `PropertyKind` | `proof/multi_prover_router.py` |
| sem-004 | `ObligationProperty` | `PropertyKind` | `proof/multi_prover_router.py` |
| sem-007 | `TranslationExactness` | `TranslationClass` | `proof/logic_translation_validation.py` |
| sem-009 | `ProofEvidenceKind` | `EvidenceKind` | `proof/formal_verification_contracts.py` |
| sem-010 | `ZKP_ATTESTATION` | `CRYPTOGRAPHIC_ATTESTATION` | `proof/formal_verification_contracts.py` |
| sem-017 | `AssuranceLevel.NONE` | `UNVERIFIED` | `proof/formal_verification_contracts.py` |
| sem-018 | `AssuranceLevel.SOLVER_VERIFIED` | `SOLVER_CHECKED` | `proof/formal_verification_contracts.py` |
| sem-021 | `RouteVerdict` | `PortfolioVerdict` | `proof/multi_prover_router.py` |
| sem-029 | `Sort` | `TermSort` | `proof/formal_logic_vocabulary.py` |

### Operational (name collision or supervisor-owned, not logic authority)

| ID | Name | Path | Rationale |
| --- | --- | --- | --- |
| sem-012 | `EvidenceAuthority` | `objectives/goal_quality.py` | Goal-quality ops vocabulary |
| sem-013 | `EvidenceAuthority` | `prompt/prompt_workflow.py` | Prompt workflow authority |
| sem-014 | `EvidenceAuthority` | `planning/plan_analysis_query_planner.py` | Plan-analysis nomination |
| sem-015 | `EvidenceKind` | `analysis/repository_surface_inventory.py` | Surface inventory discovery |
| sem-020 | `PortfolioVerdict` | `proof/multi_prover_router.py` | Portfolio aggregation outcome |
| sem-024 | `ProofProviderIsolation` | `proof/formal_verification_capabilities.py` | Isolation is supervisor operational ownership |
| sem-026 | Prover matrix types | `proof/prover_matrix_registry.py` | Executable evidence-bound matrix ownership |
| sem-027 | Formal verification cache | `proof/formal_verification_cache.py` | Placement and single-flight ownership |
| sem-032 | `LogicCapabilityBinding` | `proof/change_propagation_obligations.py` | Obligation-to-capability routing |
| sem-033 | `LogicCapabilityBinding` | `proof/contract_repair_obligations.py` | Obligation-to-capability routing |
| sem-034 | `OperationStatus` | `control/control_contracts.py` | Control-plane lifecycle status |
| sem-035 | `ProverRole` | `proof/multi_prover_router.py` | Portfolio lane trust roles |
| sem-036 | `AttemptOutcome` | `proof/multi_prover_router.py` | Per-lane routing outcome |
| sem-038 | `TemporalPropertyKind` | `runtime/runtime_temporal_monitor.py` | Runtime safety properties |

### Canonical projection adapters

| ID | Name | Path | Role |
| --- | --- | --- | --- |
| sem-030 | `VocabularyProjection` / `SupervisorCanonicalLogicAdapter` | `proof/canonical_logic_adapter.py` | Single lossless vocabulary projection boundary |
| sem-031 | `SupervisorLogicProviderFacade` | `proof/logic_provider_contract.py` | Lazy provider wire conversion |

## Ownership reminder (from plan)

| Authority | Owner |
| --- | --- |
| Family, property, profile, notation, evidence, translation, verdict, receipt, cache identity, formalization meaning | `ipfs_datasets_py.logic` |
| Scheduling, isolation, resources, model routing, worktrees, leases, cancellation, workflow | `ipfs_accelerate_py.agent_supervisor` |

The supervisor may decide **when and where** work executes. It must not
redefine **what the work means**. Duplicate types listed above are retained
public surfaces today; LPC-090/LPC-091 replace hand maps and classify leftovers
for migration or explicit retention.

## Follow-on tasks

- **LPC-008** — Compose this slice into `inventory.json` / `INDEX.md` under
  categories `supervisor_import_into_datasets` and
  `duplicate_supervisor_semantic_type`.
- **LPC-090** — Generate supervisor maps from residual projections; stop hand
  lists.
- **LPC-091** — Classify leftover supervisor semantic types
  (`LogicFamily`, `PropertyKind`, `LogicForm`, `TranslationClass`,
  capability/operation/matrix/cache) for migration path or retention.
