# Adversarial Assurance Engine supervisor taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task header prefix `## AAE-`, goal heap `adversarial_assurance_engine.objectives.md`, and board namespace `adversarial-assurance-engine-v1`.

The plan, objective heap, taskboard, scheduler profile, validator, board tests, and prerequisite receipt are operator-owned protected inputs. Workers may not edit them. All mutants and model-assisted work occur in disposable isolated worktrees; missing upstream capability is typed unavailable; held-out evidence and authorization are mandatory for policy promotion.

## Parallel waves

```text
W00 AAE-000
W01 AAE-001 | AAE-002 | AAE-003 | AAE-004
W02 AAE-005
W03 AAE-006 | AAE-007
W04 AAE-008 | AAE-009 | AAE-010 | AAE-011
W05 AAE-012
W06 AAE-013 | AAE-014 | AAE-021 | AAE-026 | AAE-027 | AAE-034
W07 AAE-015 | AAE-016 | AAE-017 | AAE-018 | AAE-019 | AAE-020 | AAE-023 | AAE-035
W08 AAE-022 | AAE-028 | AAE-036 | AAE-037
W09 AAE-024 | AAE-029 | AAE-038
W10 AAE-025 | AAE-039
W11 AAE-030 | AAE-040 | AAE-041
W12 AAE-031 | AAE-032 | AAE-042 | AAE-043
W13 AAE-033 | AAE-044
W14 AAE-045 | AAE-049
W15 AAE-046 | AAE-050 | AAE-051 | AAE-052 | AAE-053 | AAE-054 | AAE-055 | AAE-059
W16 AAE-047
W17 AAE-048 | AAE-060
W18 AAE-056 | AAE-058 | AAE-061
W19 AAE-057 | AAE-062
W20 AAE-063
```

`AAE-006` appears in the logical DAG but begins blocked and unschedulable. It may be marked completed only by an operator after the prerequisite receipt proves terminal SCG, released checkpoint and delta proof-sealer interfaces, exact clean pins, and fresh baselines. Tasks not depending on it may proceed in parallel.

## AAE-000 Seal the supervisor-native Adversarial Assurance Engine program

- Status: completed
- Completion: manual
- Completion evidence: Operator-authored controls and a non-authoritative pre-plan baseline observation in the isolated target controller; AAE-005 owns durable current-tree baseline receipts.
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: control
- Depends on:
- Goal id: AAE-G000
- Outputs: docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md, docs/architecture/adversarial_assurance_engine.objectives.md, docs/architecture/adversarial_assurance_engine.todo.md, config/adversarial_assurance_engine_scheduler.json, scripts/validate_adversarial_assurance_engine_board.py, scripts/ops/agent_supervisor/adversarial_assurance_engine_scheduler.py, test/api/test_adversarial_assurance_engine_board.py
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all && python3 -m pytest -q test/api/test_adversarial_assurance_engine_board.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/control
- Parallel lane: control
- Resource class: cpu-small
- Implementation timeout seconds: 1800
- Provider role: operator-only
- LLM context budget bytes: 1
- Predicted files: docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md, docs/architecture/adversarial_assurance_engine.objectives.md, docs/architecture/adversarial_assurance_engine.todo.md, config/adversarial_assurance_engine_scheduler.json, scripts/validate_adversarial_assurance_engine_board.py, scripts/ops/agent_supervisor/adversarial_assurance_engine_scheduler.py, test/api/test_adversarial_assurance_engine_board.py
- Interfaces: AdversarialAssuranceEnginePlan@1
- Conflict policy: Operator-only protected inputs; workers cannot edit, weaken, rebind, or bypass them.
- Preconditions: Clean controller at accelerate 7c9f3fa3 with datasets fbd1ba9f, kit c7e5feeb, and MCP++ dc316465 initialized gitlinks.
- Effects: Freezes scope, goal graph, task DAG, ownership, resource budgets, launch authority, release gates, and terminal evidence.
- Evidence subset: operator-authored plan, objectives, board, profile, validator, tests, and focused pre-change baselines
- Symbolic first: true
- Acceptance: Validator proves exact task/goal populations, acyclic dependencies, ownership, source bindings, protected paths, initial frontier, prerequisite gate, safety doctrine, and terminal fan-in.

## AAE-001 Inventory accelerate execution, verification, policy, state-machine, and ZK surfaces

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-accelerate
- Depends on: AAE-000
- Goal id: AAE-G010
- Outputs: docs/architecture/adversarial_assurance_inventory/accelerate.json, docs/architecture/adversarial_assurance_inventory/accelerate.md
- Validation: python3 -m json.tool docs/architecture/adversarial_assurance_inventory/accelerate.json >/dev/null
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/inventory-accelerate
- Parallel lane: accelerate-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: docs/architecture/adversarial_assurance_inventory/accelerate.json, docs/architecture/adversarial_assurance_inventory/accelerate.md
- Interfaces: AAEAccelerateInventory@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-000 are complete.
- Effects: Records current reusable accelerate APIs without changing product code.
- Evidence subset: accelerate source, public exports, focused tests, manifests, and benchmark artifacts
- Symbolic first: true
- Acceptance: Inventory binds exact exports, signatures, statuses, manifests, tests, receipts, isolation/resource seams, ZK nonclaims, and known blind spots to the inspected commit.

## AAE-002 Inventory datasets index, capsules, claim analysis, mutation, property, and vacuity assets

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-datasets
- Depends on: AAE-000
- Goal id: AAE-G010
- Outputs: docs/architecture/adversarial_assurance_inventory/datasets.json, docs/architecture/adversarial_assurance_inventory/datasets.md
- Validation: python3 -m json.tool docs/architecture/adversarial_assurance_inventory/datasets.json >/dev/null
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/inventory-datasets
- Parallel lane: datasets-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: docs/architecture/adversarial_assurance_inventory/datasets.json, docs/architecture/adversarial_assurance_inventory/datasets.md
- Interfaces: AAEDatasetsInventory@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-000 are complete.
- Effects: Records current reusable datasets APIs and blind spots without changing product code.
- Evidence subset: datasets semantic index/state/content/mutation/property/proof sources and tests
- Symbolic first: true
- Acceptance: Inventory distinguishes exact/conservative/heuristic/opaque analysis, functional SemanticCapsuleCompiler@1, existing fixtures/fuzzing, and narrow vacuity checks without inventing missing APIs.

## AAE-003 Inventory kit durability, CAS, receipt, campaign-history, and recovery surfaces

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-kit
- Depends on: AAE-000
- Goal id: AAE-G010
- Outputs: docs/architecture/adversarial_assurance_inventory/kit.json, docs/architecture/adversarial_assurance_inventory/kit.md
- Validation: python3 -m json.tool docs/architecture/adversarial_assurance_inventory/kit.json >/dev/null
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/inventory-kit
- Parallel lane: kit-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: docs/architecture/adversarial_assurance_inventory/kit.json, docs/architecture/adversarial_assurance_inventory/kit.md
- Interfaces: AAEKitInventory@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-000 are complete.
- Effects: Records current reusable kit APIs without changing product code.
- Evidence subset: kit DurableCoordinationStore, root adapters, governor/proof stores, tests, and vectors
- Symbolic first: true
- Acceptance: Inventory identifies exact immutable-block, idempotency, CAS, corruption, recovery, history, and proof-store contracts that the AAE domain layer must reuse.

## AAE-004 Inventory MCP++ conformance boundary and proof-sealer release surfaces

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-interoperability
- Depends on: AAE-000
- Goal id: AAE-G010
- Outputs: docs/architecture/adversarial_assurance_inventory/interoperability.json, docs/architecture/adversarial_assurance_inventory/interoperability.md
- Validation: python3 -m json.tool docs/architecture/adversarial_assurance_inventory/interoperability.json >/dev/null
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/inventory-interoperability
- Parallel lane: interop-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: docs/architecture/adversarial_assurance_inventory/interoperability.json, docs/architecture/adversarial_assurance_inventory/interoperability.md
- Interfaces: MCPPlusPlusBoundary@1, IncrementalProofSealerCapabilityProbe@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-000 are complete.
- Effects: Records current shared conformance and moving proof-sealer capability without changing authorities.
- Evidence subset: MCP++ schemas/vectors/tests and exact proof-sealer branch exports
- Symbolic first: true
- Acceptance: Inventory records existing profiles/vectors and probes checkpoint/delta sealer APIs separately; missing surfaces are typed unavailable and no profile is proposed.

## AAE-005 Reconcile authority matrix, manifests, blind spots, and focused baselines

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority-matrix
- Depends on: AAE-001, AAE-002, AAE-003, AAE-004
- Goal id: AAE-G010
- Outputs: docs/architecture/adversarial_assurance_inventory/authority_matrix.json, docs/architecture/adversarial_assurance_inventory/BASELINE.md, docs/architecture/adversarial_assurance_inventory/run_focused_baselines.py, docs/architecture/adversarial_assurance_inventory/baseline_receipts
- Validation: python3 docs/architecture/adversarial_assurance_inventory/run_focused_baselines.py --current-tree && python3 -m json.tool docs/architecture/adversarial_assurance_inventory/authority_matrix.json >/dev/null && python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/authority-matrix
- Parallel lane: integration-inventory
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 196608
- Predicted files: docs/architecture/adversarial_assurance_inventory/authority_matrix.json, docs/architecture/adversarial_assurance_inventory/BASELINE.md, docs/architecture/adversarial_assurance_inventory/run_focused_baselines.py, docs/architecture/adversarial_assurance_inventory/baseline_receipts
- Interfaces: AssuranceAuthorityMatrix@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-001, AAE-002, AAE-003, AAE-004 are complete.
- Effects: Creates the reviewed reuse/non-reimplementation and baseline authority join.
- Evidence subset: AAE-001 through AAE-004 plus exact process receipts
- Symbolic first: true
- Acceptance: Matrix records the exact scoped repository forest, interfaces, status vocabularies, test/proof manifests, known RED/stale/unavailable evidence, and green or explicitly failed focused commands before product implementation. The protected runner supports only `--current-tree` execution and `--verify-bundle` read-only verification with one of the two reviewed output roots; it emits a closed JSON verification report whose exact receipt path/CID bindings are rederived. Each closed baseline receipt binds repository/revision, exact argv, exit code, actual counts, bounded log digest, canonical UTC interval and duration, environment and dependency-lock identities, disabled network, absent production credentials, and canonical identity.

## AAE-006 Operator gate: pin terminal SCG and released IncrementalProofSealer authorities

- Status: blocked
- Blocked reason: SCG and IncrementalProofSealer programs were live and incomplete at planning; required public sealer APIs were absent.
- Completion: manual
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: prerequisite-release
- Depends on: AAE-005
- Goal id: AAE-G060
- Outputs: config/adversarial_assurance_prerequisites.json, docs/architecture/adversarial_assurance_inventory/prerequisite_release.md, docs/architecture/adversarial_assurance_inventory/prerequisite_evidence
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-prerequisites
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/prerequisite-release
- Parallel lane: operator-gate
- Resource class: cpu-small
- Implementation timeout seconds: 1800
- Provider role: operator-only
- LLM context budget bytes: 1
- Predicted files: config/adversarial_assurance_prerequisites.json, docs/architecture/adversarial_assurance_inventory/prerequisite_release.md, docs/architecture/adversarial_assurance_inventory/prerequisite_evidence
- Interfaces: AssurancePrerequisiteReceipt@1
- Conflict policy: Operator-only release authority. Workers cannot edit this task, prerequisite receipt, source binding, protected controls, or substitute local implementations.
- Preconditions: AAE-005 is complete; operator independently verifies terminal upstream receipts and exact released interfaces.
- Effects: When and only when evidence is valid, advances pin_generation, repins source/gitlinks, records copied content-addressed upstream/baseline evidence, and permits a separate signed, exact-HEAD, single-use launch admission; never emulates upstream functionality.
- Evidence subset: terminal SCG receipt, released sealer capability receipt, recursive forest, clean status, focused baseline receipts
- Symbolic first: true
- Acceptance: Only the configured operator did:key may complete this task. Validation recomputes the receipt and every copied evidence CID, verifies the signature over audience/action/receipt/pin bindings, cross-checks the genuine drained SCG lifecycle and terminal receipts, imports the released checkpoint/delta sealer API bindings, executes and parses the upstream sealer's canonical release qualification without hard-coded residual-task counts, verifies the baseline bundle report reproduces the signed receipt bindings, and rejects any self-asserted boolean/string substitute. Because the committed gate cannot self-bind its containing commit, post-gate preflight separately requires a chained operator signature over exact controller HEAD, receipt CID, pin generation, gitlinks, and a strictly increasing single-use launch generation; otherwise the typed blocker remains.

## AAE-007 Define common artifact headers, versions, identities, provenance, and closed vocabularies

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-common
- Depends on: AAE-005
- Goal id: AAE-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/common.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_common.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_common.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/contracts-common
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/common.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_common.py
- Interfaces: AssuranceArtifactHeader@1, closed status and provenance vocabularies
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-005 are complete.
- Effects: Produces AssuranceArtifactHeader@1, closed status and provenance vocabularies with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-007 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: All artifacts bind required repository/state/target/capsule/proof/environment/lock/version/status/provenance/identity fields and reject unknown fields, floats, host fallbacks, and identity mismatch.

## AAE-008 Define operator, target, candidate, policy, and campaign-plan models

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-mutation
- Depends on: AAE-007
- Goal id: AAE-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/mutation_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_mutation_contracts.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_mutation_contracts.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/contracts-mutation
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/mutation_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_mutation_contracts.py
- Interfaces: MutationOperatorDefinition, MutationTarget, MutationCandidate, MutationCampaignPolicy, MutationCampaignPlan
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-007 are complete.
- Effects: Produces MutationOperatorDefinition, MutationTarget, MutationCandidate, MutationCampaignPolicy, MutationCampaignPlan with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-008 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Models enforce every operator declaration, deterministic seed/config binding, bounded counts, risk classes, target prerequisites, sandbox, rollback, and campaign budget.

## AAE-009 Define expected-detection, execution, receipt, outcome, and equivalence models

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-execution
- Depends on: AAE-007
- Goal id: AAE-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/execution_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_execution_contracts.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_execution_contracts.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/contracts-execution
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/execution_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_execution_contracts.py
- Interfaces: ExpectedDetectionSet, MutationExecutionPlan, MutationExecutionReceipt, MutationOutcome, MutationEquivalenceAssessment
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-007 are complete.
- Effects: Produces ExpectedDetectionSet, MutationExecutionPlan, MutationExecutionReceipt, MutationOutcome, MutationEquivalenceAssessment with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-009 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Closed models distinguish predicted/selected/executed/observed detectors and never count invalid, uncompilable, infrastructure, timeout, inconclusive, or equivalent cases as killed.

## AAE-010 Define survivor, gap, vacuity, detection-failure, and adequacy models

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-analysis
- Depends on: AAE-007
- Goal id: AAE-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/analysis_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_analysis_contracts.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_analysis_contracts.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/contracts-analysis
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/analysis_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_analysis_contracts.py
- Interfaces: SurvivingMutantReport, AssuranceGap, VacuityFinding, DetectionFailure, TestAdequacyProfile, ProofAdequacyProfile, PolicyAdequacyProfile, CapsuleAdequacyProfile
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-007 are complete.
- Effects: Produces SurvivingMutantReport, AssuranceGap, VacuityFinding, DetectionFailure, TestAdequacyProfile, ProofAdequacyProfile, PolicyAdequacyProfile, CapsuleAdequacyProfile with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-010 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Gap and adequacy taxonomies are closed; reports bind minimized evidence and every vacuity record states exactly what remains proven.

## AAE-011 Define candidate remediation and evaluation models

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-remediation
- Depends on: AAE-007
- Goal id: AAE-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/remediation_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_remediation_contracts.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_remediation_contracts.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/contracts-remediation
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/remediation_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_remediation_contracts.py
- Interfaces: CandidateTestSpecification, CandidateProofObligation, CandidatePolicyConstraint, CandidateAnalyzerRule, GapRemediationPlan, RemediationEvaluationReport
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-007 are complete.
- Effects: Produces CandidateTestSpecification, CandidateProofObligation, CandidatePolicyConstraint, CandidateAnalyzerRule, GapRemediationPlan, RemediationEvaluationReport with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-011 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Candidate tests bind requirement provenance; proof obligations include assumptions/source connection/nonvacuity; model drafts begin heuristic_candidate; evaluations encode regression and overconstraint.

## AAE-012 Define signed campaign/promotion receipts, canonical schemas, and datasets exports

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-receipts
- Depends on: AAE-007, AAE-008, AAE-009, AAE-010, AAE-011
- Goal id: AAE-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/receipt_contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/schemas, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/__init__.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_receipt_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_public_api.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_receipt_contracts.py ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_public_api.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/contracts-receipts
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/receipt_contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/schemas, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/__init__.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_receipt_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_public_api.py
- Interfaces: AssuranceCampaignReceipt, AssurancePolicyPromotionReceipt, AdversarialAssuranceArtifacts@1
- Conflict policy: This is the sole datasets task allowed to freeze the adversarial_assurance package exports and canonical schemas. Reuse the existing canonical content, receipt, signer, and key-identity authorities; do not define another envelope or signature scheme.
- Preconditions: AAE-007, AAE-008, AAE-009, AAE-010, AAE-011 are complete.
- Effects: Produces signed AssuranceCampaignReceipt and AssurancePolicyPromotionReceipt contracts, schemas, and stable datasets exports with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-012 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Receipts bind complete campaign/promotion inputs, authorization, expected-old revision, held-out result, seal scope, terminal status, signer/key/audience/action identities, signature bytes, signature-verification status, and canonical identity; signed and content-addressed evidence reuses the existing receipt/signature authority and defines no new envelope or cryptography.

## AAE-013 Decide and conditionally qualify the shared schema/vector boundary

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: conformance
- Depends on: AAE-008, AAE-009, AAE-010, AAE-011, AAE-012
- Goal id: AAE-G020
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/adversarial_assurance_conformance.md, ipfs_accelerate_py/mcplusplus/schemas/adversarial_assurance_campaign.schema.json, ipfs_accelerate_py/mcplusplus/schemas/adversarial_assurance_receipt.schema.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/adversarial_assurance_campaign_valid.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/adversarial_assurance_campaign_invalid.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/adversarial_assurance_receipt_valid.json
- Validation: python3 -m pytest -q ipfs_accelerate_py/mcplusplus/tests-py/integration/test_conformance_vectors.py && (cd ipfs_accelerate_py/mcplusplus/tests-go && go test ./...) && cargo test --manifest-path ipfs_accelerate_py/mcplusplus/tests-rs/Cargo.toml && (cd ipfs_accelerate_py/mcplusplus/tests-ts && npm test)
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/conformance
- Parallel lane: interop
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/adversarial_assurance_conformance.md, ipfs_accelerate_py/mcplusplus/schemas/adversarial_assurance_campaign.schema.json, ipfs_accelerate_py/mcplusplus/schemas/adversarial_assurance_receipt.schema.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/adversarial_assurance_campaign_valid.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/adversarial_assurance_campaign_invalid.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/adversarial_assurance_receipt_valid.json
- Output policy: The decision document is mandatory; schema/vector paths are a conditional permission envelope and remain absent when no genuine cross-language contract is demonstrated. Existing flat-vector harnesses are consumed, not rewritten.
- Interfaces: AdversarialAssuranceConformanceDecision@1
- Conflict policy: Own only MCP++ shared schema, flat discoverable vector, conformance-harness, and decision-document files. Consume the frozen datasets contracts; do not edit profiles, runtimes, CID authority, application orchestration, or unrelated conformance data.
- Preconditions: AAE-008, AAE-009, AAE-010, AAE-011, AAE-012 are complete.
- Effects: Produces AdversarialAssuranceConformanceDecision@1 and, only when that decision demonstrates a shared requirement, the minimal shared schemas and flat vectors; the explicit no-shared-requirement branch changes no schema/vector/profile.
- Evidence subset: aae-013 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: The decision may conclude no MCP++ change is justified. If and only if it proves a genuine cross-language requirement, minimal shared schemas and flat canonical vectors are added and reproduced by the existing Python, Go, Rust, and TypeScript harnesses; unknown fields/enums fail closed; no MCP++ profile, runtime, or application payload is created or changed.

## AAE-014 Implement deterministic operator registry and rollback contracts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operator-registry
- Depends on: AAE-008, AAE-012
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/base.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/registry.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_registry.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_registry.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/operator-registry
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/base.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/registry.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_registry.py
- Interfaces: MutationOperatorRegistry@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-008, AAE-012 are complete.
- Effects: Produces MutationOperatorRegistry@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-014 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Registry rejects duplicate/versionless/unbounded operators, canonicalizes declarations, dispatches only supported targets, and produces deterministic rollback records.

## AAE-015 Implement control-flow mutation operators

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operators-control-flow
- Depends on: AAE-014
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/control_flow.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_control_flow.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_control_flow.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/operators-control-flow
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/control_flow.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_control_flow.py
- Interfaces: ControlFlowMutationOperators@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-014 are complete.
- Effects: Produces ControlFlowMutationOperators@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-015 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Bounded operators cover inversion, branch removal/unconditional behavior, boundary shifts, recovery/obligation early return, loop termination, and cancellation with semantic intent and equivalence hints.

## AAE-016 Implement data, schema, and interface-contract mutation operators

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operators-data-interface
- Depends on: AAE-014
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/data_schema_interface.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_data_schema_interface.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_data_schema_interface.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/operators-data-interface
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/data_schema_interface.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_data_schema_interface.py
- Interfaces: DataSchemaInterfaceMutationOperators@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-014 are complete.
- Effects: Produces DataSchemaInterfaceMutationOperators@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-016 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Operators cover required/null/default/order/version/bounds/float/Unicode/schema cases and pre/post/error/exception/version/handler/semantic-result interface cases without arbitrary text edits.

## AAE-017 Implement side-effect, error, and retry mutation operators

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operators-effects-errors
- Depends on: AAE-014
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/effects_errors.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_effects_errors.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_effects_errors.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/operators-effects-errors
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/effects_errors.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_effects_errors.py
- Interfaces: SideEffectErrorRetryMutationOperators@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-014 are complete.
- Effects: Produces SideEffectErrorRetryMutationOperators@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-017 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Operators cover omitted/wrong/early/double/reordered effects, audit/compensation, swallowed or misclassified failures, retry budgets, cancellation, and integrity failure.

## AAE-018 Implement authorization and policy mutation operators

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operators-authorization
- Depends on: AAE-014
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/authorization_policy.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_authorization_policy.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_authorization_policy.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/operators-authorization
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/authorization_policy.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_authorization_policy.py
- Interfaces: AuthorizationPolicyMutationOperators@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-014 are complete.
- Effects: Produces AuthorizationPolicyMutationOperators@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-018 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Operators cover authentication, tenant, attenuation, audience, expiry, revocation, confirmation, stale/default policy, and payment-as-authority with high-risk defaults.

## AAE-019 Implement state-machine, distributed-system, storage, and durability operators

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operators-distributed-storage
- Depends on: AAE-014
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/distributed_storage.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_distributed_storage.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_distributed_storage.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/operators-distributed-storage
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/distributed_storage.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_distributed_storage.py
- Interfaces: DistributedStorageMutationOperators@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-014 are complete.
- Effects: Produces DistributedStorageMutationOperators@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-019 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Operators cover transitions, CAS/fencing/leases/ownership/idempotency/compensation/convergence/proof forests/parents and durable commit/sync/checksum/read-back distinctions.

## AAE-020 Implement test, proof, semantic-compression, and conditional GUI operators

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operators-assurance-compression
- Depends on: AAE-014
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/assurance_compression_gui.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_assurance_compression_gui.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_assurance_compression_gui.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/operators-assurance-compression
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators/assurance_compression_gui.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/operators/test_assurance_compression_gui.py
- Interfaces: AssuranceCompressionGuiMutationOperators@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-014 are complete.
- Effects: Produces AssuranceCompressionGuiMutationOperators@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-020 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Operators cover weak/deleted/skipped tests, fixtures, vacuous/stale/incomplete proofs, capsule/dependency/context omissions, and only canonical GUI action bindings; broad visual mutation is absent.

## AAE-021 Implement claim extraction, mutation targeting, and risk-weighted selection

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: target-selection
- Depends on: AAE-008, AAE-010, AAE-012
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/targets.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/risk.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_targets_and_risk.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_targets_and_risk.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/target-selection
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/targets.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/risk.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_targets_and_risk.py
- Interfaces: identify_asserted_properties, select_mutation_targets, rank_mutation_risk
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-008, AAE-010, AAE-012 are complete.
- Effects: Produces identify_asserted_properties, select_mutation_targets, rank_mutation_risk with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-021 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Selection binds claims to symbols/artifacts and prioritizes security, durability, distributed/proof trust, fan-out, recent change, uncertainty, defects, frequency, and failure cost under explicit bounded sampling.

## AAE-022 Implement deterministic bounded semantic mutation generation

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mutation-generator
- Depends on: AAE-014, AAE-015, AAE-016, AAE-017, AAE-018, AAE-019, AAE-020, AAE-021
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/generator.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_generator.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_generator.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/mutation-generator
- Parallel lane: datasets
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/generator.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_generator.py
- Interfaces: generate_mutation_candidates
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-014, AAE-015, AAE-016, AAE-017, AAE-018, AAE-019, AAE-020, AAE-021 are complete.
- Effects: Produces generate_mutation_candidates with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-022 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Same source root, target, operator, seed, and policy produce byte-identical ordered candidates and IDs; global/per-target/operator budgets are enforced.

## AAE-023 Construct explained expected detection sets from semantic dependencies

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: detection-prediction
- Depends on: AAE-009, AAE-021
- Goal id: AAE-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/detection.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_detection.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_detection.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/detection-prediction
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/detection.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_detection.py
- Interfaces: predict_detection_set
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-009, AAE-021 are complete.
- Effects: Produces predict_detection_set with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-023 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Every detector prediction names violated claim, observation rationale, dependency path, required/optional strength, expected terminal status, and exact detector identity/revision.

## AAE-024 Implement isolated mutant rescan and semantic admission guardrails

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mutant-admission
- Depends on: AAE-008, AAE-009, AAE-022, AAE-023
- Goal id: AAE-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/admission.py, test/api/adversarial_assurance/test_admission.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_admission.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/mutant-admission
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/admission.py, test/api/adversarial_assurance/test_admission.py
- Interfaces: admit_mutation
- Conflict policy: Own admission orchestration only. Reuse WorktreeLifecycleStore, MutationLedger, semantic rescan, canonical identity, and process execution; do not modify their authorities or protected controls.
- Preconditions: AAE-008, AAE-009, AAE-022, AAE-023 are complete.
- Effects: Produces admit_mutation with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-024 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Admission validates a caller-supplied owned disposable worktree, rescans declared changes, blocks verifier/policy/key/oracle edits, parses and structurally validates, rejects trivial invalidity, estimates equivalence, predicts detection, and commits identity; it does not create or destroy worktrees.

## AAE-025 Implement bounded equivalent-mutant analysis

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: equivalence
- Depends on: AAE-009, AAE-024
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/equivalence.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_equivalence.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_equivalence.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/equivalence
- Parallel lane: datasets
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/equivalence.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_equivalence.py
- Interfaces: assess_mutation_equivalence
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-009, AAE-024 are complete.
- Effects: Produces assess_mutation_equivalence with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-025 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Analyzer composes AST/normalized IR, constant propagation, reachability, available symbolic/SMT, bounded behavior, and human-review escalation; unknown never becomes equivalent automatically.

## AAE-026 Implement formal-proof and policy vacuity analysis

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vacuity-formal-policy
- Depends on: AAE-010, AAE-012
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/vacuity_formal_policy.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_vacuity_formal_policy.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_vacuity_formal_policy.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/vacuity-formal-policy
- Parallel lane: datasets
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/vacuity_formal_policy.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_vacuity_formal_policy.py
- Interfaces: analyze_formal_vacuity, analyze_policy_vacuity
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-010, AAE-012 are complete.
- Effects: Produces analyze_formal_vacuity, analyze_policy_vacuity with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-026 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Detects unsatisfiable/unreachable/impossible/unconstrained proof obligations and unreachable/shadowed/dominated/obsolete policy behavior, stating the exact residual property.

## AAE-027 Implement test, ZK, receipt, and seal vacuity analysis

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vacuity-test-zk
- Depends on: AAE-010, AAE-012
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/vacuity_test_zk.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_vacuity_test_zk.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_vacuity_test_zk.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/vacuity-test-zk
- Parallel lane: datasets
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/vacuity_test_zk.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_vacuity_test_zk.py
- Interfaces: analyze_test_vacuity, analyze_zk_receipt_vacuity
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-010, AAE-012 are complete.
- Effects: Produces analyze_test_vacuity, analyze_zk_receipt_vacuity with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-027 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Detects tautology/type-only mocks/skips/path bypass/early-success and unbound fields/roots/environments/keys/required sets/direct-execution overclaim/delta omission with precise nonclaims.

## AAE-028 Compare predicted and observed detectors and classify assurance gaps

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: gap-classification
- Depends on: AAE-009, AAE-010, AAE-023
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/gaps.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_gaps.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_gaps.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/gap-classification
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/gaps.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_gaps.py
- Interfaces: compare_detection_sets, classify_assurance_gap
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-009, AAE-010, AAE-023 are complete.
- Effects: Produces compare_detection_sets, classify_assurance_gap with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-028 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Comparison separates not-selected, not-executed, path-unobserved, weak-property, dependency/capsule omission, unspecified, intentional, equivalence, and unknown causes using the closed gap taxonomy.

## AAE-029 Build test, proof, policy, and capsule adequacy profiles

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adequacy
- Depends on: AAE-010, AAE-028
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/adequacy.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_adequacy.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_adequacy.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/adequacy
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/adequacy.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_adequacy.py
- Interfaces: build_test_adequacy_profile, build_proof_adequacy_profile, build_policy_adequacy_profile, build_capsule_adequacy_profile
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-010, AAE-028 are complete.
- Effects: Produces build_test_adequacy_profile, build_proof_adequacy_profile, build_policy_adequacy_profile, build_capsule_adequacy_profile with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-029 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Profiles bind claims, reachable behavior, detectors, false-assurance evidence, uncertainty, gaps, and scope without converting a score into correctness.

## AAE-030 Diagnose surviving mutants with explicit product versus assurance distinctions

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: survivor-diagnosis
- Depends on: AAE-025, AAE-026, AAE-027, AAE-028, AAE-029
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/diagnosis.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_diagnosis.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_diagnosis.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/survivor-diagnosis
- Parallel lane: datasets
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/diagnosis.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_diagnosis.py
- Interfaces: diagnose_surviving_mutant
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-025, AAE-026, AAE-027, AAE-028, AAE-029 are complete.
- Effects: Produces diagnose_surviving_mutant with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-030 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Diagnosis follows the required nine-step decision path and never labels every survivor a product defect or every difficult case equivalent.

## AAE-031 Specify minimized survivor reports and bounded reproduction evidence

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: survivor-minimization
- Depends on: AAE-030
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/minimization.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_minimization.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_minimization.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/survivor-minimization
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/minimization.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_minimization.py
- Interfaces: build_surviving_mutant_report
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-030 are complete.
- Effects: Produces build_surviving_mutant_report with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-031 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Reports contain the smallest changed region/input, identities, property, detector inventory, behavior delta, spans, dependency path, proof/receipt IDs, command, and risk; logs remain bounded.

## AAE-032 Generate requirement-grounded candidate remediation specifications

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: remediation-specification
- Depends on: AAE-011, AAE-030
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/remediation.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_remediation.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_remediation.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/remediation-specification
- Parallel lane: datasets
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/remediation.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_remediation.py
- Interfaces: propose_gap_remediation
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-011, AAE-030 are complete.
- Effects: Produces propose_gap_remediation with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-032 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Allowed candidate types bind intended behavior and provenance; tests do not merely encode implementation; proof candidates include nonvacuity; model output remains heuristic_candidate.

## AAE-033 Implement deterministic diagnosis, development, and held-out evaluation policy

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: held-out-policy
- Depends on: AAE-011, AAE-012, AAE-032
- Goal id: AAE-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/held_out.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_held_out.py
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_held_out.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/held-out-policy
- Parallel lane: datasets
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/held_out.py, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance/test_held_out.py
- Interfaces: partition_mutants, qualify_remediation_evaluation
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-011, AAE-012, AAE-032 are complete.
- Effects: Produces partition_mutants, qualify_remediation_evaluation with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-033 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Partitions are deterministic and leakage-resistant; evaluation requires unmutated, diagnosis, development, held-out, unrelated, cost, false-positive, overconstraint, regression, and safety evidence.

## AAE-034 Store immutable mutant and campaign artifacts over DurableCoordinationStore

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-artifacts
- Depends on: AAE-007, AAE-008, AAE-009, AAE-010, AAE-011, AAE-012
- Goal id: AAE-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/contracts.py, ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/artifacts.py, ipfs_kit_py/tests/adversarial_assurance_store/test_contracts.py, ipfs_kit_py/tests/adversarial_assurance_store/test_artifacts.py
- Validation: python3 -m pytest -q ipfs_kit_py/tests/adversarial_assurance_store/test_contracts.py ipfs_kit_py/tests/adversarial_assurance_store/test_artifacts.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/storage-artifacts
- Parallel lane: kit
- Resource class: io-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/contracts.py, ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/artifacts.py, ipfs_kit_py/tests/adversarial_assurance_store/test_contracts.py, ipfs_kit_py/tests/adversarial_assurance_store/test_artifacts.py
- Interfaces: AssuranceArtifactStore@1
- Conflict policy: Own artifact projections and storage only; AAE-038 is the sole final package-export owner. Reuse datasets contracts plus the existing DurableCoordinationStore, content identity, receipt signer, and key-identity authorities; define no alternate envelope, CID, signature, or storage authority.
- Preconditions: AAE-007, AAE-008, AAE-009, AAE-010, AAE-011, AAE-012 are complete.
- Effects: Produces AssuranceArtifactStore@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-034 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Typed projections consume the datasets schemas without redefining them; canonical objects and signed receipts are size bounded and signature-verified before persistence, including before the first durable write, content addressing, Merkle inclusion, or seal eligibility, then rederived and signature-verified again on read through the existing durable and signer authorities.

## AAE-035 Persist campaign state, receipts, gaps, and append-only histories

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-campaigns
- Depends on: AAE-034
- Goal id: AAE-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/campaigns.py, ipfs_kit_py/tests/adversarial_assurance_store/test_campaigns.py
- Validation: python3 -m pytest -q ipfs_kit_py/tests/adversarial_assurance_store/test_campaigns.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/storage-campaigns
- Parallel lane: kit
- Resource class: io-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/campaigns.py, ipfs_kit_py/tests/adversarial_assurance_store/test_campaigns.py
- Interfaces: MutationCampaignRepository@1, AssuranceGapRepository@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-034 are complete.
- Effects: Produces MutationCampaignRepository@1, AssuranceGapRepository@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-035 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Operation-id replay is deterministic; transitions are closed; invalid, unknown-key, wrong-audience/action, or unverified signed receipts are rejected before persistence; completed artifacts survive restart; partial and ambiguous execution claims cannot become terminal success.

## AAE-036 Persist benchmark artifacts, Merkle roots, and seal manifests

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-merkle
- Depends on: AAE-034, AAE-035
- Goal id: AAE-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/merkle.py, ipfs_kit_py/tests/adversarial_assurance_store/test_merkle.py
- Validation: python3 -m pytest -q ipfs_kit_py/tests/adversarial_assurance_store/test_merkle.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/storage-merkle
- Parallel lane: kit
- Resource class: io-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/merkle.py, ipfs_kit_py/tests/adversarial_assurance_store/test_merkle.py
- Interfaces: AssuranceCampaignMerkleRepository@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-034, AAE-035 are complete.
- Effects: Produces AssuranceCampaignMerkleRepository@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-036 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Deterministic roots commit operator/policy/admitted/detection/outcome/survivor/vacuity/held-out sets with required-set completeness and explicit seal availability/status; signature verification occurs before persistence, Merkle inclusion, or seal input, so no invalid or not-yet-verified signed receipt can enter a manifest.

## AAE-037 Implement assurance-policy revision and promotion compare-and-swap

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-policy-cas
- Depends on: AAE-012, AAE-034, AAE-035
- Goal id: AAE-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/policy.py, ipfs_kit_py/tests/adversarial_assurance_store/test_policy.py
- Validation: python3 -m pytest -q ipfs_kit_py/tests/adversarial_assurance_store/test_policy.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/storage-policy-cas
- Parallel lane: kit
- Resource class: io-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/policy.py, ipfs_kit_py/tests/adversarial_assurance_store/test_policy.py
- Interfaces: AssurancePolicyRepository@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-012, AAE-034, AAE-035 are complete.
- Effects: Produces AssurancePolicyRepository@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-037 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Promotion requires exact candidate/evaluation/authorization identities and expected-old revision; stale or concurrent writers fail without overwriting newer policy.

## AAE-038 Implement crash recovery, idempotent replay, and concurrency fencing

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-recovery
- Depends on: AAE-034, AAE-035, AAE-036, AAE-037
- Goal id: AAE-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/recovery.py, ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/__init__.py, ipfs_kit_py/tests/adversarial_assurance_store/test_recovery.py, ipfs_kit_py/tests/adversarial_assurance_store/test_public_api.py
- Validation: python3 -m pytest -q ipfs_kit_py/tests/adversarial_assurance_store/test_recovery.py ipfs_kit_py/tests/adversarial_assurance_store/test_public_api.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/storage-recovery
- Parallel lane: kit
- Resource class: io-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/recovery.py, ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store/__init__.py, ipfs_kit_py/tests/adversarial_assurance_store/test_recovery.py, ipfs_kit_py/tests/adversarial_assurance_store/test_public_api.py
- Interfaces: recover_assurance_campaigns, AssuranceRecoveryReport@1, ipfs_kit_py.adversarial_assurance_store
- Conflict policy: This is the sole final kit package-export owner after AAE-034 through AAE-037 exist. Reuse canonical authorities and export their public contracts without redefining identity, receipt, signature, CAS, or storage behavior.
- Preconditions: AAE-034, AAE-035, AAE-036, AAE-037 are complete.
- Effects: Produces recover_assurance_campaigns, AssuranceRecoveryReport@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-038 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Injected interruptions at every required persistence/CAS boundary resume safely, preserve immutable completions, reject ambiguity, avoid partial promotion, and prevent stale writers; the final package exports artifact, campaign, Merkle, policy-CAS, and recovery interfaces with import and negative tests.

## AAE-039 Bind released canonical authorities and create assurance manifests

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-adapters
- Depends on: AAE-006, AAE-013, AAE-038
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/adapters.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/manifest.py, test/api/adversarial_assurance/test_manifest.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_manifest.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/runtime-adapters
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/adapters.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/manifest.py, test/api/adversarial_assurance/test_manifest.py
- Interfaces: create_assurance_manifest
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-006 release receipt, AAE-013 contracts, and AAE-038 durable recovery are complete.
- Effects: Produces create_assurance_manifest with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-039 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Adapters bind exact released index/capsule/context/verification/policy/state/storage/sealer interfaces and status mappings; missing or drifted authority remains typed unavailable.

## AAE-040 Compose risk-weighted campaign planning, generation, and detector prediction

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: campaign-planning
- Depends on: AAE-024, AAE-039
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/planning.py, test/api/adversarial_assurance/test_planning.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_planning.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/campaign-planning
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/planning.py, test/api/adversarial_assurance/test_planning.py
- Interfaces: plan_mutation_campaign, generate_mutation_candidates, predict_detection_set
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-024, AAE-039 are complete.
- Effects: Produces plan_mutation_campaign, generate_mutation_candidates, predict_detection_set with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-040 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Planning establishes baseline requirements, budgets risk-weighted targets, preserves deterministic identities/partitions, and composes canonical semantic generation/detector explanations.

## AAE-041 Implement disposable Git worktree mutation executor and admission pipeline

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: worktree-execution
- Depends on: AAE-024, AAE-039
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/worktrees.py, test/api/adversarial_assurance/test_worktrees.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_worktrees.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/worktree-execution
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/worktrees.py, test/api/adversarial_assurance/test_worktrees.py
- Interfaces: IsolatedMutationWorktreeExecutor@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-024, AAE-039 are complete.
- Effects: Produces IsolatedMutationWorktreeExecutor@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-041 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: This is the sole mutation-worktree lifecycle owner: it creates and destroys disposable owned worktrees, while AAE-024 only validates a caller-supplied worktree. Mutations never touch production trees/branches, escape owned roots, access credentials/network, or alter undeclared authority; cleanup is fenced and recoverable.

## AAE-042 Implement parallel mutation workers, resource admission, timeout, and cancellation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mutation-workers
- Depends on: AAE-041
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/workers.py, test/api/adversarial_assurance/test_workers.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_workers.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/mutation-workers
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/workers.py, test/api/adversarial_assurance/test_workers.py
- Interfaces: MutationWorkerPool@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-041 are complete.
- Effects: Produces MutationWorkerPool@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-042 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Workers reuse ResourceScheduler and process-tree cancellation, enforce concurrency/budgets/network policy, record infrastructure separately, and remain restartable and leak free.

## AAE-043 Integrate incremental invalidation, verification cache, temporary proof forests, and cost accounting

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: incremental-verification
- Depends on: AAE-006, AAE-039, AAE-040
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/incremental.py, test/api/adversarial_assurance/test_incremental.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_incremental.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/incremental-verification
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/incremental.py, test/api/adversarial_assurance/test_incremental.py
- Interfaces: IncrementalMutationVerifier@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-006, AAE-039, AAE-040 are complete.
- Effects: Produces IncrementalMutationVerifier@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-043 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Only affected units invalidate; reuse requires complete keys; survivors broaden by policy; temporary forests never replace canonical seals; full and incremental costs and cache reuse are measured.

## AAE-044 Execute individual mutants and classify closed outcomes

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mutation-execution
- Depends on: AAE-040, AAE-041, AAE-042, AAE-043
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/execution.py, test/api/adversarial_assurance/test_execution.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_execution.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/mutation-execution
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/execution.py, test/api/adversarial_assurance/test_execution.py
- Interfaces: execute_mutation, classify_mutation_outcome
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-040, AAE-041, AAE-042, AAE-043 are complete.
- Effects: Produces execute_mutation, classify_mutation_outcome with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-044 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Unmutated baseline is green or explicitly blocked; predicted checks run first; broader fallback is policy-bound; observed detectors and one closed terminal outcome are persisted honestly.

## AAE-045 Orchestrate bounded counterexample minimization and survivor diagnosis

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: diagnosis-orchestration
- Depends on: AAE-031, AAE-044
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/diagnosis.py, test/api/adversarial_assurance/test_diagnosis_runtime.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_diagnosis_runtime.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/diagnosis-orchestration
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/diagnosis.py, test/api/adversarial_assurance/test_diagnosis_runtime.py
- Interfaces: diagnose_surviving_mutant
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-031, AAE-044 are complete.
- Effects: Produces diagnose_surviving_mutant with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-045 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Existing CounterexampleMinimizer and semantic diagnostics produce bounded reproductions; minimization failure is explicit; every high-risk survivor always persists an AssuranceGap, with human review accompanying an unknown gap rather than replacing it.

## AAE-046 Generate candidates and execute held-out remediation evaluation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: remediation-evaluation
- Depends on: AAE-032, AAE-033, AAE-045
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/remediation.py, test/api/adversarial_assurance/test_remediation_runtime.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_remediation_runtime.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/remediation-evaluation
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/remediation.py, test/api/adversarial_assurance/test_remediation_runtime.py
- Interfaces: propose_gap_remediation, evaluate_remediation
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-032, AAE-033, AAE-045 are complete.
- Effects: Produces propose_gap_remediation, evaluate_remediation with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-046 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Evaluation covers original, diagnosis, development, held-out, unrelated, performance, false-positive, overconstraint, and safety behavior; one-mutant overfit and mock bypass are rejected.

## AAE-047 Orchestrate authorized assurance-policy promotion, CAS, and new seal

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: policy-promotion
- Depends on: AAE-037, AAE-046
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/promotion.py, test/api/adversarial_assurance/test_promotion.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_promotion.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/policy-promotion
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/promotion.py, test/api/adversarial_assurance/test_promotion.py
- Interfaces: promote_assurance_policy
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-037, AAE-046 are complete.
- Effects: Produces promote_assurance_policy with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-047 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Canonical candidate, held-out pass, no regression/vacuity, declared cost/coverage, authorization, verified signer/key/audience/action bindings on campaign and promotion receipts, expected-old CAS, and released incremental seal are mandatory; candidates cannot self-promote.

## AAE-048 Compose the complete public Python campaign API

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: public-api
- Depends on: AAE-040, AAE-044, AAE-045, AAE-046, AAE-047
- Goal id: AAE-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/api.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/__init__.py, test/api/adversarial_assurance/test_api.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_api.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/public-api
- Parallel lane: accelerate
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/api.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/__init__.py, test/api/adversarial_assurance/test_api.py
- Interfaces: all required AAE APIs including execute_mutation_campaign
- Conflict policy: This is the sole task allowed to edit the AAE package export file. It composes existing modules without changing their semantics or canonical upstream exports.
- Preconditions: AAE-040, AAE-044, AAE-045, AAE-046, AAE-047 are complete.
- Effects: Produces all required AAE APIs including execute_mutation_campaign with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-048 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Every requested API has stable typed inputs/outputs, exact canonical bindings, safe errors, no arbitrary path exposure, and end-to-end contract tests.

## AAE-049 Create deterministic fixture corpus, requirement oracles, and held-out partitions

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: fixture-corpus
- Depends on: AAE-013, AAE-024, AAE-033, AAE-038
- Goal id: AAE-G070
- Outputs: test/fixtures/adversarial_assurance/manifest.json, test/fixtures/adversarial_assurance/schemas, test/api/adversarial_assurance/test_fixture_manifest.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_fixture_manifest.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/fixture-corpus
- Parallel lane: fixtures
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: test/fixtures/adversarial_assurance/manifest.json, test/fixtures/adversarial_assurance/schemas, test/api/adversarial_assurance/test_fixture_manifest.py
- Interfaces: AssuranceFixtureCorpus@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-013, AAE-024, AAE-033, AAE-038 are complete.
- Effects: Produces AssuranceFixtureCorpus@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-049 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Every fixture binds intended requirement/provenance, risk, operator, expected detector, bounded oracle, diagnosis/development/held-out partition, and deterministic identity with no partition leakage.

## AAE-050 Implement controlled security mutations 1 through 10

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: campaign-security-a
- Depends on: AAE-049, AAE-044, AAE-045
- Goal id: AAE-G070
- Outputs: test/fixtures/adversarial_assurance/security_a, test/api/adversarial_assurance/test_security_campaign_a.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_security_campaign_a.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/campaign-security-a
- Parallel lane: fixtures
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/fixtures/adversarial_assurance/security_a, test/api/adversarial_assurance/test_security_campaign_a.py
- Interfaces: SecurityAssuranceCampaignA@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-049, AAE-044, AAE-045 are complete.
- Effects: Produces SecurityAssuranceCampaignA@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-050 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Fixtures cover authentication bypass, caller tenant, attenuation, expired/revoked delegation, missing/replayed confirmation, default allow, payment authority, and stale fencing with expected mechanisms.

## AAE-051 Implement controlled security mutations 11 through 20

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: campaign-security-b
- Depends on: AAE-049, AAE-044, AAE-045
- Goal id: AAE-G070
- Outputs: test/fixtures/adversarial_assurance/security_b, test/api/adversarial_assurance/test_security_campaign_b.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_security_campaign_b.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/campaign-security-b
- Parallel lane: fixtures
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/fixtures/adversarial_assurance/security_b, test/api/adversarial_assurance/test_security_campaign_b.py
- Interfaces: SecurityAssuranceCampaignB@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-049, AAE-044, AAE-045 are complete.
- Effects: Produces SecurityAssuranceCampaignB@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-051 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Fixtures cover retry double execution, uncompensated partial mutation, provider-ack storage, early receipt, invalid signature, pseudo-CID, stale receipt, omitted unit, unknown prover pass, and simulated evidence.

## AAE-052 Implement the required semantic-compression mutation campaign

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: campaign-compression
- Depends on: AAE-049, AAE-044, AAE-045
- Goal id: AAE-G070
- Outputs: test/fixtures/adversarial_assurance/semantic_compression, test/api/adversarial_assurance/test_semantic_compression_campaign.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_semantic_compression_campaign.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/campaign-compression
- Parallel lane: fixtures
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/fixtures/adversarial_assurance/semantic_compression, test/api/adversarial_assurance/test_semantic_compression_campaign.py
- Interfaces: SemanticCompressionAssuranceCampaign@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-049, AAE-044, AAE-045 are complete.
- Effects: Produces SemanticCompressionAssuranceCampaign@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-052 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: All eight required dependency/exception/fixture/stale/heuristic/opaque/selection/expanded-context cases run and produce SCG calibration evidence without automatic production policy change.

## AAE-053 Implement the required ZK and incremental-seal mutation campaign

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: campaign-zk-seal
- Depends on: AAE-006, AAE-049, AAE-044, AAE-045
- Goal id: AAE-G070
- Outputs: test/fixtures/adversarial_assurance/zk_incremental_seal, test/api/adversarial_assurance/test_zk_seal_campaign.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_zk_seal_campaign.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/campaign-zk-seal
- Parallel lane: fixtures
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/fixtures/adversarial_assurance/zk_incremental_seal, test/api/adversarial_assurance/test_zk_seal_campaign.py
- Interfaces: ZKIncrementalSealAssuranceCampaign@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-006, AAE-049, AAE-044, AAE-045 are complete.
- Effects: Produces ZKIncrementalSealAssuranceCampaign@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-053 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: All twelve required receipt/unit/root/environment/key/statement/parent/test/simulation/child/order/replay cases run; every controlled critical case is rejected by the released sealer.

## AAE-054 Implement distributed-state, storage-durability, and crash mutation campaigns

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: campaign-distributed-crash
- Depends on: AAE-049, AAE-044, AAE-045
- Goal id: AAE-G070
- Outputs: test/fixtures/adversarial_assurance/distributed_storage_crash, test/api/adversarial_assurance/test_distributed_storage_crash_campaign.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_distributed_storage_crash_campaign.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/campaign-distributed-crash
- Parallel lane: fixtures
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/fixtures/adversarial_assurance/distributed_storage_crash, test/api/adversarial_assurance/test_distributed_storage_crash_campaign.py
- Interfaces: DistributedStorageCrashAssuranceCampaign@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-049, AAE-044, AAE-045 are complete.
- Effects: Produces DistributedStorageCrashAssuranceCampaign@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-054 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Campaign covers transitions, CAS, fencing, owners, leases, idempotency, compensation, durable acknowledgement/read-back, and every required injected crash boundary.

## AAE-055 Implement vacuity, test/proof/policy, and conditional GUI action-binding campaigns

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: campaign-vacuity-gui
- Depends on: AAE-026, AAE-027, AAE-049, AAE-044, AAE-045
- Goal id: AAE-G070
- Outputs: test/fixtures/adversarial_assurance/vacuity_gui, test/api/adversarial_assurance/test_vacuity_gui_campaign.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_vacuity_gui_campaign.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/campaign-vacuity-gui
- Parallel lane: fixtures
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/fixtures/adversarial_assurance/vacuity_gui, test/api/adversarial_assurance/test_vacuity_gui_campaign.py
- Interfaces: VacuityAndActionBindingCampaign@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-026, AAE-027, AAE-049, AAE-044, AAE-045 are complete.
- Effects: Produces VacuityAndActionBindingCampaign@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-055 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Controlled formal/policy/test/ZK vacuity cases state residual proof; canonical GUI fixtures cover action binding/accessibility only when available, with visual mutation explicitly excluded.

## AAE-056 Add mutate plan, run, target, explain, and report CLI commands

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: cli-campaign
- Depends on: AAE-048
- Goal id: AAE-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli_campaign.py, ipfs_accelerate_py/cli.py, test/api/adversarial_assurance/test_cli_campaign.py, test/api/adversarial_assurance/test_cli_host.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_cli_campaign.py test/api/adversarial_assurance/test_cli_host.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/cli-campaign
- Parallel lane: accelerate
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli_campaign.py, ipfs_accelerate_py/cli.py, test/api/adversarial_assurance/test_cli_campaign.py, test/api/adversarial_assurance/test_cli_host.py
- Interfaces: assurance mutate plan|run|target|explain, assurance report
- Conflict policy: This is the sole task allowed to register the parser-only `assurance` group in the existing `ipfs-accelerate` CLI host. Dispatch remains lazy and all product logic stays in the AAE API.
- Preconditions: AAE-048 are complete.
- Effects: Produces assurance mutate plan|run|target|explain, assurance report with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-056 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: The installed `ipfs-accelerate assurance ...` host reaches mutate plan/run/target/explain and report, exposes typed APIs, never arbitrary external repositories or paths, prints bounded deterministic JSON/human output, honors cancellation/resources, and requires explicit run authority.

## AAE-057 Add gaps, vacuity, remediate, evaluate, promote, and benchmark CLI commands

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: cli-assurance
- Depends on: AAE-047, AAE-048, AAE-056
- Goal id: AAE-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli_assurance.py, test/api/adversarial_assurance/test_cli_assurance.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_cli_assurance.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/cli-assurance
- Parallel lane: accelerate
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli_assurance.py, test/api/adversarial_assurance/test_cli_assurance.py
- Interfaces: assurance gaps|vacuity|remediate|evaluate-remediation|promote|benchmark
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-047, AAE-048, AAE-056 are complete.
- Effects: Produces assurance gaps|vacuity|remediate|evaluate-remediation|promote|benchmark with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-057 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Commands preserve candidate versus authority status, require authorization for promotion, expose no arbitrary path/network service, and return honest unavailable/inconclusive results.

## AAE-058 Implement disjoint campaign metrics, economics, and report builders

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: metrics-reporting
- Depends on: AAE-035, AAE-044, AAE-046, AAE-048
- Goal id: AAE-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/metrics.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/reporting.py, test/api/adversarial_assurance/test_metrics_reporting.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_metrics_reporting.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/metrics-reporting
- Parallel lane: accelerate
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- LLM context budget bytes: 131072
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/metrics.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/reporting.py, test/api/adversarial_assurance/test_metrics_reporting.py
- Interfaces: AssuranceMetrics@1, build_assurance_report
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-035, AAE-044, AAE-046, AAE-048 are complete.
- Effects: Produces AssuranceMetrics@1, build_assurance_report with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-058 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Mutation coverage, detection quality, gap, remediation, and economics populations are disjoint and reproducible; denominators exclude invalid/equivalent infrastructure cases as specified.

## AAE-059 Qualify sandbox, credential, network, authority, path, and instruction-isolation security

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: security-e2e
- Depends on: AAE-041, AAE-042, AAE-049
- Goal id: AAE-G080
- Outputs: test/api/adversarial_assurance/test_security_e2e.py, test/fixtures/adversarial_assurance/security
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_security_e2e.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/security-e2e
- Parallel lane: integration
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/api/adversarial_assurance/test_security_e2e.py, test/fixtures/adversarial_assurance/security
- Interfaces: AAESecurityQualification@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-041, AAE-042, AAE-049 are complete.
- Effects: Produces AAESecurityQualification@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-059 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Mutants cannot escape disposable roots, access production credentials/network, edit verifier/policy/keys/oracles, self-promote, treat comments as policy, or expose arbitrary paths.

## AAE-060 Qualify crash recovery, deterministic replay, cancellation, and concurrent stale writers

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: crash-concurrency-e2e
- Depends on: AAE-038, AAE-042, AAE-044, AAE-047, AAE-049
- Goal id: AAE-G080
- Outputs: test/api/adversarial_assurance/test_crash_concurrency_e2e.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_crash_concurrency_e2e.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/crash-concurrency-e2e
- Parallel lane: integration
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/api/adversarial_assurance/test_crash_concurrency_e2e.py
- Interfaces: AAECrashConcurrencyQualification@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-038, AAE-042, AAE-044, AAE-047, AAE-049 are complete.
- Effects: Produces AAECrashConcurrencyQualification@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-060 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: All ten required crash points restart safely; completed immutable evidence survives; ambiguous or partial claims fail; CAS permits one current writer; worktrees/processes are fenced.

## AAE-061 Qualify held-out remediation, controlled promotion, and initial success targets

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: promotion-e2e
- Depends on: AAE-050, AAE-051, AAE-052, AAE-053, AAE-054, AAE-055, AAE-046, AAE-047, AAE-059, AAE-060
- Goal id: AAE-G080
- Outputs: test/api/adversarial_assurance/test_remediation_promotion_e2e.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_remediation_promotion_e2e.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/promotion-e2e
- Parallel lane: integration
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: test/api/adversarial_assurance/test_remediation_promotion_e2e.py
- Interfaces: AAERemediationPromotionQualification@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-050, AAE-051, AAE-052, AAE-053, AAE-054, AAE-055, AAE-046, AAE-047, AAE-059, AAE-060 are complete.
- Effects: Produces AAERemediationPromotionQualification@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-061 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Every promoted candidate has held-out evidence, no critical regression/new vacuity, cost/coverage, authorization, CAS, and seal; unmet zero/90/50-percent targets remain explicit results, not fabricated passes.

## AAE-062 Seal campaigns, benchmark incremental economics, and emit SCG calibration evidence

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: sealing-benchmark
- Depends on: AAE-006, AAE-036, AAE-043, AAE-052, AAE-053, AAE-058, AAE-061
- Goal id: AAE-G080
- Outputs: benchmarks/agent_supervisor/adversarial_assurance.py, artifacts/agent_supervisor/adversarial_assurance/benchmark.json, artifacts/agent_supervisor/adversarial_assurance/campaign_receipt.json, artifacts/agent_supervisor/adversarial_assurance/scg_calibration.json, test/api/adversarial_assurance/test_benchmark_sealing.py
- Validation: python3 -m pytest -q test/api/adversarial_assurance/test_benchmark_sealing.py && python3 benchmarks/agent_supervisor/adversarial_assurance.py --output artifacts/agent_supervisor/adversarial_assurance/benchmark.json
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/sealing-benchmark
- Parallel lane: integration
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- LLM context budget bytes: 131072
- Predicted files: benchmarks/agent_supervisor/adversarial_assurance.py, artifacts/agent_supervisor/adversarial_assurance/benchmark.json, artifacts/agent_supervisor/adversarial_assurance/campaign_receipt.json, artifacts/agent_supervisor/adversarial_assurance/scg_calibration.json, test/api/adversarial_assurance/test_benchmark_sealing.py
- Interfaces: AssuranceBenchmarkReport@1, AssuranceCampaignSeal@1
- Conflict policy: Own only the listed files and tests; do not edit protected controls or shared package exports. Reuse canonical authorities and fail closed on missing capability.
- Preconditions: AAE-006, AAE-036, AAE-043, AAE-052, AAE-053, AAE-058, AAE-061 are complete.
- Effects: Produces AssuranceBenchmarkReport@1, AssuranceCampaignSeal@1 with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-062 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Benchmark reports actual counts, detector rates, cache reuse, full/incremental cost and savings, model economics, gap/remediation cost; the released signer authority signs the content-addressed campaign receipt, invalid/unverified signatures are rejected before persistence or seal input, and signature verification is tested; the seal commits every declared artifact and SCG evidence is non-authoritative.

## AAE-063 Publish trust model, limitations, current-tree qualification, and final report

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: final-qualification
- Depends on: AAE-056, AAE-057, AAE-058, AAE-061, AAE-062
- Goal id: AAE-G090
- Outputs: docs/guides/adversarial_assurance_engine.md, docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md, test/api/adversarial_assurance/test_current_tree_conformance.py
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all && python3 -m pytest -q test/api/adversarial_assurance ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance ipfs_kit_py/tests/adversarial_assurance_store && python3 docs/architecture/adversarial_assurance_inventory/run_focused_baselines.py --current-tree && python3 -m pytest -q ipfs_accelerate_py/mcplusplus/tests-py/integration/test_conformance_vectors.py && (cd ipfs_accelerate_py/mcplusplus/tests-go && go test ./...) && cargo test --manifest-path ipfs_accelerate_py/mcplusplus/tests-rs/Cargo.toml && (cd ipfs_accelerate_py/mcplusplus/tests-ts && npm test) && python3 -m pytest -q test/api/adversarial_assurance/test_zk_seal_campaign.py test/api/adversarial_assurance/test_current_tree_conformance.py
- Board namespace: adversarial-assurance-engine-v1
- Bundle: adversarial-assurance/final-qualification
- Parallel lane: integration
- Resource class: cpu-large
- Implementation timeout seconds: 21600
- LLM context budget bytes: 262144
- Predicted files: docs/guides/adversarial_assurance_engine.md, docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md, test/api/adversarial_assurance/test_current_tree_conformance.py
- Interfaces: AdversarialAssuranceEngine current-tree qualification
- Conflict policy: Terminal integration may edit only final documentation and conformance test. It must not repair failures by weakening tests, policies, outcomes, manifests, proof scope, or protected controls.
- Preconditions: AAE-056, AAE-057, AAE-058, AAE-061, AAE-062 are complete.
- Effects: Produces AdversarialAssuranceEngine current-tree qualification with bounded current-tree evidence and no production policy change.
- Evidence subset: aae-063 current-tree source, focused tests, and negative cases
- Symbolic first: true
- Acceptance: Report contains exact commits/reuse/operators/counts/scores/survivors/vacuity/gaps/detection/cost/cache/remediation/promotion/regression/seal/improvement/limits/next steps and only the prescribed bounded final claim; terminal evidence reruns the focused pre-change matrix, all four MCP++ conformance harnesses, released proof-sealer/ZK qualification, and unrelated behavior checks.
