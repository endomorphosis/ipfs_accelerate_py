# Semantic Compression Governor supervisor taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `SCG-`,
goal heap `semantic_compression_governor.objectives.md`, and board namespace
`semantic-compression-governor-v1`.

The plan, objective heap, taskboard, scheduler profile, validator, and board
tests are operator-owned protected inputs. Workers may not edit them. All model
work occurs in isolated supervisor worktrees. Expanded evaluations are never
accepted automatically, policies require held-out evaluation and authorization,
and missing canonical upstream capability is typed unavailable rather than
reimplemented.

## Parallel waves

```text
W0  SCG-000
W1  SCG-001 | SCG-002 | SCG-003 | SCG-004
W2  SCG-005
W3  SCG-006 | SCG-040
W4  SCG-007 | SCG-008 | SCG-009 | SCG-010
W5  SCG-011 | SCG-013 | SCG-016 | SCG-019
W6  SCG-012 | SCG-014 | SCG-020 | SCG-021
W7  SCG-015 | SCG-022
W8  SCG-017
W9  SCG-018
W10 SCG-023 | SCG-041
W11 SCG-024 | SCG-028
W12 SCG-025
W13 SCG-026
W14 SCG-027
W15 SCG-029
W16 SCG-030
W17 SCG-031
W18 SCG-032
W19 SCG-033 | SCG-038 | SCG-042 | SCG-043
W20 SCG-035
W21 SCG-034
W22 SCG-036
W23 SCG-039
W24 SCG-037
W25 SCG-044
W26 SCG-045
W27 SCG-046 | SCG-047
W28 SCG-048
```

## SCG-000 Seal the supervisor-native governor program

- Status: completed
- Completion: manual
- Completion evidence: Operator-authored plan, objective graph, board, scheduler profile, validator, and board tests committed on the isolated target branch after focused baseline tests.
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: control
- Depends on:
- Goal id: SCG-G000
- Outputs: docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md, docs/architecture/semantic_compression_governor.objectives.md, docs/architecture/semantic_compression_governor.todo.md, config/semantic_compression_governor_scheduler.json, scripts/validate_semantic_compression_governor_board.py, scripts/ops/agent_supervisor/semantic_compression_governor_scheduler.py, scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py, test/api/test_semantic_compression_governor_board.py
- Validation: python3 scripts/validate_semantic_compression_governor_board.py --check-all && python3 -m pytest -q test/api/test_semantic_compression_governor_board.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/control
- Parallel lane: control
- Resource class: cpu-small
- Implementation timeout seconds: 1800
- Provider role: operator-only
- Context budget tokens: 0
- LLM context budget bytes: 0
- Predicted files: docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md, docs/architecture/semantic_compression_governor.objectives.md, docs/architecture/semantic_compression_governor.todo.md, config/semantic_compression_governor_scheduler.json, scripts/validate_semantic_compression_governor_board.py, scripts/ops/agent_supervisor/semantic_compression_governor_scheduler.py, scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py, test/api/test_semantic_compression_governor_board.py
- Interfaces: SemanticCompressionGovernorPlan@1
- Conflict policy: Operator-only. Workers cannot edit, weaken, rebind, or bypass protected control artifacts.
- Preconditions: Clean controller at accelerate dfd92b554 with exact datasets 1330038f, kit df2f9cc0, and MCP++ dc316465 gitlinks.
- Effects: Freezes scope, trust boundaries, dependency graph, ownership, resource budgets, initial frontier, and terminal evidence requirements.
- Evidence subset: SCG planning seal and focused baseline results
- Symbolic first: true
- Acceptance: Validator proves exact populations, acyclic dependencies, repository ownership, protected paths, source bindings, initial ready frontier, safety doctrine, and terminal fan-in.

## SCG-001 Inventory accelerate harness, verification, routing, execution, and benchmarks

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-accelerate
- Depends on: SCG-000
- Goal id: SCG-G010
- Outputs: docs/architecture/semantic_compression_governor_inventory/accelerate.json, docs/architecture/semantic_compression_governor_inventory/accelerate.md
- Validation: python3 -m json.tool docs/architecture/semantic_compression_governor_inventory/accelerate.json >/dev/null
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/inventory
- Parallel lane: accelerate-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/semantic_compression_governor_inventory/accelerate.json, docs/architecture/semantic_compression_governor_inventory/accelerate.md
- Interfaces: SemanticCompressionHarness, ContextPacker, IncrementalVerificationPlanner, VerificationReceiptCache, ModelRoutePlanner
- Conflict policy: Read-only inventory of public current-tree surfaces; record RED, simulated, unavailable, and known-failure evidence honestly.
- Preconditions: SCG-000 is sealed.
- Effects: Records exact exports, signatures, schemas, statuses, context/routing rules, tests, benchmark metrics, failure cases, execution isolation, resource and provider seams.
- Evidence subset: accelerate source, tests, docs, checked-in benchmark artifacts
- Symbolic first: true
- LLM context budget bytes: 131072
- Acceptance: Every claimed interface has an exact source/test path and revision; no rollout shadow mode is mistaken for paired semantic shadowing.

## SCG-002 Inventory datasets semantic index, state, capsule, invalidation, and selection

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-datasets
- Depends on: SCG-000
- Goal id: SCG-G010
- Outputs: docs/architecture/semantic_compression_governor_inventory/datasets.json, docs/architecture/semantic_compression_governor_inventory/datasets.md
- Validation: python3 -m json.tool docs/architecture/semantic_compression_governor_inventory/datasets.json >/dev/null
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/inventory
- Parallel lane: datasets-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/semantic_compression_governor_inventory/datasets.json, docs/architecture/semantic_compression_governor_inventory/datasets.md
- Interfaces: IncrementalSemanticIndex, SemanticCapsuleCompiler@1, SemanticStateView, SemanticInvalidationPlan, TestSelection
- Conflict policy: Do not infer a public class from an interface name or propose a second scanner, graph, compiler, or CID implementation.
- Preconditions: SCG-000 is sealed.
- Effects: Records exact APIs, canonical identity rules, statuses, relations, exclusion/fallback behavior, fixtures, metrics, limitations, and focused tests.
- Evidence subset: datasets source, schemas, tests, controlled fixture, docs
- Symbolic first: true
- LLM context budget bytes: 131072
- Acceptance: Functional capsule compiler and verified state-view boundaries are described precisely; opaque/dynamic limitations remain visible.

## SCG-003 Inventory kit immutable blocks, history, recovery, and root CAS

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-kit
- Depends on: SCG-000
- Goal id: SCG-G010
- Outputs: docs/architecture/semantic_compression_governor_inventory/kit.json, docs/architecture/semantic_compression_governor_inventory/kit.md
- Validation: python3 -m json.tool docs/architecture/semantic_compression_governor_inventory/kit.json >/dev/null
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/inventory
- Parallel lane: kit-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/semantic_compression_governor_inventory/kit.json, docs/architecture/semantic_compression_governor_inventory/kit.md
- Interfaces: DurableCoordinationStore, DurableStateRootAdapter, DurableStateRoots
- Conflict policy: Inventory reusable storage/CAS only; do not design another store, WAL, or daemon.
- Preconditions: SCG-000 is sealed.
- Effects: Records exact immutable block, namespace, operation-id, expected-version CAS, corruption, concurrency, replay, recovery, and metrics contracts.
- Evidence subset: kit source, tests, docs, hermetic vectors
- Symbolic first: true
- LLM context budget bytes: 98304
- Acceptance: Inventory identifies the thin governor domain layer still missing and the primitive that must back it.

## SCG-004 Inventory MCP++ shared schemas/vectors and proof-sealer availability

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-interoperability
- Depends on: SCG-000
- Goal id: SCG-G010
- Outputs: docs/architecture/semantic_compression_governor_inventory/interoperability.json, docs/architecture/semantic_compression_governor_inventory/interoperability.md
- Validation: python3 -m json.tool docs/architecture/semantic_compression_governor_inventory/interoperability.json >/dev/null
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/inventory
- Parallel lane: interoperability-inventory
- Resource class: cpu-small
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/semantic_compression_governor_inventory/interoperability.json, docs/architecture/semantic_compression_governor_inventory/interoperability.md
- Interfaces: MCP++ Profile A, Profile B, Profile F, existing Profile G scheduling/artifact codecs and vectors, FullCheckpointSeal/create_full_checkpoint/publish_full_checkpoint and DeltaSeal/build_delta_seal/publish_delta_seal capability probes
- Conflict policy: No new MCP++ profile or local generic envelope; absence of a released sealer is typed unavailable.
- Preconditions: SCG-000 is sealed.
- Effects: Records exact shared wire/vector scope, proof/non-proof distinctions, current sealer development status, and separate release-time full-checkpoint and delta/incremental public-interface probes or typed-unavailable results.
- Evidence subset: MCP++ conformance vectors and upstream proof-sealer program state
- Symbolic first: true
- LLM context budget bytes: 98304
- Acceptance: Full and incremental seal interfaces are each located and commit-bound or independently typed unavailable; IVP Merkle commitment is explicitly non-ZK and cannot substitute for either.

## SCG-005 Synthesize and test the authority consumption matrix

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority-matrix
- Depends on: SCG-001, SCG-002, SCG-003, SCG-004
- Goal id: SCG-G010
- Outputs: docs/architecture/semantic_compression_governor_inventory/authority_matrix.json, test/api/semantic_governor/test_authority_matrix.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_authority_matrix.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/authority
- Parallel lane: integration
- Resource class: cpu-small
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/semantic_compression_governor_inventory/authority_matrix.json, test/api/semantic_governor/test_authority_matrix.py
- Interfaces: SemanticGovernorAuthorityMatrix@1
- Conflict policy: Matrix declares one owner per identity, receipt, state, proof, execution, and storage responsibility.
- Preconditions: All four inventories validate.
- Effects: Converts inventory into executable allowed/forbidden import and ownership assertions, including sealer capability gating.
- Evidence subset: SCG-001 through SCG-004 artifacts
- Symbolic first: true
- LLM context budget bytes: 98304
- Acceptance: Tests reject alternate CID/envelope/store/index/compiler/cache/provider/profile ownership and stale authority pins.

## SCG-006 Define canonical artifact base, statuses, provenance, and identity

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-base
- Depends on: SCG-005
- Goal id: SCG-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/base.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/schemas/base.schema.json, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_base.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_base.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/contracts
- Parallel lane: datasets-contracts
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/base.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/schemas/base.schema.json, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_base.py
- Interfaces: GovernorArtifactHeader@1, ContextSufficiencyState, GovernorTerminalStatus
- Conflict policy: Use `software_contracts.content` only; common header binds repository, ContextPack, verification bundle, generator, provenance, assumptions, and terminal status.
- Preconditions: Authority matrix passes.
- Effects: Defines immutable closed base types and deterministic identity verification used by all artifacts.
- Evidence subset: canonical datasets CID vectors and authority matrix
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Identical inputs have identical CIDs; unknown fields/statuses, floats, forged CIDs, private data, and model-written authority fail closed.

## SCG-007 Define coverage, audit, omission, expansion, and decision contracts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-analysis
- Depends on: SCG-006
- Goal id: SCG-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/audit_contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/schemas/audit.schema.json, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_audit_contracts.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_audit_contracts.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/contracts
- Parallel lane: datasets-contracts
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/audit_contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/schemas/audit.schema.json, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_audit_contracts.py
- Interfaces: CompressionAuditCase, ContextSufficiencyClaim, ContextCoverageManifest, ExcludedArtifactRecord, OmissionHypothesis, OmissionEvidence, ContextExpansionPlan, ContextExpansionStep, GovernorDecision, GovernorRunReceipt
- Conflict policy: Every exclusion and expansion is explicit, costed, graph/state bound, and bounded.
- Preconditions: Canonical artifact base exists.
- Effects: Defines the neutral semantic evidence vocabulary and exact closed exclusion/outcome states.
- Evidence subset: required data-structure and status specification
- Symbolic first: true
- LLM context budget bytes: 229376
- Acceptance: Missing exclusion reasons, unbounded paths/spans/steps, inconsistent totals, and verification-pass-only sufficiency claims reject.

## SCG-008 Define shadow execution and semantic differential contracts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-execution
- Depends on: SCG-006
- Goal id: SCG-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/contracts.py, test/api/semantic_governor/test_contracts.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_contracts.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/contracts
- Parallel lane: accelerate-contracts
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/contracts.py, test/api/semantic_governor/test_contracts.py
- Interfaces: ShadowExecutionPlan, ShadowExecutionResult, DifferentialPatchReport, SemanticOutcomeComparison
- Conflict policy: Execution contracts reference canonical datasets and verification artifacts; they do not mint another receipt hierarchy.
- Preconditions: Canonical artifact base exists.
- Effects: Defines bounded paired-attempt, cost/timing, semantic edit, verification, acceptance, and human-review projections.
- Evidence subset: existing harness scheduling and verification contracts
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Text difference alone cannot classify failure; expanded output is never marked accepted by construction; simulated/live provenance is unambiguous.

## SCG-009 Define calibration profiles and bounded declarative rule DSL

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-policy
- Depends on: SCG-006
- Goal id: SCG-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/calibration_contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/policy_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_policy_contracts.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_policy_contracts.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/contracts
- Parallel lane: datasets-policy
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/calibration_contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/policy_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_policy_contracts.py
- Interfaces: CapsuleCalibrationRecord, AnalyzerCalibrationProfile, TaskClassCalibrationProfile, ModelRouteCalibrationProfile, RuleProposal, RuleEvaluationReport, CompressionPolicy, CompressionPolicyCandidate, CompressionPolicyPromotionReceipt
- Conflict policy: DSL is typed data with an operation allowlist; no expressions, imports, commands, templates, or executable model output.
- Preconditions: Canonical artifact base exists.
- Effects: Defines empirical counters, partitions, confidence intervals, route metrics, rules, candidates, evaluations, authorization and rollback bindings.
- Evidence subset: calibration, rule, and promotion requirements
- Symbolic first: true
- LLM context budget bytes: 229376
- Acceptance: Empirical results cannot set proof classification to exact; candidates cannot self-authorize or reduce protected thresholds without distinct authorization.

## SCG-010 Define the narrow durable governor store protocol

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts-storage
- Depends on: SCG-006
- Goal id: SCG-G020
- Outputs: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/contracts.py, ipfs_kit_py/ipfs_kit_py/semantic_governor_store/__init__.py, ipfs_kit_py/tests/semantic_governor_store/test_contracts.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q ipfs_kit_py/tests/semantic_governor_store/test_contracts.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/contracts
- Parallel lane: kit-contracts
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/contracts.py, ipfs_kit_py/ipfs_kit_py/semantic_governor_store/__init__.py, ipfs_kit_py/tests/semantic_governor_store/test_contracts.py
- Interfaces: SemanticGovernorStore, GovernorArtifactKind, PolicyVersionSnapshot, PolicyCASResult, AuditRecoveryReport
- Conflict policy: Thin protocols over durable coordination/root CAS; no new storage engine, identity function, network, or generic receipt.
- Preconditions: Canonical artifact base exists.
- Effects: Defines closed namespaces and operations for immutable artifacts, histories, policy heads, promotion heads, recovery, and durable issuance/envelope binding of neutral datasets receipt payloads without a second receipt hierarchy.
- Evidence subset: kit durable-root contracts and governor model identities
- Symbolic first: true
- LLM context budget bytes: 163840
- Acceptance: Protocols require caller-supplied verified CIDs, expected generation/root, operation IDs, and typed conflict/corrupt/unavailable outcomes.

## SCG-011 Build a complete ContextCoverageManifest from verified views

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: coverage
- Depends on: SCG-007
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/coverage.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_coverage.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_coverage.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/analysis
- Parallel lane: datasets-coverage
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/coverage.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_coverage.py
- Interfaces: build_context_coverage_manifest
- Conflict policy: Consume ContextPack plus verified semantic-state/index views; do not rescan, invent edges, or infer missing source.
- Preconditions: Audit contracts exist.
- Effects: Attributes every inclusion/exclusion, graph path, proof/test/state/schema/config/fixture/dynamic dependency, assumption, confidence, cost, budget and gap.
- Evidence subset: semantic state facts/links/capsules/invalidation/source and existing ContextPack
- Symbolic first: true
- LLM context budget bytes: 262144
- Acceptance: Critical heuristic exclusion rejects; sufficient exact contexts remain unexpanded; identical inputs yield deterministic manifest identities.

## SCG-012 Implement conservative pre-execution sufficiency evaluation

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: sufficiency
- Depends on: SCG-009, SCG-011
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/sufficiency.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_sufficiency.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_sufficiency.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/analysis
- Parallel lane: datasets-sufficiency
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/sufficiency.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_sufficiency.py
- Interfaces: evaluate_context_sufficiency
- Conflict policy: Closed precedence table; verification pass is one input, never sole sufficiency authority.
- Preconditions: Coverage and calibration contracts exist.
- Effects: Joins risk, confidence, opacity, obligations, cone/cut, proof/test coverage, history, budget, route, and the task-class required-check matrix into one explained state.
- Evidence subset: coverage manifest, policy, calibration, repository state
- Symbolic first: true
- LLM context budget bytes: 245760
- Acceptance: Opaque critical and stale capsules force expansion/raw regeneration; an absent/unknown task-class mapping or missing required selected/full/static/type/proof/review check fails closed; policy boundaries and conflicting evidence require human review; complete-but-hard work can request frontier.

## SCG-013 Detect and quarantine instruction-like untrusted task data

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: security-analysis
- Depends on: SCG-007
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/untrusted_input.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_untrusted_input.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_untrusted_input.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/security
- Parallel lane: datasets-security
- Resource class: security-review
- Implementation timeout seconds: 5400
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/untrusted_input.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_untrusted_input.py
- Interfaces: detect_instruction_like_content, UntrustedInstructionEvidence
- Conflict policy: Detection creates bounded evidence only; source text cannot mutate policy, routing, assurance, keys, proof systems, sampling, verification, or promotion.
- Preconditions: Audit contracts exist.
- Effects: Scans comments, docstrings, task text, tests, logs, and docs for instruction-like patterns while preserving them as untrusted data.
- Evidence subset: adversarial prompt-injection requirements
- Symbolic first: true
- LLM context budget bytes: 131072
- Acceptance: Injection strings cannot alter deterministic decisions even when they mimic trusted configuration or authorization.

## SCG-014 Diagnose and rank omission versus reasoning hypotheses

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: omission
- Depends on: SCG-011, SCG-013
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/omission.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_omission.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_omission.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/analysis
- Parallel lane: datasets-omission
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/omission.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_omission.py
- Interfaces: diagnose_omission
- Conflict policy: Rank evidence; never automatically blame compression or treat model reasoning as formal evidence.
- Preconditions: Coverage and untrusted-input evidence exist.
- Effects: Maps minimized failures and counterexamples to omitted artifacts, graph paths, spans, exclusion/classification, relevance/cost/confidence, expansion, and long-term rule proposals.
- Evidence subset: counterexample receipts, coverage manifest, dependency graph
- Symbolic first: true
- LLM context budget bytes: 245760
- Acceptance: Compressed fail plus expanded success yields ranked omission evidence; both fail does not; evidenced model insufficiency remains a route hypothesis.

## SCG-015 Plan the smallest bounded context expansion

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: expansion-planning
- Depends on: SCG-012, SCG-014
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/expansion.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_expansion.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_expansion.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/analysis
- Parallel lane: datasets-expansion
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/expansion.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_expansion.py
- Interfaces: plan_context_expansion
- Conflict policy: Complete affected cone is preferred to repository dump; every plan has hard token/step/retry/escalation/time/spend limits.
- Preconditions: Sufficiency and ranked hypotheses exist.
- Effects: Chooses raw source or stronger capsule additions by coverage value, evidence, disclosure, and cost while recording changed assumptions.
- Evidence subset: omission hypotheses and coverage gaps
- Symbolic first: true
- LLM context budget bytes: 212992
- Acceptance: Expanded context remains bounded; impossible/unsafe budget returns human review; omission expansion precedes model escalation where supported.

## SCG-016 Implement empirical capsule, analyzer, task, and route calibration

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: calibration
- Depends on: SCG-007, SCG-009
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/calibration.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_calibration.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_calibration.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/calibration
- Parallel lane: datasets-calibration
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/calibration.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_calibration.py
- Interfaces: update_calibration, merge_calibration_profiles
- Conflict policy: Empirical statistics affect route/audit frequency only and never mathematical proof classification.
- Preconditions: Calibration contracts exist.
- Effects: Updates keyed counters, costs, disagreements, omission rates and confidence intervals deterministically with revision binding.
- Evidence subset: CompressionAuditCase histories
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Simulated outputs are excluded from live quality; concurrent/replayed inputs are idempotent; false exact and stale failures remain explicit.

## SCG-017 Generate bounded declarative analyzer, invalidation, packing, and route proposals

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rule-proposals
- Depends on: SCG-015, SCG-016
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/rules.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_rules.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_rules.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/policy
- Parallel lane: datasets-rules
- Resource class: security-review
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/rules.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_rules.py
- Interfaces: propose_rule_change, validate_rule_proposal
- Conflict policy: Typed allowlisted operations only; no code, templates, shell, import paths, provider IDs, keys, or promotion authority.
- Preconditions: Expansion and calibration algorithms exist.
- Effects: Creates evidence-bound dependency-extraction, invalidation, capsule-completeness, raw-source-inclusion, context-ranking/packing, budget, route-threshold, shadow-sampling, and safe full-suite-fallback proposals with current version, scope, benefit, safety analysis, benchmark and rollback plans.
- Evidence subset: calibration profiles and supporting audits
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Arbitrary code cannot execute; full-suite fallback cannot be disabled; high-risk assurance cannot be reduced in a normal proposal.

## SCG-018 Freeze datasets public governor API and conformance matrix

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-release
- Depends on: SCG-012, SCG-015, SCG-017
- Goal id: SCG-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/__init__.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_public_api.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_conformance.py
- Validation: PYTHONPATH=ipfs_datasets_py:. python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/datasets-release
- Parallel lane: datasets-integration
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/__init__.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_public_api.py, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor/test_conformance.py
- Interfaces: evaluate_context_sufficiency, diagnose_omission, plan_context_expansion, update_calibration, propose_rule_change
- Conflict policy: Lazy public exports only; no I/O, optional install, or accelerate/kit implementation import at import time.
- Preconditions: All datasets analysis modules pass focused tests.
- Effects: Publishes the reviewed neutral analysis surface and joined adversarial conformance.
- Evidence subset: SCG-G030 implementation tests
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Required APIs work on canonical objects and mappings; identities/statuses are deterministic; all uncertainty and injection tests fail closed.

## SCG-019 Implement immutable audit artifact storage

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-artifacts
- Depends on: SCG-007, SCG-008, SCG-009, SCG-010
- Goal id: SCG-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/artifacts.py, ipfs_kit_py/tests/semantic_governor_store/test_artifacts.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q ipfs_kit_py/tests/semantic_governor_store/test_artifacts.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/storage
- Parallel lane: kit-artifacts
- Resource class: io-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/artifacts.py, ipfs_kit_py/tests/semantic_governor_store/test_artifacts.py
- Interfaces: DurableSemanticGovernorStore, put_artifact, get_verified_artifact
- Conflict policy: Compose DurableCoordinationStore; never trust a supplied CID without recomputing canonical bytes.
- Preconditions: Store protocol exists.
- Effects: Stores immutable typed audit, calibration, benchmark, policy, evaluation, and receipt blocks under closed artifact kinds.
- Evidence subset: kit immutable block and CID verification contracts
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Corrupt, forged, wrong-kind, oversized, private-raw-source, or unknown-version artifacts fail closed.

## SCG-020 Implement append-only audit, calibration, and benchmark histories

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-history
- Depends on: SCG-019
- Goal id: SCG-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/history.py, ipfs_kit_py/tests/semantic_governor_store/test_history.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q ipfs_kit_py/tests/semantic_governor_store/test_history.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/storage
- Parallel lane: kit-history
- Resource class: io-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/history.py, ipfs_kit_py/tests/semantic_governor_store/test_history.py
- Interfaces: AuditHistoryStore, append_audit, append_calibration, append_benchmark
- Conflict policy: Histories reference immutable CIDs and preserve rejected/stale records; no destructive rewrite.
- Preconditions: Immutable artifact store exists.
- Effects: Adds deterministic append manifests, idempotent operation IDs, pagination bounds, and public/private projections.
- Evidence subset: immutable governor artifacts
- Symbolic first: true
- LLM context budget bytes: 163840
- Acceptance: Replay is idempotent; concurrent writers preserve both histories; public projection exposes no raw private source or arbitrary local path.

## SCG-021 Implement versioned compression-policy and promotion CAS repositories

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-policy
- Depends on: SCG-019
- Goal id: SCG-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/policy.py, ipfs_kit_py/tests/semantic_governor_store/test_policy_cas.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q ipfs_kit_py/tests/semantic_governor_store/test_policy_cas.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/storage
- Parallel lane: kit-policy
- Resource class: io-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/policy.py, ipfs_kit_py/tests/semantic_governor_store/test_policy_cas.py
- Interfaces: CompressionPolicyRepository, PromotionStateRepository, compare_and_swap_policy
- Conflict policy: Expected generation plus expected policy CID; candidate and authorization CIDs are separate and immutable.
- Preconditions: Immutable artifact store exists.
- Effects: Publishes policy/promotions atomically with operation idempotency, history links, and rollback references.
- Evidence subset: kit root CAS and policy contracts
- Symbolic first: true
- LLM context budget bytes: 180224
- Acceptance: Candidate cannot promote itself; stale candidate cannot overwrite current; ABA and concurrent writers yield at most one success; rollback preserves history.

## SCG-022 Prove store corruption, interruption, concurrency, privacy, and recovery

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: storage-conformance
- Depends on: SCG-020, SCG-021
- Goal id: SCG-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/recovery.py, ipfs_kit_py/tests/semantic_governor_store/test_recovery.py, ipfs_kit_py/tests/semantic_governor_store/test_concurrency.py, ipfs_kit_py/tests/semantic_governor_store/test_privacy.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q ipfs_kit_py/tests/semantic_governor_store
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/storage-conformance
- Parallel lane: kit-integration
- Resource class: io-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_governor_store/recovery.py, ipfs_kit_py/tests/semantic_governor_store/test_recovery.py, ipfs_kit_py/tests/semantic_governor_store/test_concurrency.py, ipfs_kit_py/tests/semantic_governor_store/test_privacy.py
- Interfaces: recover_governor_store, AuditRecoveryReport
- Conflict policy: Recovery rebuilds indexes from verified immutable blocks and never invents promotion or completion.
- Preconditions: Histories and policy CAS exist.
- Effects: Exercises crash boundaries, corrupted derived state, concurrent calibration writers, interrupted audits, stale pointers, public report redaction, and replay.
- Evidence subset: full kit governor store
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Interrupted audits recover safely; writers never silently overwrite; corruption and ambiguous promotion fail closed.

## SCG-023 Adapt canonical datasets, harness, verification, storage, and sealer surfaces

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-adapters
- Depends on: SCG-018, SCG-022
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/adapters.py, test/api/semantic_governor/test_adapters.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_adapters.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-adapters
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/adapters.py, test/api/semantic_governor/test_adapters.py
- Interfaces: GovernorDatasetsAdapter, GovernorHarnessAdapter, GovernorVerificationAdapter, GovernorStoreAdapter, IncrementalSealerCapability
- Conflict policy: Lazy version/fingerprint probes and typed unavailable results; no copied upstream models or fallback implementation.
- Preconditions: Datasets and kit governor surfaces are frozen.
- Effects: Normalizes canonical objects into narrow runtime views and probes the released IncrementalProofSealer without importing unfinished private code.
- Evidence subset: authority matrix and public exports
- Symbolic first: true
- LLM context budget bytes: 229376
- Acceptance: Stale/missing/incompatible capability fails closed; IVP commitment cannot satisfy sealer capability; imports perform no I/O.

## SCG-024 Enforce source disclosure, redaction, provider, and worktree policy

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-privacy
- Depends on: SCG-013, SCG-023
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/privacy.py, test/api/semantic_governor/test_privacy.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_privacy.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/security
- Parallel lane: accelerate-privacy
- Resource class: security-review
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/privacy.py, test/api/semantic_governor/test_privacy.py
- Interfaces: ShadowDisclosurePolicy, redact_context_for_provider, authorize_shadow_disclosure
- Conflict policy: Local-only expansion by default; exact explicit authority required for broader external disclosure.
- Preconditions: Runtime adapters and untrusted-input evidence exist.
- Effects: Binds provider capability, source privacy, allowed repository/path classes, redaction, secret scan, and isolated evaluation-worktree policy.
- Evidence subset: provider gateway redaction and source disclosure requirements
- Symbolic first: true
- LLM context budget bytes: 180224
- Acceptance: Private source is never sent to an unapproved external shadow provider; secrets and arbitrary host paths cannot enter invocation or public reports.

## SCG-025 Implement risk- and information-value-aware shadow planning

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: shadow-planning
- Depends on: SCG-024
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow_plan.py, test/api/semantic_governor/test_shadow_plan.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_shadow_plan.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-shadow
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow_plan.py, test/api/semantic_governor/test_shadow_plan.py
- Interfaces: create_shadow_plan, ShadowSamplingPolicy
- Conflict policy: Deterministic sampling with explicit random seed and privacy/resource gates; expanded result is oracle only.
- Preconditions: Privacy policy and adapters pass.
- Effects: Selects audits by risk, uncertainty, novelty, savings, reuse, recent failures, QC sample, and promotion evaluation with configurable rates.
- Evidence subset: context pack, repository state, audit policy, calibration
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Development/high risk can shadow 100 percent; mature low risk samples; forbidden disclosure produces local-only or no external call, never policy bypass.

## SCG-026 Execute paired compressed and expanded attempts in isolated worktrees

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: shadow-execution
- Depends on: SCG-025
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow.py, test/api/semantic_governor/test_shadow.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_shadow.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-shadow
- Resource class: cpu-large
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow.py, test/api/semantic_governor/test_shadow.py
- Interfaces: ShadowExecutor, execute_shadow_plan
- Conflict policy: Reuse ResourceScheduler, ProviderExecutionGateway, semantic work scheduling, and isolated worktree lifecycle; no production checkout edits.
- Preconditions: A valid shadow plan is admitted.
- Effects: Executes separately bound contexts, captures costs/tokens/time/proposals and canonical verification references, and fences cancellation/recovery.
- Evidence subset: harness, resource leases, provider receipts, worktree lifecycle
- Symbolic first: true
- LLM context budget bytes: 262144
- Acceptance: Expanded output never auto-accepts; budgets and disclosure are rechecked before invocation; cancellation/timeouts leave production state unchanged.

## SCG-027 Compare semantic patch outcomes beyond text

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: differential
- Depends on: SCG-026
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/differential.py, test/api/semantic_governor/test_differential.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_differential.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-differential
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/differential.py, test/api/semantic_governor/test_differential.py
- Interfaces: compare_shadow_results
- Conflict policy: Semantic equivalence uses structural/verification evidence; model agreement and textual equality are not proof.
- Preconditions: Paired shadow results exist.
- Effects: Compares file/symbol/AST/interface/effect/exception/schema/test/proof/counterexample/static/performance/acceptance/review/token/cost/time evidence into a closed outcome.
- Evidence subset: paired results and verification bundles
- Symbolic first: true
- LLM context budget bytes: 212992
- Acceptance: Equivalent valid patches classify equivalent; compressed-failed/expanded-succeeded is distinct; inconclusive verification stays inconclusive.

## SCG-028 Bridge verification bundles and minimized counterexamples into audits

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: verification-bridge
- Depends on: SCG-023
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/verification.py, test/api/semantic_governor/test_verification.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_verification.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-verification
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/verification.py, test/api/semantic_governor/test_verification.py
- Interfaces: GovernorVerificationBridge, build_audit_verification_evidence
- Conflict policy: Reuse canonical VerificationBundle/TestReceipt/ProofReceipt/CounterexampleReceipt and acceptance policy; no receipt translation that upgrades status.
- Preconditions: Runtime adapters pass.
- Effects: Runs or opens the exact selected/full/static/type/proof/review checks declared for the task class, rejects unknown mappings, recomputes acceptance, minimizes failure evidence, and binds selected/full/proof conflict signals.
- Evidence subset: IncrementalVerificationPlanner and VerificationExecutor
- Symbolic first: true
- LLM context budget bytes: 212992
- Acceptance: Patch/model/one-test/receipt/aggregate presence cannot accept; missing task-class policy or any required check fails closed; stale/simulated/unavailable evidence remains nonaccepting.

## SCG-029 Execute bounded counterexample-guided context expansion before route escalation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: expansion-runtime
- Depends on: SCG-018, SCG-027, SCG-028
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/expansion_loop.py, test/api/semantic_governor/test_expansion_loop.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_expansion_loop.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-expansion
- Resource class: cpu-large
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/expansion_loop.py, test/api/semantic_governor/test_expansion_loop.py
- Interfaces: execute_expansion_loop
- Conflict policy: Same model after supported context expansion when appropriate; model escalation only after expansion is insufficient or evidence says reasoning failure.
- Preconditions: Differential, verification, and datasets expansion planner exist.
- Effects: Performs bounded hypothesis/add/retry/verify cycles with token, step, retry, escalation, time, and spend caps.
- Evidence subset: omission hypotheses, expansion plan, counterexamples, model policy
- Symbolic first: true
- LLM context budget bytes: 262144
- Acceptance: Limits are enforced across restart; supported omission can repair before frontier; both-context failure can request route escalation without blaming compression.

## SCG-030 Calibrate model routes separately from context sufficiency

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: route-calibration
- Depends on: SCG-016, SCG-029
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/routes.py, test/api/semantic_governor/test_routes.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_routes.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-routing
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/routes.py, test/api/semantic_governor/test_routes.py
- Interfaces: update_model_route_calibration, propose_route_threshold_change
- Conflict policy: Capability tier only; no provider ID or direct production route mutation; high-risk requirements never auto-lower.
- Preconditions: Expansion loop and calibration exist.
- Effects: Tracks accepted rate, retries, expansion, verification, omission/reasoning failures, cost and latency by deterministic/small/medium/frontier/human routes.
- Evidence subset: GovernorRunReceipt histories and ModelRoutePlanner decisions
- Symbolic first: true
- LLM context budget bytes: 180224
- Acceptance: Context omission and reasoning failure are separate counters; unavailable required tier never downgrades; changes are proposals only.

## SCG-031 Implement active audit scheduling by expected information value

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: audit-scheduler
- Depends on: SCG-022, SCG-030
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/scheduler.py, test/api/semantic_governor/test_scheduler.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_scheduler.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate-scheduler
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/scheduler.py, test/api/semantic_governor/test_scheduler.py
- Interfaces: ActiveAuditScheduler, AuditPriority, schedule_audits
- Conflict policy: Priority is bounded/deterministic and resource-admitted; mature repetitive low-risk tasks cannot monopolize audit spend.
- Preconditions: Durable histories and route calibration exist.
- Effects: Ranks risk, uncertainty, savings, rule exposure, sample deficit, failures, cost/escalation, cone size, dynamic features and policy importance.
- Evidence subset: calibration histories, resource capacity, audit policy
- Symbolic first: true
- LLM context budget bytes: 180224
- Acceptance: Configured shadow rates and privacy zero-call policy are honored; starvation and unbounded queue growth tests pass.

## SCG-032 Freeze runtime APIs and prove shadow/expansion resilience

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-conformance
- Depends on: SCG-031
- Goal id: SCG-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/runtime.py, test/api/semantic_governor/test_runtime_conformance.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_runtime_conformance.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/runtime-release
- Parallel lane: accelerate-integration
- Resource class: cpu-large
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/runtime.py, test/api/semantic_governor/test_runtime_conformance.py
- Interfaces: GovernorRuntime, audit_task, shadow_task, expand_audit
- Conflict policy: One composition path over existing harness/verification/store/resource/provider/worktree authorities.
- Preconditions: Scheduler and all runtime components pass.
- Effects: Joins recovery, idempotency, budgets, privacy, audit persistence, differential and decision publication into one resumable runtime.
- Evidence subset: SCG-G050 implementation tests
- Symbolic first: true
- LLM context budget bytes: 262144
- Acceptance: Interrupted audits recover; duplicate inputs preserve identities; private external shadow, unbounded expansion, suppressed failure, and simulated live-quality claims are rejected.

## SCG-033 Evaluate rule candidates only on disjoint held-out evidence

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: policy-evaluation
- Depends on: SCG-017, SCG-022, SCG-032
- Goal id: SCG-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/policy_evaluation.py, test/api/semantic_governor/test_policy_evaluation.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_policy_evaluation.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/policy
- Parallel lane: policy-evaluation
- Resource class: cpu-large
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/policy_evaluation.py, test/api/semantic_governor/test_policy_evaluation.py
- Interfaces: evaluate_rule_candidate
- Conflict policy: Calibration/development/held-out identities are disjoint and immutable; candidate-generating cases cannot score promotion.
- Preconditions: Runtime, rule DSL and durable evidence pass.
- Effects: Checks schema/integrity, omission/stale/high-risk/regression/cost thresholds and emits a reproducible report without mutation.
- Evidence subset: candidate, held-out benchmark manifest, baseline policy
- Symbolic first: true
- LLM context budget bytes: 229376
- Acceptance: Missing/overlapping held-out data rejects; critical omission detection and stale rejection cannot regress; hidden accepted regressions block.

## SCG-034 Implement authorized compare-and-swap promotion and rollback

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: policy-promotion
- Depends on: SCG-021, SCG-033, SCG-035
- Goal id: SCG-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/promotion.py, test/api/semantic_governor/test_promotion.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_promotion.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/policy
- Parallel lane: policy-promotion
- Resource class: security-review
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/promotion.py, test/api/semantic_governor/test_promotion.py
- Interfaces: promote_compression_policy, rollback_compression_policy
- Conflict policy: Separate trusted authorization and expected-version CAS; model output, candidate, evaluation, or seal cannot authorize itself.
- Preconditions: Held-out evaluation, a current IncrementalProofSealer qualification or separately authorized VerificationBundle-backed release qualification, and the current policy repository all pass.
- Effects: Revalidates all gates at publication, records promotion/rollback receipts, and publishes one version atomically.
- Evidence subset: RuleEvaluationReport, release qualification or released incremental seal, authorization, current policy snapshot
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Stale candidate, absent or unavailable qualification, absent authorization, reduced high-risk assurance, mismatched evaluation, CAS conflict, or self-promotion cannot mutate the head.

## SCG-035 Gate promotion on release qualification and bind incremental seals without overclaiming

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: sealing
- Depends on: SCG-023, SCG-033
- Goal id: SCG-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/sealing.py, test/api/semantic_governor/test_sealing.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_sealing.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/sealing
- Parallel lane: proof-integration
- Resource class: security-review
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/sealing.py, test/api/semantic_governor/test_sealing.py
- Interfaces: SemanticGovernorSealAdapter, qualify_policy_candidate, seal_governor_run, verify_governor_seal
- Conflict policy: Use released IncrementalProofSealer only; otherwise typed unavailable. VerificationCommitment is structural non-ZK evidence only.
- Preconditions: Held-out evaluation and the canonical verification adapter exist; the released sealer is capability-detected rather than assumed.
- Effects: Requires either current released incremental-seal evidence or a separately authorized VerificationBundle-backed release qualification before promotion, and binds benchmark, ContextPacks, bundles, differential, calibration, candidates and decisions to exact policy/evaluation identities.
- Evidence subset: released sealer public API, or current release-qualification VerificationBundle and typed sealer unavailability
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Missing sealer is typed unavailable and never replaced by VerificationCommitment; promotion remains blocked unless the independently authorized release-qualification path passes; signed/sealed artifacts bind evaluated policy and make only the encoded bounded claim.

## SCG-036 Compose the required SemanticCompressionGovernor APIs

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: public-api
- Depends on: SCG-032, SCG-034, SCG-035
- Goal id: SCG-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py, test/api/semantic_governor/test_public_api.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_public_api.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/interfaces
- Parallel lane: interfaces
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py, test/api/semantic_governor/test_public_api.py
- Interfaces: SemanticCompressionGovernor, ten required module-level APIs
- Conflict policy: Lazy composition and dependency injection; no I/O, process, network, provider, or optional install on import.
- Preconditions: Runtime, promotion and sealing surfaces exist.
- Effects: Exposes the ten required API equivalents with closed inputs/outputs and typed unavailable results.
- Evidence subset: datasets, runtime, policy and sealing public APIs
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Signatures and return types are stable; all safety/identity gates survive facade use; unknown commands or fields reject.

## SCG-037 Add the narrowly scoped semantic-governor CLI

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: cli
- Depends on: SCG-036, SCG-039
- Goal id: SCG-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/cli.py, test/api/semantic_governor/test_cli.py, setup.py, pyproject.toml
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_cli.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/interfaces
- Parallel lane: cli
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/cli.py, test/api/semantic_governor/test_cli.py, setup.py, pyproject.toml
- Interfaces: semantic-governor audit, shadow, diagnose, expand, calibrate, propose-rules, evaluate-policy, promote-policy, report, dashboard-data
- Conflict policy: Deterministic bounded JSON default; no GUI, listener, arbitrary path exposure, provider configuration, or implicit promotion.
- Preconditions: Public API exists.
- Effects: Maps ten subcommands to the same typed APIs and managed artifact references.
- Evidence subset: public API and existing semantic-state CLI conventions
- Symbolic first: true
- LLM context budget bytes: 163840
- Acceptance: Exact ten commands work; private raw source stays out of output; promotion requires explicit authorization input and CAS.

## SCG-038 Implement complete compression, quality, omission, routing, economic, and calibration metrics

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: metrics
- Depends on: SCG-016, SCG-032
- Goal id: SCG-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/metrics.py, test/api/semantic_governor/test_metrics.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_metrics.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/metrics
- Parallel lane: metrics
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/metrics.py, test/api/semantic_governor/test_metrics.py
- Interfaces: GovernorMetricsCollector, GovernorMetricReport
- Conflict policy: Exact integer/fixed-point accounting with provenance; unavailable data is missing, not zero or success.
- Preconditions: Runtime receipts and calibration exist.
- Effects: Computes all specified distributions, precision/recall, route shares, cost/savings, confidence intervals, revisions and coverage.
- Evidence subset: canonical audit/run/calibration receipts
- Symbolic first: true
- LLM context budget bytes: 163840
- Acceptance: Simulated and live cohorts stay separate; percentiles and costs are reproducible; audit overhead is included in net savings.

## SCG-039 Implement privacy-filtered report and dashboard-data projections

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: reports
- Depends on: SCG-036, SCG-038
- Goal id: SCG-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor/report.py, test/api/semantic_governor/test_report.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_report.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/interfaces
- Parallel lane: reports
- Resource class: cpu-small
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/report.py, test/api/semantic_governor/test_report.py
- Interfaces: build_governor_report, build_dashboard_data
- Conflict policy: Machine-readable projection only; no graphical dashboard or public server.
- Preconditions: Public API and metrics exist.
- Effects: Produces bounded summary/detail views with CIDs/managed references and explicit unavailable, simulated, heuristic and proof-scope fields.
- Evidence subset: governor histories and metrics
- Symbolic first: true
- LLM context budget bytes: 147456
- Acceptance: Required final-report fields are representable; raw private source, secrets, arbitrary paths and human/model free-form authority are absent.

## SCG-040 Build deterministic partitioned fixture repositories and manifests

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark-corpus
- Depends on: SCG-005
- Goal id: SCG-G080
- Outputs: test/fixtures/semantic_governor, test/api/semantic_governor/test_fixture_corpus.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_fixture_corpus.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/benchmark
- Parallel lane: fixtures
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Predicted files: test/fixtures/semantic_governor, test/api/semantic_governor/test_fixture_corpus.py
- Interfaces: SemanticGovernorFixtureCorpus@1
- Conflict policy: Controlled source/patch/oracle data only; no model outputs, receipts, state DB, hidden external dependency, or overlap between partitions.
- Preconditions: Authority matrix identifies canonical scanner and harness.
- Effects: Creates calibration/development/held-out families for bugs, exceptions, migrations, state/config/fixture/dynamic/generated/plugin/refactor/docs/proof tasks.
- Evidence subset: existing semantic-state and incremental-verification fixtures
- Symbolic first: true
- LLM context budget bytes: 229376
- Acceptance: Partitions and expected omissions/outcomes are deterministic, disjoint, scanner-derived, and independently declared.

## SCG-041 Prove structural omission, stale-artifact, policy, and bounded-expansion cases

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adversarial-static
- Depends on: SCG-018, SCG-040
- Goal id: SCG-G080
- Outputs: test/api/semantic_governor/test_adversarial_static.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_adversarial_static.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/benchmark
- Parallel lane: adversarial-static
- Resource class: cpu-large
- Implementation timeout seconds: 9000
- Predicted files: test/api/semantic_governor/test_adversarial_static.py
- Interfaces: static omission conformance
- Conflict policy: Real canonical views and fixtures; no hand-injected passing identity or fabricated receipt.
- Preconditions: Datasets governor public API and corpus exist.
- Effects: Tests hidden callee effect, caller exception, config, fixture, serializer, generated interface, stale capsule, confidence misclassification, behavior-only change, security invariant and migration path.
- Evidence subset: held-out static fixture partition
- Symbolic first: true
- LLM context budget bytes: 229376
- Acceptance: Critical omissions are detected before automatic acceptance; exact sufficient context is not needlessly expanded; expansion limits hold.

## SCG-042 Prove dynamic, prompt-injection, selected/full, proof, and model-capability cases

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adversarial-dynamic
- Depends on: SCG-024, SCG-028, SCG-029, SCG-032, SCG-041
- Goal id: SCG-G080
- Outputs: test/api/semantic_governor/test_adversarial_dynamic.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_adversarial_dynamic.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/benchmark
- Parallel lane: adversarial-dynamic
- Resource class: cpu-large
- Implementation timeout seconds: 9000
- Predicted files: test/api/semantic_governor/test_adversarial_dynamic.py
- Interfaces: dynamic omission and reasoning conformance
- Conflict policy: Prompt/source text is untrusted; no test may alter trusted runtime configuration through fixture content.
- Preconditions: Static corpus and privacy gate pass.
- Effects: Tests opaque dynamic import, monkey patch/plugin behavior, misleading comments, prompt injection, selected-pass/full-fail, test-pass/formal-fail, raw-correct/compressed-wrong, and both-context model failure.
- Evidence subset: held-out dynamic/security fixture partition
- Symbolic first: true
- LLM context budget bytes: 245760
- Acceptance: Governor distinguishes omission from model insufficiency where evidence permits; injection cannot alter behavior; verification conflict requires review.

## SCG-043 Prove interruption, concurrency, disclosure, simulation, and cost boundaries end to end

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: resilience-privacy
- Depends on: SCG-022, SCG-026, SCG-029, SCG-032, SCG-041
- Goal id: SCG-G080
- Outputs: test/api/semantic_governor/test_resilience_privacy.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_resilience_privacy.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/benchmark
- Parallel lane: resilience
- Resource class: security-review
- Implementation timeout seconds: 9000
- Predicted files: test/api/semantic_governor/test_resilience_privacy.py
- Interfaces: governor resilience conformance
- Conflict policy: Hermetic local providers/stores and controlled crash hooks only; no external source disclosure.
- Preconditions: Store, shadow executor, and static corpus pass.
- Effects: Tests identical identity, interrupted recovery, concurrent calibration CAS, unauthorized external context, redaction, simulated/live separation, and every budget fence.
- Evidence subset: runtime and storage adversarial fixtures
- Symbolic first: true
- LLM context budget bytes: 229376
- Acceptance: No silent overwrite/disclosure/quality claim; cancellation and spend/token/time/retry limits survive recovery.

## SCG-044 Prove the complete API, CLI, policy, rollback, and audit loop

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: end-to-end
- Depends on: SCG-037, SCG-039, SCG-042, SCG-043
- Goal id: SCG-G080
- Outputs: test/api/semantic_governor/test_end_to_end.py, test/api/semantic_governor/test_install_and_import.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_end_to_end.py test/api/semantic_governor/test_install_and_import.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/benchmark
- Parallel lane: e2e
- Resource class: cpu-large
- Implementation timeout seconds: 10800
- Predicted files: test/api/semantic_governor/test_end_to_end.py, test/api/semantic_governor/test_install_and_import.py
- Interfaces: SemanticCompressionGovernor end-to-end acceptance
- Conflict policy: Run real controlled repositories and canonical artifacts; no injected acceptance, fabricated provider receipt, or public server.
- Preconditions: CLI/report and adversarial suites pass.
- Effects: Executes compress, verify, sample, shadow, diagnose, expand, calibrate, propose, held-out evaluate, authorize/CAS promote, rollback, report and seal/unavailable paths.
- Evidence subset: complete controlled fixture corpus
- Symbolic first: true
- LLM context budget bytes: 262144
- Acceptance: Every required test invariant is exercised through public APIs; rollout/promotion remains disabled unless explicitly authorized.

## SCG-045 Run the benchmark and persist honest measured evidence

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark
- Depends on: SCG-038, SCG-044
- Goal id: SCG-G080
- Outputs: benchmarks/agent_supervisor/semantic_compression_governor.py, test/benchmarks/test_semantic_compression_governor_benchmark.py, artifacts/agent_supervisor/semantic_compression_governor/benchmark.json, artifacts/agent_supervisor/semantic_compression_governor/summary.json
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/benchmarks/test_semantic_compression_governor_benchmark.py && PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 benchmarks/agent_supervisor/semantic_compression_governor.py --check
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/benchmark
- Parallel lane: benchmark
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- Predicted files: benchmarks/agent_supervisor/semantic_compression_governor.py, test/benchmarks/test_semantic_compression_governor_benchmark.py, artifacts/agent_supervisor/semantic_compression_governor/benchmark.json, artifacts/agent_supervisor/semantic_compression_governor/summary.json
- Interfaces: SemanticGovernorBenchmark@1
- Conflict policy: Targets are thresholds, not output constants; simulated/local/unavailable/live cohorts are labeled and separated.
- Preconditions: E2E and metrics pass.
- Effects: Measures all required compression, quality, omission, routing, economic and calibration fields with exact commit/tool/policy provenance.
- Evidence subset: disjoint controlled benchmark partitions
- Symbolic first: true
- LLM context budget bytes: 262144
- Acceptance: Reports actual outcome distribution, detection, critical acceptance, expansion, reduction, routes, quality, regressions, overhead, cost, proposals and rejections; missing evidence is explicit.

## SCG-046 Publish trust, privacy, promotion, operation, and limitation documentation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: documentation
- Depends on: SCG-045
- Goal id: SCG-G090
- Outputs: docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md, docs/guides/SEMANTIC_COMPRESSION_GOVERNOR.md
- Validation: python3 scripts/docs/check_agent_supervisor_docs.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/release
- Parallel lane: documentation
- Resource class: security-review
- Implementation timeout seconds: 7200
- Predicted files: docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md, docs/guides/SEMANTIC_COMPRESSION_GOVERNOR.md
- Interfaces: SemanticGovernorTrustModel@1, SemanticGovernorOperations@1
- Conflict policy: Document exact authority and nonclaims; never present empirical calibration or seals as universal semantic proof.
- Preconditions: Actual benchmark results exist.
- Effects: Documents disclosure/redaction, untrusted inputs, assurance, authorization, promotion/rollback, recovery, CLI, metrics, seal scope, limitations and incident response.
- Evidence subset: implementation and benchmark artifacts
- Symbolic first: true
- LLM context budget bytes: 196608
- Acceptance: Operators can reproduce evaluation/promotion/rollback and understand what is structural, empirical, heuristic, unavailable, and formally proven.

## SCG-047 Qualify released IncrementalProofSealer binding and rollback evidence

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-sealing
- Depends on: SCG-035, SCG-045
- Goal id: SCG-G090
- Outputs: test/api/semantic_governor/test_release_sealing.py, artifacts/agent_supervisor/semantic_compression_governor/seal_qualification.json, artifacts/agent_supervisor/semantic_compression_governor/rollback.json
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor/test_release_sealing.py
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/release
- Parallel lane: release-sealing
- Resource class: security-review
- Implementation timeout seconds: 9000
- Predicted files: test/api/semantic_governor/test_release_sealing.py, artifacts/agent_supervisor/semantic_compression_governor/seal_qualification.json, artifacts/agent_supervisor/semantic_compression_governor/rollback.json
- Interfaces: IncrementalProofSealer release qualification, rollback qualification
- Conflict policy: If the canonical sealer has not released, produce a truthful unavailable qualification and block any proof-backed promotion claim; do not clone it.
- Preconditions: Seal adapter and actual benchmark results exist.
- Effects: Probes exact release API, binds artifacts/policy, tests stale/tamper/blocking-status cases, and executes authorized promotion/rollback CAS in a hermetic namespace.
- Evidence subset: released sealer or typed unavailability, policy CAS history
- Symbolic first: true
- LLM context budget bytes: 212992
- Acceptance: Seal scope is precise; stale/corrupt/mismatched candidate fails; rollback is reproducible; unavailable sealer never stalls unrelated governor qualification.

## SCG-048 Run terminal current-tree qualification and publish the final report

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release
- Depends on: SCG-046, SCG-047
- Goal id: SCG-G090
- Outputs: docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_REPORT.md, artifacts/agent_supervisor/semantic_compression_governor/release.json
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor test/benchmarks/test_semantic_compression_governor_benchmark.py && python3 scripts/validate_semantic_compression_governor_board.py --check-all
- Board namespace: semantic-compression-governor-v1
- Bundle: semantic-governor/release
- Parallel lane: terminal
- Resource class: cpu-large
- Implementation timeout seconds: 14400
- Predicted files: docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_REPORT.md, artifacts/agent_supervisor/semantic_compression_governor/release.json
- Interfaces: SemanticGovernorReleaseQualification@1
- Conflict policy: Terminal fan-in only; no target fabrication, silent policy promotion, control-file edit, or universal sufficiency claim.
- Preconditions: Trust docs and release-seal/rollback qualification pass or report typed unavailable without false proof claims.
- Effects: Runs current-tree suites and records exact commits/interfaces, cases, outcomes, detection/acceptance, expansion/reduction, routes/escalation, quality/regression, overhead/cost, rule decisions, promotion/rollback, proof scope, heuristics and remaining risks.
- Evidence subset: all SCG goals and immutable benchmark/release artifacts
- Symbolic first: true
- LLM context budget bytes: 262144
- Acceptance: Final report contains every required field and the bounded claim from the plan; no policy is reported promoted unless a separate authorization and successful CAS receipt exist.
