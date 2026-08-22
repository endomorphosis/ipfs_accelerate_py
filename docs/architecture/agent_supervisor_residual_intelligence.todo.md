# Verified Residual Intelligence Foundry task board

Executable bootstrap projection for
`agent-supervisor-verified-residual-intelligence-foundry-v1`, prefix `VRIF-`,
root `VRIF-G000`, plan `VRIF-PLAN-R1`. DuckDB is authoritative for goals,
tasks, revisions, dependencies, attempts, claims, leases/fencing, evidence,
validation, and completion. Quack is the authenticated loopback state-owner
transport. DuckLake is non-authoritative history and benchmark projection only.
After materialization this file is a sealed human/export view.

Task-board status is not completion evidence. A `completed` bootstrap row for
VRIF-000 through VRIF-008 is admitted only with exact current-tree validation
from the declared producers. Workers cannot self-complete and model output is
never proof, authority, policy, confirmation, or completion.

Current PGIR freeze/no-go remains binding. There is no rights-admitted training
corpus, tokenizer, checkpoint, or promotion authority, so training returns
`training_unavailable`. There is no training on unadmitted data. Contract,
corpus-construction, deterministic baseline, evaluation, and runtime work may
continue without training; an optional training blocker cannot globally block
independent ready tasks.

The first live frontier after Tranche 1 is VRIF-009, VRIF-010, VRIF-011, and
VRIF-012. Each owns disjoint modules. Unknown scope serializes; all work uses
fenced isolated worktrees and exact merge-target revalidation. Sibling
repositories are read-only.

## VRIF-000 Seal current authority and prerequisite baseline

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:**
- **Objective:** Bind the fetched origin/main commit and tree, Python/dependency identities, gitlink pins, environment, provider/hardware/model-serving state, learning/checkpoint contracts, and every named prerequisite to exact source plus test/proof evidence and schema versions.
- **Acceptance subset:** exact-revision; tree-identity; environment-lock; gitlinks; prerequisite-status-enum; source-test-schema-binding; pgir-no-go
- **Predicted files:** docs/architecture/residual_intelligence_inventory/baseline.json, docs/architecture/residual_intelligence_inventory/prerequisite_matrix.json, docs/architecture/residual_intelligence_inventory/pgir_training_gate.json, test/api/residual_intelligence/test_inventory_artifacts.py
- **Predicted symbols:** PrerequisiteFinding; TypedBlocker; baseline and prerequisite manifest schemas
- **Data rights:** Repository source and local qualification receipts are inspection-only; they acquire no training rights without an admitted TrainingCorpusAdmission.
- **Privacy class:** repository_private
- **Effect class:** read_only_analysis_plus_compact_local_evidence
- **Risk class:** R1
- **Resource class:** cpu-standard-local-proof
- **Token budget:** input=32000; output=9000
- **Training budget:** 0; TrainingCorpusAdmission is prohibited in this baseline task
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_inventory_artifacts.py test/api/residual_intelligence/test_contracts_and_ir.py
- **Proof requirements:** Git commit/tree equality, canonical manifest identity, source/test blob bindings, explicit pass/fail/skip/not-run, and no unclassified prerequisite.
- **Rollback:** Remove only task-owned manifests/tests from the isolated worktree; retain failed probe receipts and do not advance the tranche goal.
- **Conflict policy:** Exclusive baseline-manifest writer; all prerequisite roots and sibling repositories are read-only; opaque qualification state serializes.
- **Completion evidence:** Exact current-tree validation passed for committed baseline/prerequisite manifests and focused tests; database acceptance must bind their final tree and producer receipts.

## VRIF-001 Inventory residual model calls and task families

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-000
- **Objective:** Inventory every model invocation in accepted and rejected supervisor trajectories and classify its stage, contracts, cost, validation/outcome, answerability, decision relevance, and authority under the closed 24-family taxonomy.
- **Acceptance subset:** all-required-call-fields; accepted-and-rejected; closed-taxonomy; semantic-family-boundaries; no-prompt-similarity-grouping; authoritative-flag
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/inventory.py, docs/architecture/residual_intelligence_inventory/residual_model_call_inventory.json, test/api/residual_intelligence/test_inventory.py
- **Predicted symbols:** ModelInvocationObservation; ResidualFamilyBoundary; ResidualReasoningInventory; build_inventory
- **Data rights:** Inventory metadata is first-party inspection data, not a training corpus; raw bodies remain excluded unless a future TrainingCorpusAdmission admits their exact identities.
- **Privacy class:** repository_private
- **Effect class:** read_only_analysis_plus_bounded_inventory
- **Risk class:** R1
- **Resource class:** cpu-standard-local-analysis
- **Token budget:** input=30000; output=9000
- **Training budget:** 0; no TrainingCorpusAdmission is requested
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_inventory.py test/api/residual_intelligence/test_inventory_artifacts.py
- **Proof requirements:** Duplicate invocation rejection, complete trajectory disposition, exact closed enum, and family equality across input/output/risk/authority/validation/error/abstention semantics.
- **Rollback:** Revert only inventory module/artifacts/tests; preserve raw trajectory authority and record inventory gaps as typed blockers.
- **Conflict policy:** Own inventory module and compact report only; do not mutate provider ledgers, trajectories, context, proof, or learning stores.
- **Completion evidence:** Exact current-tree validation passed for inventory contracts, closed taxonomy, boundary rejection, and duplicate-call tests; database acceptance remains receipt-gated.

## VRIF-002 Define residual intelligence contracts

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-000
- **Objective:** Define strict provider-free shared enums, canonical identities, prerequisite findings, typed blockers, bounded fields, and recursive rejection of model-created authority, policy, proof, or completion claims.
- **Acceptance subset:** canonical-round-trip; unknown-field-rejection; bounded-fields; nonfinite-rejection; candidate-authority-rejection; no-import-effects
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/contracts.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/__init__.py, test/api/residual_intelligence/test_contracts_and_ir.py
- **Predicted symbols:** ResidualTaskFamily; RiskClass; PrivacyClass; ExpertDisposition; PrerequisiteFinding; TypedBlocker
- **Data rights:** Contract schemas contain no corpus rows; future data use still requires an admitted TrainingCorpusAdmission.
- **Privacy class:** internal
- **Effect class:** provider_free_contract_definition
- **Risk class:** R2
- **Resource class:** cpu-small-hermetic
- **Token budget:** input=24000; output=8000
- **Training budget:** 0; TrainingCorpusAdmission is outside this task
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_contracts_and_ir.py
- **Proof requirements:** Canonical serialization/identity equality, recursive authority/completion-shaped field rejection, cold-import purity, and exact enum population.
- **Rollback:** Revert new provider-free contract files and their tests; no persistent or remote state requires rollback.
- **Conflict policy:** Exclusive owner of base residual contracts; downstream modules depend on this task and may not redefine identity, risk, privacy, or dispositions.
- **Completion evidence:** Exact current-tree validation passed for round trips, canonical IDs, unknown fields, bounds, secret-shaped inputs, candidate-only authority, and critical-risk behavior.

## VRIF-003 Define training corpus admission and rights contracts

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-000, VRIF-002
- **Objective:** Implement TrainingCorpusAdmission with exact source/transformation rights, privacy/tenant/retention, roots, split/holdout/dedup/leakage, tokenizer/compiler, label, negative, adversarial, and environment bindings; fail closed to training_unavailable.
- **Acceptance subset:** complete-admission-record; rights-rejection; privacy-rejection; leakage-audit; credentials-prohibited; pgir-freeze-preserved; training-unavailable
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/rights.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/corpus.py, test/api/residual_intelligence/test_corpus_rights_and_splits.py
- **Predicted symbols:** TrainingCorpusAdmission; LeakageAudit; SourceRight; TransformationRight; TrainingAvailability
- **Data rights:** No implicit rights; only exact documented first-party, synthetic, mutant, counterexample, authorized private, reviewed, or license-compatible public sources can enter an admitted TrainingCorpusAdmission.
- **Privacy class:** repository_private
- **Effect class:** fail_closed_data_governance_contract
- **Risk class:** R4
- **Resource class:** cpu-small-hermetic
- **Token budget:** input=28000; output=9000
- **Training budget:** 0; constructing or admitting a TrainingCorpusAdmission does not authorize training in this task
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_corpus_rights_and_splits.py -k 'admission or rights or unavailable or credential'
- **Proof requirements:** All required roots/producers bind canonically; rights and leakage are conjunctive; credentials/proof-witness-public/private-CoT cases reject; current PGIR identities remain quarantined.
- **Rollback:** Revert contracts/tests only; never manufacture a replacement admission, checkpoint, or promotion.
- **Conflict policy:** Exclusive admission/rights owner; PGIR data and learning-checkpoint stores are read-only; no corpus materialization or training side effect.
- **Completion evidence:** Exact current-tree validation passed for admitted/rejected round trips, rights/privacy failures, leakage audit binding, secret exclusion, and typed training_unavailable.

## VRIF-004 Build first-party trajectory corpus

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-001, VRIF-003
- **Objective:** Build bounded first-party trajectory examples with exact input/context/source/validation/evidence/repository/right/privacy/split identities and reject any positive lacking independent current-tree validation.
- **Acceptance subset:** exact-example-identity; positive-validation; stale-simulated-rejection; authority-safe; rejected-alternatives; proof-test-evidence; no-hidden-tests
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/corpus.py, docs/architecture/residual_intelligence_inventory/corpus_construction.json, test/api/residual_intelligence/test_corpus_rights_and_splits.py, test/api/residual_intelligence/test_inventory_artifacts.py
- **Predicted symbols:** ResidualDistillationExample; ResidualDistillationCorpus; build_first_party_trajectory_corpus
- **Data rights:** Builder accepts only source identities explicitly covered by an admitted TrainingCorpusAdmission; current output is a schema/small fixture manifest, not admitted training data.
- **Privacy class:** repository_private
- **Effect class:** bounded_offline_corpus_construction
- **Risk class:** R3
- **Resource class:** cpu-medium-local-io
- **Token budget:** input=28000; output=9000
- **Training budget:** 0; no admitted TrainingCorpusAdmission currently authorizes training
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_corpus_rights_and_splits.py test/api/residual_intelligence/test_inventory_artifacts.py
- **Proof requirements:** Source/context/current-tree identities, independent validator result, proof/test evidence where required, no authority violation, and no stale/simulated status.
- **Rollback:** Withdraw task-owned manifests/fixtures by identity; retain withdrawal record; do not modify original trajectories.
- **Conflict policy:** Corpus builder owns only residual corpus records; trajectory/provider/proof stores are read-only and raw private bodies are never copied into Git.
- **Completion evidence:** Exact current-tree validation passed for corpus identity, independently validated positives, invalid positives, hidden-test exclusion, and training-unavailable disposition.

## VRIF-005 Build synthetic and adversarial corpus

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-003
- **Objective:** Construct small rights-clear synthetic fixtures, mechanically generated counterexamples, and adversarial-assurance mutant records covering plausible errors, authority/completion violations, stale reuse, missing validation, injection, weakening, leakage, and cross-repository transfer.
- **Acceptance subset:** negative-policy; counterexample-required; mutant-lineage; injection; test-weakening; fake-proof; unsafe-nonabstention; cross-repository-denial
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/corpus.py, benchmarks/agent_supervisor/residual_intelligence/manifest.json, benchmarks/agent_supervisor/residual_intelligence/tranche1_contract_cases.json, test/api/residual_intelligence/test_inventory_artifacts.py
- **Predicted symbols:** CorpusSourceKind; LabelDisposition; build_synthetic_adversarial_corpus
- **Data rights:** Generated fixtures use first-party synthetic rights recorded by an admitted TrainingCorpusAdmission before training; no quarantined PGIR or private body is consumed.
- **Privacy class:** internal
- **Effect class:** bounded_synthetic_fixture_generation
- **Risk class:** R2
- **Resource class:** cpu-small-hermetic
- **Token budget:** input=24000; output=8000
- **Training budget:** 0; synthetic generation alone is not an admitted TrainingCorpusAdmission or training permission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_corpus_rights_and_splits.py test/api/residual_intelligence/test_inventory_artifacts.py
- **Proof requirements:** Deterministic generator/version identity, mutant family grouping, negative/counterexample presence, and no source-rights or privacy downgrade.
- **Rollback:** Delete only small generated fixtures/manifests from the isolated worktree and retain generator failure evidence.
- **Conflict policy:** Own synthetic/adversarial fixture paths only; Adversarial Assurance remains authoritative and is invoked through an adapter in VRIF-028.
- **Completion evidence:** Exact current-tree validation passed for synthetic/adversarial example construction, negative constraints, canonical identity, and rights/privacy preservation.

## VRIF-006 Implement lineage-safe semantic splits

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-004, VRIF-005
- **Objective:** Split by connected semantic lineage rather than rows, keeping repository state, task/failure/source/procedure/mutant/proof groups together and excluding hidden-test knowledge from train/development.
- **Acceptance subset:** union-find-lineage; deterministic-split; group-cohesion; forced-holdout; adversarial-partition; zero-leakage; hidden-test-protection
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/splits.py, benchmarks/agent_supervisor/residual_intelligence/synthetic_split_manifest.json, test/api/residual_intelligence/test_corpus_rights_and_splits.py, test/api/residual_intelligence/test_inventory_artifacts.py
- **Predicted symbols:** SemanticSplitPolicy; SemanticSplitManifest; semantic_lineage_split; assert_training_view_excludes_hidden
- **Data rights:** Split construction cannot expand rights; every component inherits the strictest source disposition from its admitted TrainingCorpusAdmission.
- **Privacy class:** repository_private
- **Effect class:** deterministic_offline_partitioning
- **Risk class:** R3
- **Resource class:** cpu-medium-memory
- **Token budget:** input=24000; output=7000
- **Training budget:** 0; a passing split is necessary but insufficient for TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_corpus_rights_and_splits.py -k 'split or leakage or hidden'
- **Proof requirements:** Connected-component partition proof, deterministic root identity, no group across partitions, no train-hidden edge, and passing leakage audit.
- **Rollback:** Revoke only split manifest/root; original examples remain unchanged and no training epoch may reference the revoked root.
- **Conflict policy:** Exclusive split module owner; corpus records are immutable inputs and benchmark freezing remains VRIF-030.
- **Completion evidence:** Exact current-tree validation passed for deterministic grouping, linked-lineage cohesion, forced partitions, hidden-test protection, and leakage detection.

## VRIF-007 Implement compact ResidualIntelligenceIR

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-002, VRIF-003
- **Objective:** Implement strict ResidualTaskInput and ResidualTaskOutput envelopes with exact CIDs, compact bounded features/outputs, risk and validation policy, calibration, abstention, evidence, reason codes, and immutable candidate-only semantics.
- **Acceptance subset:** strict-input; strict-output; canonical-cid; bounded-compact-features; allowed-output; candidate-only; critical-risk-validation; secret-rejection
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/residual_ir.py, test/api/residual_intelligence/test_contracts_and_ir.py
- **Predicted symbols:** ResidualTaskInput; ResidualTaskOutput; ResidualIntelligenceIR
- **Data rights:** IR references content identities and bounded admitted features; payload use in training still requires an admitted TrainingCorpusAdmission.
- **Privacy class:** repository_private
- **Effect class:** provider_free_typed_ir
- **Risk class:** R3
- **Resource class:** cpu-small-hermetic
- **Token budget:** input=26000; output=8000
- **Training budget:** 0; TrainingCorpusAdmission is not exercised
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_contracts_and_ir.py -k 'ir or candidate or critical or secret'
- **Proof requirements:** Canonical round trip/identity, unknown-field and size rejection, allowed-output membership, candidate_only always true, and R4/R5 validation-required behavior.
- **Rollback:** Revert IR module/tests; dependent tasks remain waiting and no serialized record is promoted.
- **Conflict policy:** Exclusive IR owner; base enum/content identity contracts are reused, not duplicated.
- **Completion evidence:** Exact current-tree validation passed for input/output round trips, IDs, bounds, allowed outputs, candidate-only enforcement, and R4/R5 conservative behavior.

## VRIF-008 Implement structured-output grammars

- **Status:** completed
- **Goal:** VRIF-G011
- **Depends on:** VRIF-002, VRIF-007
- **Objective:** Define one closed strict grammar for every residual family with required fields, enumerations, list/output bounds, abstention, and post-decode validation; parse failures yield invalid_output without prose recovery.
- **Acceptance subset:** 24-family-coverage; duplicate-key-rejection; unknown-field-rejection; max-output; enum-bounds; abstention; invalid-output; no-prose-fallback
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/structured_decoding.py, test/api/residual_intelligence/test_structured_decoding.py
- **Predicted symbols:** ExpertGrammar; decode_structured_output; DEFAULT_GRAMMARS; DecodeStatus.INVALID_OUTPUT; grammar_for
- **Data rights:** Grammar definitions have no data rights effect; decoded records require an admitted TrainingCorpusAdmission before training use.
- **Privacy class:** internal
- **Effect class:** deterministic_strict_decode
- **Risk class:** R3
- **Resource class:** cpu-small-hermetic
- **Token budget:** input=26000; output=8000
- **Training budget:** 0; no TrainingCorpusAdmission or model inference occurs
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_structured_decoding.py
- **Proof requirements:** Exact taxonomy-to-grammar bijection, strict JSON duplicate/unknown rejection, bounded lists/text, candidate-only output, and parse failure as typed invalid_output.
- **Rollback:** Revert grammar module/tests and keep learned decoding disabled; do not interpret stored prose.
- **Conflict policy:** Exclusive grammar owner; future decoders extend through registered family schemas and may not accept arbitrary fields.
- **Completion evidence:** Exact current-tree validation passed for all-family compilation, valid/invalid decode, duplicate and unknown fields, payload bounds, and abstention.

## VRIF-009 Implement deterministic and linear baselines

- **Status:** todo
- **Goal:** VRIF-G021
- **Depends on:** VRIF-006, VRIF-007, VRIF-008
- **Objective:** Implement exact lookup, declarative rules, deterministic rankings, and bounded linear/logistic baseline contracts before any learned or remote route, with complete abstention and cost receipts.
- **Acceptance subset:** exact-first; procedure-compatible; deterministic-rule; bounded-linear; stable-features; no-model-call-on-precondition-failure; denominators
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/baselines.py, test/api/residual_intelligence/test_baselines.py
- **Predicted symbols:** DeterministicResidualExpert; LinearResidualExpert; BaselinePrediction; BaselineEvaluation
- **Data rights:** Evaluation fixtures must be synthetic or covered by an admitted TrainingCorpusAdmission; rules cannot memorize or expose private bodies.
- **Privacy class:** repository_private
- **Effect class:** deterministic_candidate_inference
- **Risk class:** R2
- **Resource class:** cpu-small-batch
- **Token budget:** input=30000; output=9000
- **Training budget:** 0 until admitted TrainingCorpusAdmission; any later linear fit is capped at examples=10000, cpu_seconds=1800, checkpoints=1
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_baselines.py
- **Proof requirements:** Exact/procedure/deterministic precedence, reproducible features and coefficients, candidate-only output, critical boundary abstention, and no provider invocation on early exit.
- **Rollback:** Revoke baseline artifact/rules by exact version and return affected families to the previous cascade route.
- **Conflict policy:** Own baselines module; consume IR/grammar/splits read-only; no router or model-serving edits.

## VRIF-010 Implement task-family expert specifications

- **Status:** todo
- **Goal:** VRIF-G021
- **Depends on:** VRIF-001, VRIF-002, VRIF-007
- **Objective:** Define each expert's shared semantic family boundary, class, input/output schemas, grammar, token/output limits, risk ceiling, privacy, capabilities, validation, errors, and abstention behavior.
- **Acceptance subset:** exact-family-boundary; class-A-through-E; smallest-form-order; closed-schemas; risk-ceiling; validator-required; no-prose-default
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/task_families.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/expert_specs.py, test/api/residual_intelligence/test_expert_specs.py
- **Predicted symbols:** ResidualTaskFamilySpec; ResidualExpertSpec; ExpertClass; ModelSizePolicy
- **Data rights:** Specifications carry no examples; any evaluation/training dataset reference must resolve to an admitted TrainingCorpusAdmission.
- **Privacy class:** internal
- **Effect class:** provider_free_capability_contract
- **Risk class:** R3
- **Resource class:** cpu-small-hermetic
- **Token budget:** input=30000; output=9000
- **Training budget:** 0; TrainingCorpusAdmission references are declarative only
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_expert_specs.py
- **Proof requirements:** Every taxonomy family maps to explicit semantics/error/abstention/validation; unsupported family-risk pairs reject; larger form needs a routing-changing quality delta.
- **Rollback:** Revert spec registry/version and force all affected expert artifacts stale until compatible specs are restored.
- **Conflict policy:** Exclusive spec registry owner; inventory and IR contracts are immutable dependencies; no runtime route mutation.

## VRIF-011 Implement calibration and abstention contracts

- **Status:** todo
- **Goal:** VRIF-G021
- **Depends on:** VRIF-002, VRIF-006, VRIF-007
- **Objective:** Implement selective prediction with separate calibration groups for family, repository, language, framework, risk, model, quantization, hardware, and context tier and six closed dispositions.
- **Acceptance subset:** group-key; current-evidence; accept-abstain-reject-ood-capability-validation; no-global-threshold; r4-r5-proposal; self-threshold-rejection
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/calibration.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/abstention.py, test/api/residual_intelligence/test_calibration_abstention.py
- **Predicted symbols:** CalibrationGroup; CalibrationEvidence; AbstentionDecision; SelectivePredictionPolicy
- **Data rights:** Calibration rows must be held-out and covered by an admitted TrainingCorpusAdmission; groups expose metrics and CIDs, not private bodies.
- **Privacy class:** repository_private
- **Effect class:** bounded_route_gate_decision
- **Risk class:** R4
- **Resource class:** cpu-medium-statistical
- **Token budget:** input=30000; output=9000
- **Training budget:** 0 until admitted TrainingCorpusAdmission; later calibration cap examples=20000, cpu_seconds=3600, candidate_thresholds=128
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_calibration_abstention.py
- **Proof requirements:** Group isolation, current evidence, no critical false accept, R4/R5 candidate-only, threshold changes through authorized CAS with rollback, and model self-modification rejection.
- **Rollback:** Restore prior admitted calibration root/threshold by authorized CAS and force shadow/abstention during re-evaluation.
- **Conflict policy:** Calibration and abstention modules are owned together; promotion changes wait for VRIF-031 and cannot be made by model output.

## VRIF-012 Implement OOD and boundary detection

- **Status:** todo
- **Goal:** VRIF-G021
- **Depends on:** VRIF-002, VRIF-006, VRIF-007
- **Objective:** Implement bounded advisory OOD signals and independent conservative family, schema, effect, authority, repository, calibration, capability, and context boundary checks.
- **Acceptance subset:** feature-range; unknown-schema-operation-repository; unseen-effects-authority; disagreement; calibration-absence; context-incomplete; conservative-high-risk
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/ood.py, test/api/residual_intelligence/test_ood.py
- **Predicted symbols:** OODSignal; OODAssessment; BoundaryContract; assess_out_of_distribution
- **Data rights:** Reference distributions require an admitted TrainingCorpusAdmission; compact statistics cannot contain recoverable private source.
- **Privacy class:** repository_private
- **Effect class:** advisory_detection_plus_hard_contract_gate
- **Risk class:** R4
- **Resource class:** cpu-medium-statistical
- **Token budget:** input=28000; output=8000
- **Training budget:** 0 until admitted TrainingCorpusAdmission; later reference fitting cap examples=20000, cpu_seconds=1800
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_ood.py
- **Proof requirements:** Missing OOD detection never establishes safety; high-risk unknown/missing group/context independently abstains; known in-boundary fixtures remain eligible.
- **Rollback:** Revert OOD artifact/version and widen abstention; never restore autonomous eligibility without current group evidence.
- **Conflict policy:** Own OOD module only; calibration and expert specifications are read-only inputs; router integration waits for VRIF-013.

## VRIF-013 Implement expert cascade router

- **Status:** todo
- **Goal:** VRIF-G021
- **Depends on:** VRIF-009, VRIF-010, VRIF-011, VRIF-012
- **Objective:** Route through cache, verified procedure, deterministic rule, local specialists/general, remote standard/strong, and human using hard family/risk/capability/privacy/validation constraints plus budget and expected decision value.
- **Acceptance subset:** deterministic-first; procedure-first; hard-rejections; privacy-route; hardware-provider-health; no-simulation; candidate-evidence; safe-fallback
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/router.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/cascade.py, test/api/residual_intelligence/test_router.py
- **Predicted symbols:** ResidualExpertRouter; ResidualRouteRequest; ResidualRouteDecision; ResidualCascade
- **Data rights:** Remote routes are allowed only when privacy labels and provider authorization in the admitted TrainingCorpusAdmission/inference policy permit exact compact inputs.
- **Privacy class:** repository_private
- **Effect class:** typed_candidate_routing
- **Risk class:** R4
- **Resource class:** cpu-small-control
- **Token budget:** input=36000; output=10000
- **Training budget:** 0; router evaluation references only admitted TrainingCorpusAdmission fixtures
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_router.py
- **Proof requirements:** Exact route ordering, all candidates/hard rejections recorded, no out-of-bound/risk/unavailable/private route, required validation preserved, and human fallback reachable.
- **Rollback:** Revert router policy/version to previous admitted route table; drain fenced requests and preserve disagreement/attempt receipts.
- **Conflict policy:** Exclusive cascade owner; provider router/model serving/procedure compiler remain canonical dependencies and are accessed through typed adapters.

## VRIF-014 Implement local classification and ranking experts

- **Status:** todo
- **Goal:** VRIF-G021
- **Depends on:** VRIF-009, VRIF-010, VRIF-011, VRIF-012
- **Objective:** Implement first local exact/linear/small-ranking candidates for admitted low/medium-risk classification and ranking families with batching, abstention, OOD, calibration, and independent validation.
- **Acceptance subset:** smallest-reliable-form; classification; ranking; batch; calibrated-abstention; ood; held-out-quality; candidate-only; training-unavailable
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/local_experts.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/distillation.py, test/api/residual_intelligence/test_local_experts.py
- **Predicted symbols:** LocalClassificationExpert; LocalRankingExpert; BatchedExpertRequest; ExpertEvaluation
- **Data rights:** Fitting is forbidden until an admitted TrainingCorpusAdmission binds rights, privacy, splits, tokenizer/compiler, labels, and environment; synthetic deterministic fixtures may test contracts.
- **Privacy class:** repository_private
- **Effect class:** local_candidate_inference
- **Risk class:** R3
- **Resource class:** cpu-medium-batch
- **Token budget:** input=40000; output=12000
- **Training budget:** 0 while training_unavailable; after admitted TrainingCorpusAdmission cap examples=50000, steps=10000, wall_seconds=7200, gpu_seconds=0, checkpoints=3
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_local_experts.py
- **Proof requirements:** Current held-out/adversarial evaluation by group, structured validity, no critical false accepts, complete denominators, safe abstention, and independent validators decide acceptance.
- **Rollback:** Revoke candidate artifact and route family to deterministic/remote fallback; preserve evaluation and rejected-model lineage.
- **Conflict policy:** Own local expert/distillation modules; model-serving and packaging integration waits for VRIF-024/025; large artifacts stay outside Git.

## VRIF-015 Implement constrained structured-decoder expert

- **Status:** todo
- **Goal:** VRIF-G021
- **Depends on:** VRIF-008, VRIF-010, VRIF-011, VRIF-012
- **Objective:** Implement one family-bounded grammar-constrained structured specialist for typed procedure-hole or patch-sketch candidates with strict parsing, output bounds, abstention, and no arbitrary prose/fields.
- **Acceptance subset:** grammar-constrained; strict-post-parse; invalid-output; max-length; candidate-only; bounded-context; abstain-escalate; no-freeform-authority
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/structured_specialist.py, test/api/residual_intelligence/test_structured_specialist.py
- **Predicted symbols:** ConstrainedStructuredExpert; StructuredDecodeRequest; StructuredDecodeResult
- **Data rights:** Fine-tuning is forbidden without an admitted TrainingCorpusAdmission; contract tests use rights-clear synthetic fixtures and no private model download.
- **Privacy class:** repository_private
- **Effect class:** local_structured_candidate_generation
- **Risk class:** R4
- **Resource class:** cpu-gpu-optional-bounded
- **Token budget:** input=40000; output=12000
- **Training budget:** 0 while training_unavailable; after admitted TrainingCorpusAdmission cap examples=30000, steps=8000, wall_seconds=10800, gpu_seconds=7200, checkpoints=3
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_structured_specialist.py
- **Proof requirements:** Decoder cannot emit arbitrary fields/shell/policy/authority/completion; all parse failures are invalid_output; admitted validator checks every non-abstained candidate.
- **Rollback:** Revoke specialist artifact, unload weights, reclaim resources, and return family to narrower or remote candidate route.
- **Conflict policy:** Own structured specialist adapter only; grammar remains VRIF-008 authority and model serving remains existing canonical owner.

## VRIF-016 Integrate procedure-hole resolution

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-013, VRIF-015
- **Objective:** Connect experts only to declared typed procedure holes under current compiler preconditions, bounds, validation, source identity, and authority; expose a narrow inactive adapter when the compiler is unavailable.
- **Acceptance subset:** compiler-capability; typed-hole; preconditions; exact-procedure-root; candidate-only; validator-decides; repeated-hole-rule-nomination
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/procedure_experts.py, test/api/residual_intelligence/test_procedure_experts.py
- **Predicted symbols:** ProcedureHoleExpertAdapter; ProcedureHoleResolution; ProcedureHoleCapability
- **Data rights:** Successful/failed hole records become examples only under an admitted TrainingCorpusAdmission covering exact source, validation, privacy, and retention.
- **Privacy class:** repository_private
- **Effect class:** bounded_procedure_candidate_integration
- **Risk class:** R4
- **Resource class:** cpu-medium-local-proof
- **Token budget:** input=36000; output=10000
- **Training budget:** 0; adapter records are not training data without TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_procedure_experts.py
- **Proof requirements:** Current compiler capability, exact hole/precondition/root, grammar and line/list bounds, procedure validator result, and no procedure authority/validation mutation.
- **Rollback:** Disable adapter/capability, revoke candidate records, and return holes to existing compiler escalation without replacing the compiler.
- **Conflict policy:** Own narrow residual adapter only; procedure compiler files and authority are read-only unless their canonical owner exposes an explicit extension point.

## VRIF-017 Integrate proof and tactic experts

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-013, VRIF-014
- **Objective:** Integrate local candidate rankings for premises, lemmas, tactics, proof branches, counterexamples, and proof-failure attribution while leaving the actual prover and exact obligation/environment authoritative.
- **Acceptance subset:** premise-rank; lemma-rank; tactic-rank; branch-rank; counterexample-class; obligation-binding; prover-check; failed-tactic-lineage
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/proof_experts.py, test/api/residual_intelligence/test_proof_experts.py
- **Predicted symbols:** ProofExpertAdapter; TacticCandidate; PremiseRanking; ProofCandidateReceipt
- **Data rights:** Proof/tactic traces enter a corpus only under an admitted TrainingCorpusAdmission; proof witnesses never enter public artifacts or unauthorized experts.
- **Privacy class:** proof_witness
- **Effect class:** proof_candidate_nomination
- **Risk class:** R5
- **Resource class:** cpu-medium-prover
- **Token budget:** input=40000; output=10000
- **Training budget:** 0 while training_unavailable; admitted TrainingCorpusAdmission later caps examples=30000, prover_seconds=14400, checkpoints=2
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_proof_experts.py
- **Proof requirements:** Actual prover checks exact obligation/source/assumptions/environment; suggestion is never labeled proof; proof omission and stale obligation reject.
- **Rollback:** Disable expert nomination and restore existing prover search order; retain successful/failed tactic evidence under its original privacy label.
- **Conflict policy:** Own proof-expert adapter only; prover, tacticians, proof cache, verification planner, and sealer remain canonical owners.

## VRIF-018 Integrate patch-sketch experts

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-013, VRIF-015
- **Objective:** Generate PatchSketchIR, ProcedureHoleResolution, RefactorOperatorSelection, and TestSketchIR candidates with deterministic rendering or an existing strict proposal envelope.
- **Acceptance subset:** exact-files-symbols; allowed-paths; max-lines; no-binary; no-test-deletion; no-validation-weakening; no-key-authority-shell; isolated-worktree; rollback
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/patch_experts.py, test/api/residual_intelligence/test_patch_experts.py
- **Predicted symbols:** PatchSketchIR; TestSketchIR; PatchExpertAdapter; PatchScopePolicy
- **Data rights:** Source/sketch pairs require an admitted TrainingCorpusAdmission and may not include credentials, private chain-of-thought, or unauthorized tenant data.
- **Privacy class:** repository_private
- **Effect class:** isolated_patch_candidate_nomination
- **Risk class:** R5
- **Resource class:** cpu-medium-isolated-worktree
- **Token budget:** input=44000; output=12000
- **Training budget:** 0 while training_unavailable; admitted TrainingCorpusAdmission later caps examples=20000, steps=6000, changed_lines_per_example=200, checkpoints=2
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_patch_experts.py
- **Proof requirements:** Exact scope and base tree, strict IR parse, prohibited-effect checks, predetermined tests/proofs/effects, independent validation, and merge result.
- **Rollback:** Discard fenced task worktree or reverse exact admitted patch; revoke sketch artifact and preserve validator/counterexample evidence.
- **Conflict policy:** Own patch IR/adapter only; worktree, renderer, repair, merge, effect, authority, and validation systems remain canonical.

## VRIF-019 Implement teacher-disagreement handling

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-005, VRIF-014, VRIF-015
- **Objective:** Preserve all teacher outputs/provenance, run admitted independent validators, attach counterexamples, and retain unresolved cases as ambiguous, inconclusive, or human_review_required without confidence voting.
- **Acceptance subset:** all-teachers-preserved; independent-validation; counterexample-separation; unresolved-labels; no-confidence-ground-truth; no-agreement-reward
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/labels.py, test/api/residual_intelligence/test_teacher_disagreement.py
- **Predicted symbols:** TeacherCandidate; TeacherDisagreement; LabelDisposition; resolve_teacher_disagreement
- **Data rights:** Teacher inputs/outputs remain subject to the admitted TrainingCorpusAdmission and provider privacy authorization; unresolved private bodies are not published.
- **Privacy class:** repository_private
- **Effect class:** offline_label_candidate_reconciliation
- **Risk class:** R4
- **Resource class:** cpu-medium-validation
- **Token budget:** input=30000; output=9000
- **Training budget:** 0; disagreement records cannot train without TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_teacher_disagreement.py
- **Proof requirements:** Provenance completeness, independent producer identities, unresolved preservation, no teacher confidence/order winner, and counterexample lineage.
- **Rollback:** Revert reconciliation records/adapter; restore all original teacher candidates without selecting a winner.
- **Conflict policy:** Own label/disagreement module; validators and human-review authority remain external and immutable inputs.

## VRIF-020 Implement proof-grounded label production

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-004, VRIF-005, VRIF-017, VRIF-019
- **Objective:** Produce labels only from admitted type/static/test/proof/policy/authority/effect/merge/human/current-tree evidence and retain ambiguous or inconclusive cases rather than force clean labels.
- **Acceptance subset:** independent-producers; exact-tree; positive-and-negative-evidence; no-model-agreement-reward; ambiguous; inconclusive; human-review-required
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/labels.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/distillation.py, test/api/residual_intelligence/test_label_production.py
- **Predicted symbols:** IndependentLabelProducer; ProofGroundedLabel; LabelEvidencePolicy
- **Data rights:** Label production cannot admit data; every resulting example still requires a TrainingCorpusAdmission with exact producers, rights, privacy, retention, and splits.
- **Privacy class:** repository_private
- **Effect class:** offline_independent_label_production
- **Risk class:** R4
- **Resource class:** cpu-medium-local-proof
- **Token budget:** input=36000; output=10000
- **Training budget:** 0; labels do not authorize training absent TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_label_production.py
- **Proof requirements:** Each positive maps to independent current evidence; each negative/counterexample is retained; authority/test/proof/validation violations cannot be rewarded.
- **Rollback:** Withdraw derived label CIDs, preserve source producer receipts, and invalidate dependent corpus/checkpoint candidates.
- **Conflict policy:** Shares labels.py only after VRIF-019; proof/test/policy/authority/effect/human systems are read-only producers.

## VRIF-021 Implement active-learning planner

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-006, VRIF-011, VRIF-019, VRIF-020
- **Objective:** Rank bounded acquisition actions by uncertainty plus abstention, disagreement, validation failure, novelty, task frequency, token/human cost, and expected route improvement under authority/resource budgets.
- **Acceptance subset:** impact-aware-selection; no-uncertainty-only; acquisition-actions; authority-class; resource-budget; idempotency; human-review-cap
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/active_learning.py, test/api/residual_intelligence/test_active_learning.py
- **Predicted symbols:** ResidualActiveLearningPlanner; AcquisitionCandidate; AcquisitionAction; AcquisitionBudget
- **Data rights:** Selection exposes no new data right; acquisition into a corpus requires an admitted TrainingCorpusAdmission before use.
- **Privacy class:** repository_private
- **Effect class:** bounded_offline_acquisition_planning
- **Risk class:** R3
- **Resource class:** cpu-small-planning
- **Token budget:** input=30000; output=9000
- **Training budget:** 0; active-learning selection does not create TrainingCorpusAdmission or execute training
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_active_learning.py
- **Proof requirements:** Deterministic score factors/denominators, minimum expected impact, bounded action cost/authority, no production exploration, and idempotent selection.
- **Rollback:** Cancel unstarted acquisitions, fence active requests, retain cost/evidence receipts, and restore prior acquisition plan root.
- **Conflict policy:** Own active-learning planner; validators/providers/humans are invoked later through authorized existing operations, never shell commands.

## VRIF-022 Implement continual-learning epoch contracts

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-003, VRIF-006, VRIF-011, VRIF-021
- **Objective:** Define bounded offline epochs binding parent model, architecture, tokenizer/vocabulary, corpus/split, curriculum/loss/optimizer/scheduler/seed, environment, code/compiler, evaluation, and hard resource limits.
- **Acceptance subset:** exact-epoch-binding; offline-only; example-step-wall-gpu-spend-candidate-checkpoint-review-bounds; compatible-resume; no-promotion
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/continual_learning.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/training_plan.py, test/api/residual_intelligence/test_continual_learning.py
- **Predicted symbols:** ContinualLearningEpoch; TrainingEpochLimits; TrainingPlan; ResumeCompatibility
- **Data rights:** An epoch cannot enter planned/running without one admitted TrainingCorpusAdmission whose roots exactly match the epoch.
- **Privacy class:** repository_private
- **Effect class:** bounded_offline_training_contract
- **Risk class:** R4
- **Resource class:** cpu-gpu-bounded-offline
- **Token budget:** input=36000; output=10000
- **Training budget:** 0 while training_unavailable; admitted TrainingCorpusAdmission cap examples=50000, steps=10000, wall_seconds=14400, gpu_seconds=10800, spend_usd=25, candidates=3, checkpoints=4, human_reviews=50
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_continual_learning.py
- **Proof requirements:** Exact lineage equality for resume, bounds cannot widen during run, production episodes are candidate-only, and trained/resumed/checkpointed state cannot imply promotion.
- **Rollback:** Cancel fenced epoch, reclaim resources, retain partial checkpoint as non-promotable or corrupt, and leave current production route unchanged.
- **Conflict policy:** Own epoch/training plan contracts; no new training framework, scheduler, artifact store, or checkpoint store.

## VRIF-023 Integrate learning-checkpoint lineage

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-022
- **Objective:** Reuse LearningCheckpointBinding for exact parent/data/split/tokenizer/code/compiler/environment/evaluation lineage, compatible resume, corruption/withdrawal invalidation, and explicit separation from promotion.
- **Acceptance subset:** canonical-checkpoint-binding; compatible-resume; incompatible-reject; resume-without-promotion; corruption; withdrawal; stale-no-promotion
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/checkpoint.py, test/api/residual_intelligence/test_checkpoint_lineage.py
- **Predicted symbols:** ResidualCheckpointAdapter; ExpertCheckpointLineage; validate_residual_resume
- **Data rights:** Checkpoint lineage must reference the same admitted TrainingCorpusAdmission and propagate corpus withdrawal/privacy retention decisions.
- **Privacy class:** repository_private
- **Effect class:** immutable_checkpoint_lineage_integration
- **Risk class:** R4
- **Resource class:** cpu-io-medium
- **Token budget:** input=30000; output=9000
- **Training budget:** 0; TrainingCorpusAdmission is checked but no epoch runs
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_checkpoint_lineage.py
- **Proof requirements:** Byte/content identity, complete parent lineage, exact-compatible resume, corruption detection, withdrawn corpus rejection, and zero checkpoint-created promotion authority.
- **Rollback:** Revoke residual checkpoint binding and fall back to parent/non-learned route; never alter canonical checkpoint history.
- **Conflict policy:** Own narrow checkpoint adapter; runtime/learning_checkpoint.py and artifact storage stay canonical and are not duplicated.

## VRIF-024 Implement quantization and packaging qualification

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-011, VRIF-012, VRIF-014, VRIF-015, VRIF-023
- **Objective:** Package expert architecture/weights/tokenizer/quantization/runtime/operators/hardware/environment/contracts/evaluation identities and independently qualify every quantized artifact.
- **Acceptance subset:** immutable-package; no-weights-in-git; quantization-re-evaluation; hardware-live-qualified; warm-cold-latency; operator-compatibility; capability-unavailable
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/packaging.py, test/api/residual_intelligence/test_packaging.py
- **Predicted symbols:** PackagedExpert; QuantizationQualification; ExpertRuntimeManifest
- **Data rights:** Packaged artifacts bind the originating admitted TrainingCorpusAdmission and privacy/export policy; proof witnesses/private text cannot enter public packages.
- **Privacy class:** repository_private
- **Effect class:** managed_artifact_packaging_qualification
- **Risk class:** R4
- **Resource class:** cpu-gpu-hardware-qualified
- **Token budget:** input=36000; output=10000
- **Training budget:** 0; packaging cannot create or replace TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_packaging.py
- **Proof requirements:** Full/quantized independent comparison by calibration group, exact artifact/runtime/operator/hardware roots, no regression beyond approved bounds, and unavailable hardware returns capability_unavailable.
- **Rollback:** Unload/revoke package identity, reclaim resources, and route to prior qualified artifact or fallback; keep weights outside Git.
- **Conflict policy:** Own residual package manifests/adapters only; existing model-serving, hardware capability, artifact, and scheduler authorities remain canonical.

## VRIF-025 Integrate shared model serving and batching

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-013, VRIF-014, VRIF-015, VRIF-024
- **Objective:** Submit compatible compact requests to existing model-serving/batching and scheduler authorities with bounded queues, shared immutable weights/tokenization, deterministic reclamation, and no simulated fallback.
- **Acceptance subset:** batch-compatible; no-duplicate-weights; bounded-queue; resource-schedule; deterministic-reclaim; immutable-cache; warm-cold-metrics; safe-unavailable
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/runtime.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/cascade.py, test/api/residual_intelligence/test_runtime_batching.py
- **Predicted symbols:** ResidualInferenceRuntime; ExpertBatch; ExpertResourceLease; BatchInferenceReceipt
- **Data rights:** Inference payloads obey privacy/provider policy; training capture is disabled unless an admitted TrainingCorpusAdmission separately covers exact episodes.
- **Privacy class:** repository_private
- **Effect class:** bounded_local_inference_runtime
- **Risk class:** R4
- **Resource class:** cpu-gpu-scheduled-batch
- **Token budget:** input=40000; output=11000
- **Training budget:** 0; runtime episode capture cannot bypass TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_runtime_batching.py
- **Proof requirements:** Batch semantic compatibility, resource lease/fence, queue bounds, no duplicate loads, deterministic teardown, provider/hardware unavailable routes, and simulation rejection.
- **Rollback:** Drain/fence runtime queues, unload residual artifacts, reclaim resources, and restore previous cascade without dropping receipts.
- **Conflict policy:** Own residual runtime adapter; existing serving/batching/provider/scheduler modules change only through their supported typed interface.

## VRIF-026 Implement expert drift, demotion, and revocation

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-011, VRIF-012, VRIF-023, VRIF-024, VRIF-025
- **Objective:** Detect contract/family/repository/calibration/hardware/quantization/validation/abstention/false-accept/token/procedure/architecture drift and enforce candidate, shadow, promoted, degraded, stale, revoked, superseded, or rejected state.
- **Acceptance subset:** drift-signals; stale-unroutable; demotion; wider-abstention; shadow-only; reevaluation; retraining-proposal; revocation
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/drift.py, test/api/residual_intelligence/test_drift.py
- **Predicted symbols:** ExpertDriftMonitor; ExpertState; DriftEvent; DriftDisposition
- **Data rights:** Drift evidence uses metrics/CIDs; retraining proposals require a new or current admitted TrainingCorpusAdmission and cannot silently retain withdrawn rows.
- **Privacy class:** repository_private
- **Effect class:** fenced_expert_lifecycle_control
- **Risk class:** R5
- **Resource class:** cpu-small-control
- **Token budget:** input=34000; output=10000
- **Training budget:** 0; drift may propose but cannot create TrainingCorpusAdmission or train
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_drift.py
- **Proof requirements:** Every drift root is exact/current, stale experts reject autonomous routing, thresholds never silently persist across group changes, and demotion/revocation CAS has rollback.
- **Rollback:** Authorized CAS restores only the prior still-current qualified state; otherwise keep shadow/abstention and escalate review.
- **Conflict policy:** Exclusive expert lifecycle owner; promotion authority remains VRIF-031/control service, and model output cannot modify state.

## VRIF-027 Implement privacy and information-flow gates

- **Status:** todo
- **Goal:** VRIF-G031
- **Depends on:** VRIF-003, VRIF-013, VRIF-025, VRIF-026
- **Objective:** Enforce public/internal/repository/tenant/matter/credential/personal/health/legal/proof-witness labels across admission, inference, providers, artifacts, reports, withdrawal, and future epochs.
- **Acceptance subset:** credentials-never-train; proof-witness-no-public; private-provider-authorization; tenant-no-global; cid-summary-report; withdrawal-propagation; provenance-complete
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/privacy.py, ipfs_accelerate_py/agent_supervisor/residual_intelligence/rights.py, test/api/residual_intelligence/test_privacy.py
- **Predicted symbols:** InformationFlowPolicy; PrivacyRouteDecision; CorpusWithdrawal; DeclassificationAuthority
- **Data rights:** Only explicit TrainingCorpusAdmission and scoped declassification can permit reuse; credentials, private chain-of-thought, and unauthorized legal/private records remain denied.
- **Privacy class:** matter_confidential
- **Effect class:** hard_information_flow_gate
- **Risk class:** R5
- **Resource class:** cpu-small-security
- **Token budget:** input=36000; output=10000
- **Training budget:** 0; privacy policy cannot self-authorize TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_privacy.py
- **Proof requirements:** End-to-end taint/route tests, unauthorized remote denial, artifact/report redaction, no memorized source reproduction, withdrawal invalidates future epochs/packages, and exact provenance.
- **Rollback:** Fail closed to local/no-training/no-export routes, revoke affected artifacts/epochs, and preserve deletion/withdrawal audit receipts.
- **Conflict policy:** Own privacy adapter/policy; credentials/providers/artifacts/checkpoints/control remain canonical owners and sibling/private systems are not mutated.

## VRIF-028 Run adversarial assurance campaign

- **Status:** todo
- **Goal:** VRIF-G041
- **Depends on:** VRIF-016, VRIF-017, VRIF-018, VRIF-019, VRIF-020, VRIF-021, VRIF-022, VRIF-023, VRIF-024, VRIF-025, VRIF-026, VRIF-027
- **Objective:** Use the existing Adversarial Assurance authority to attack family/risk/effect/test/proof/cache/procedure/abstention/injection/confidence/staleness/quantization/disagreement/leakage/privacy/authority/completion boundaries.
- **Acceptance subset:** all-mutant-families; critical-zero-escape; false-nonabstention; prompt-injection; leakage; quantization; stale; teacher; authority-completion-shaped
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/adversarial.py, docs/architecture/residual_intelligence_inventory/adversarial_campaign_report.json, test/api/residual_intelligence/test_adversarial.py
- **Predicted symbols:** ResidualAdversarialAdapter; ResidualMutantCampaign; CriticalMutantResult
- **Data rights:** Mutants retain parent rights/privacy and enter future training only under an admitted TrainingCorpusAdmission with exact mutant family and adversarial partition.
- **Privacy class:** repository_private
- **Effect class:** isolated_adversarial_qualification
- **Risk class:** R5
- **Resource class:** cpu-gpu-prover-isolated
- **Token budget:** input=48000; output=14000
- **Training budget:** 0; campaign outputs cannot train without TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_adversarial.py
- **Proof requirements:** Existing assurance engine receipts, exact final tree/model/group/quantization/environment roots, all critical mutants enumerated, zero escaped critical mutant, and no aggregate compensation.
- **Rollback:** Stop/fence campaign, discard isolated mutants/worktrees, retain failure receipts, and demote/revoke any implicated expert.
- **Conflict policy:** Own residual assurance adapter/report only; Adversarial Assurance remains authoritative and qualification never mutates production routes.

## VRIF-029 Add control service, CLI, and MCP surfaces

- **Status:** todo
- **Goal:** VRIF-G041
- **Depends on:** VRIF-013, VRIF-021, VRIF-022, VRIF-023, VRIF-026, VRIF-027
- **Objective:** Extend the canonical typed SupervisorControlService with expert reads/mutations and expose `ipfs-accelerate agent experts` plus canonical MCP operations that call the service directly.
- **Acceptance subset:** read-operation-catalog; mutation-operation-catalog; python-cli-mcp-parity; no-mcp-shell; authorization; idempotency; lease-fence; dry-run-budget-audit
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/cli.py, ipfs_accelerate_py/agent_supervisor/control/control_plane.py, test/api/residual_intelligence/test_control_surface.py
- **Predicted symbols:** ResidualExpertControlBackend; ExpertControlRequest; ExpertControlResult; register_expert_operations
- **Data rights:** Corpus/epoch APIs return identities and bounded summaries; raw private bodies require explicit authorized scope and TrainingCorpusAdmission does not grant control mutation authority.
- **Privacy class:** repository_private
- **Effect class:** authorized_fenced_control_mutation
- **Risk class:** R5
- **Resource class:** cpu-small-control
- **Token budget:** input=44000; output=12000
- **Training budget:** 0; start_training must return training_unavailable without an admitted TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_control_surface.py
- **Proof requirements:** Exact operation catalog, service/CLI/MCP parity, MCP direct call, mutation authorization/idempotency/lease/fence/dry-run/budget/audit, and restart recovery.
- **Rollback:** Deregister residual operations or restore prior catalog revision; fence active mutations and preserve audit receipts.
- **Conflict policy:** Coordinate shared control_plane.py as an explicit serialized integration; CLI/MCP adapters cannot bypass or duplicate the typed service.

## VRIF-030 Build frozen paired benchmark

- **Status:** todo
- **Goal:** VRIF-G041
- **Depends on:** VRIF-005, VRIF-006, VRIF-008, VRIF-014, VRIF-015, VRIF-019, VRIF-020
- **Objective:** Freeze training/development/held-out/adversarial/boundary/negative/cross-repository cases for every residual family with repository/objective/catalog/provider/tokenizer/model/fault/validation identities.
- **Acceptance subset:** all-24-families; unknown-ood; all-partitions; group-lineage; hidden-test-denial; cross-repository; frozen-roots; paired-baseline
- **Predicted files:** benchmarks/agent_supervisor/residual_intelligence/manifest.json, benchmarks/agent_supervisor/residual_intelligence/cases.jsonl, test/api/residual_intelligence/test_benchmark.py
- **Predicted symbols:** ResidualBenchmarkManifest; FrozenBenchmarkCase; PairedBenchmarkRunner
- **Data rights:** Every case and partition is covered by an admitted TrainingCorpusAdmission for its permitted use; held-out/adversarial cases remain inaccessible to training.
- **Privacy class:** repository_private
- **Effect class:** frozen_offline_benchmark
- **Risk class:** R4
- **Resource class:** cpu-gpu-bounded-evaluation
- **Token budget:** input=46000; output=13000
- **Training budget:** 0; benchmark evaluation cannot create or amend TrainingCorpusAdmission
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_benchmark.py
- **Proof requirements:** Family/partition coverage, exact frozen roots, semantic-lineage separation, hidden-test protection, no private leakage, and paired prior/current evaluation with complete denominators.
- **Rollback:** Revoke benchmark root/version and dependent evaluations; retain prior frozen benchmark and do not edit published cases in place.
- **Conflict policy:** Exclusive benchmark manifest/case owner; large datasets/evaluations stay in managed content-addressed storage outside Git.

## VRIF-031 Implement promotion and rollback gates

- **Status:** todo
- **Goal:** VRIF-G041
- **Depends on:** VRIF-024, VRIF-026, VRIF-027, VRIF-028, VRIF-030
- **Objective:** Enforce conjunctive rights/lineage/leakage/privacy/safety/quality/efficiency/autonomy/amortization gates, authorized CAS promotion, exact rollback, and proposal-only R4/R5 behavior.
- **Acceptance subset:** noncompensable-data-safety; 99-percent-precision; critical-abstention; 45-35-60-50-30-efficiency; 70-40-25-autonomy; amortization; cas-rollback
- **Predicted files:** ipfs_accelerate_py/agent_supervisor/residual_intelligence/promotion.py, test/api/residual_intelligence/test_promotion.py
- **Predicted symbols:** ExpertPromotionGate; PromotionEvidence; PromotionDecision; ExpertRollbackReceipt
- **Data rights:** Promotion requires 100% rights/lineage from the admitted TrainingCorpusAdmission and zero leakage/privacy violations; checkpoint or model confidence grants no permission.
- **Privacy class:** repository_private
- **Effect class:** authorized_expert_route_promotion
- **Risk class:** R5
- **Resource class:** cpu-medium-control-proof
- **Token budget:** input=44000; output=12000
- **Training budget:** 0; promotion cannot create TrainingCorpusAdmission or trigger training
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_promotion.py
- **Proof requirements:** Every hard gate passes independently on current held-out/adversarial evidence; no critical false accept/mutant; exact cost denominators/break-even; authorized CAS and tested rollback.
- **Rollback:** Authorized CAS revokes the promoted identity, restores only a current prior route, drains fenced work, and records a complete rollback receipt.
- **Conflict policy:** Exclusive promotion state owner; models, checkpoints, task status, DuckLake, reports, and aggregate scores cannot mutate promotion.

## VRIF-032 Produce current-tree release and residual-gap report

- **Status:** todo
- **Goal:** VRIF-G041
- **Depends on:** VRIF-028, VRIF-029, VRIF-030, VRIF-031
- **Objective:** Requalify the actual final merged tree and publish machine/human reports covering exact lineage, data, experts, metrics, costs, proofs, drift, rollback, blockers, eligibility, and unsupported gaps.
- **Acceptance subset:** start-end-tree; files-symbols; corpus-rights-splits; architecture-tokenizer-checkpoint; expert-dispositions; before-after-denominators; costs-break-even; proof-validation; blockers-rollback
- **Predicted files:** docs/architecture/residual_intelligence_inventory/final_release_report.json, docs/architecture/residual_intelligence_inventory/final_release_report.md, test/api/residual_intelligence/test_release_report.py
- **Predicted symbols:** ResidualIntelligenceReleaseReport; ResidualGapReport; validate_release_claims
- **Data rights:** Public report contains CIDs and bounded summaries only; all corpus/source dispositions trace to an admitted TrainingCorpusAdmission or explicit training_unavailable outcome.
- **Privacy class:** public
- **Effect class:** nonauthoritative_current_tree_reporting
- **Risk class:** R5
- **Resource class:** cpu-medium-local-proof
- **Token budget:** input=52000; output=16000
- **Training budget:** 0; reporting cannot create TrainingCorpusAdmission, checkpoint, expert, promotion, proof, or completion
- **Validation:** python3 -m pytest -q test/api/residual_intelligence/test_release_report.py && python3 scripts/validate_agent_supervisor_residual_intelligence_board.py --check-all
- **Proof requirements:** Exact final commit/tree, declared producer receipts, complete denominators/costs/not-run results, promotion eligibility and rollback target, no unsupported learned/verified/safe/autonomous/token-efficient/production-ready claim.
- **Rollback:** Revoke report root and regenerate from the corrected exact tree/evidence; never rewrite evidence or promote from a report.
- **Conflict policy:** Exclusive final report joiner; it reads accepted receipts and DuckLake history but cannot alter source, task, goal, proof, expert, threshold, checkpoint, or release authority.
