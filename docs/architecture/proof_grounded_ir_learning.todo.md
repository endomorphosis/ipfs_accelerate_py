# Proof-Grounded IR Learning Fabric Dependency Board

This is the canonical executable board for `ProofGroundedIRLearningFabric`.
It is ordered by dependencies, not prose order. Workers must use the exact
checked-out-authority snapshots below. The reviewed revisions are comparison
inputs only because each reviewed revision is a descendant of its checkout.

## Frozen identities and policy aliases

- `SRC-DATASETS-AUTH-1`: repository `endomorphosis/ipfs_datasets_py`; checkout HEAD `d144be65ffe4c6423e4e1c30cd692812607343eb`; live authority commit `df93e91e6338c84a17c3208ef68b88de8566f78c`; tree `37b9cb40644831c85c6fdf07d0228e45061e239a`; live-delta SHA-256 `2f93a232612d1b8d1da6b52abfa1639621a86ac82eef2180f163eaa9d6b547f4`; reviewed comparison revision `7f0fe2bbad3c70928234c6e2312ee3182fd7681f` is 1,717 commits ahead; complete range `d144be65ffe4c6423e4e1c30cd692812607343eb..7f0fe2bbad3c70928234c6e2312ee3182fd7681f`; range-log SHA-256 `aaeff6d8976787159e8ec747fc60a5d27b6515773068c06e968cfb3a107dd21e`.
- `SRC-ACCEL-AUTH-1`: repository `endomorphosis/ipfs_accelerate_py`; checkout HEAD `0cc04ebb640c4c981cf4650016e096a73ab0e8c0`; live authority commit `8d46a6d25dd006c8cab3c9d9612707d2a014e79c`; tree `697ee660025fbf14a1cbe6c24fd8da5365df84d5`; live-delta SHA-256 `0d13706bbdd5f50118999dc928172c8f0df29aea8f86613b0f5664e60435c87c`; reviewed comparison revision `c821d0b43877591bbb0fa3f328fbccff187b56e7` is 3,616 commits ahead; complete range `0cc04ebb640c4c981cf4650016e096a73ab0e8c0..c821d0b43877591bbb0fa3f328fbccff187b56e7`; range-log SHA-256 `0a70de8c18be990e59660a0a4cbaf00cf81cf31b3321ad9b03bab0a666eaf61e`.
- `SRCSET-1`: ordered pair (`SRC-DATASETS-AUTH-1`, `SRC-ACCEL-AUTH-1`). No worker may substitute the reviewed trees without a new admitted plan revision.
- `JDAO-PINSET-1`: `justice_dao_pinset.yaml`, SHA-256 `8e3a4b1bd81639393ddda35e5dfb3b95f9e7320afa898bde0b3eb9a0317a6b76`, containing all 21 exact Hub revisions. It admits zero repositories for proof-grounded training at launch.
- `COMPILER-CURRENT-1`: `SRC-DATASETS-AUTH-1` plus `ipfs_datasets_py/logic/legal_ir/canonical_compiler.py` and the registered typed family adapters.
- `DECOMPILER-CURRENT-1`: `SRC-DATASETS-AUTH-1` plus `ipfs_datasets_py/logic/legal_ir/canonical_decompiler.py` and controlled family decoders.
- `MODEL-LEGACY-1`: `justicedao/legal-ir-autoencoder-checkpoints@94ca549d102e3e31781370aec1247f91365440eb`, checkpoint `legal-ir-autoencoder-canonical-20260630T221836Z`, state SHA-256 `7236de26bd3d7f8414ffa04805f1b6e8a8849f9e0103cec6edb4985b911658be`; artifact-only and never promotion authority.
- `RESULT(task)`: CID of canonical `pgir-task-result@1` over task ID/revision, exact input identities, lease fence, effects, output CIDs, test/proof receipts, disposition, and recovery record.
- `LEASE-DEFAULT`: one renewable 30-minute lease, heartbeat at most 60 seconds, monotonically increasing fence, maximum three attempts. Checkpoint, tokenizer, corpus, split, compiler-contract, loss-config, proof shard, evaluation shard, promotion pointer, and publication mutations use distinct exclusive keys. Duplicate attempts may compute; compare-and-swap admits one `RESULT(task)` per input root.
- `ROLLBACK-DEFAULT`: stop descendants; retain immutable attempt/evidence records; revert only the task commit or unaccepted pointer; restore the previous content-addressed root by compare-and-swap; never delete source releases, hidden tests, verified evidence, or another task's worktree.

Resource profiles are admission ceilings:

- `RP-CPU-S`: 2 CPU, 4 GiB RAM, 4 GiB disk, no GPU/provider/prover/network, 30 minutes.
- `RP-CPU-M`: 4 CPU, 12 GiB RAM, 20 GiB disk, no GPU, optional local parser, 90 minutes.
- `RP-IO-PINNED`: 4 CPU, 12 GiB RAM, 100 GiB disk, allowlisted HTTPS only to exact pinned Hub revisions, 4 hours.
- `RP-PROVER`: 6 CPU, 16 GiB RAM, 40 GiB disk, no GPU, admitted local prover portfolio, 2 hours and 60 seconds per obligation.
- `RP-GPU`: 8 CPU, 32 GiB RAM, one leased GPU with at least 16 GiB VRAM, 250 GiB disk, 8 hours, bounded provider tokens; deny when telemetry is missing.
- `RP-MIXED`: 8 CPU, 32 GiB RAM, optional one leased GPU, 250 GiB disk, allowlisted providers/provers/network, 8 hours with stage-specific subleases.

The freeze chain is `PGIR-001..014`. No learned-model, pair-mining, proof-
curriculum, training, or publication task may be leased before `PGIR-014`
materializes exact schema, source, split, compiler, tokenizer-policy, and
authority roots into a revised task input binding.

## PGIR-001 Record source history and live authority

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: inventory
- Parent goal: PGIR-G010
- Subgoal: revision-reconciliation
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: none; no dataset rows consumed
- Data split identity: not-applicable/inventory
- Compiler identity: `COMPILER-CURRENT-1` (observation only)
- Decompiler identity: `DECOMPILER-CURRENT-1` (observation only)
- Model checkpoint identity: `MODEL-LEGACY-1` (classification only)
- Objective: Emit complete ordered commit manifests, tree/delta manifests, submodule residuals, dirty-path inventories, and a reconciliation no-go/allow decision without changing either authority snapshot.
- Depends on: none
- Resource profile: `RP-CPU-S`
- Expected inputs: both Git object databases and the frozen identities above
- Expected outputs: `IRSourceCodeRevisionManifest@1` and full machine-readable commit/path manifests
- Allowed effects: write only owned evidence paths
- Prohibited effects: checkout/reset/rebase/pull, source edits, auto-commit of user changes, baseline substitution
- Acceptance criteria: all 1,717 and 3,616 commits are enumerated; directions/counts/digests reproduce; every residual dirty submodule is explicit
- Required proof or evaluation evidence: deterministic replay receipt plus `git fsck`/range-count receipt
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `source-revision-manifest`; checkpoint after each repository
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-001)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json
- Validation: `python -m pytest test/api/test_agent_supervisor_repository_forest_manifest.py -q`
- Bundle: pgir/freeze/revisions
- Parallel lane: revision-audit
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json
- Conflict policy: serial owner of source-root identities

## PGIR-002 Inventory canonical semantic implementations

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: inventory
- Parent goal: PGIR-G010
- Subgoal: datasets-module-classification
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/inventory/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `JDAO-PINSET-1` metadata only
- Data split identity: not-applicable/inventory
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: `MODEL-LEGACY-1`
- Objective: Classify every requested IR, parser, formalization, bridge, compiler/decompiler, canonicalization, source-map, proof, counterexample, trace, provider, embedding, vector, loss, checkpoint, and publication implementation using actual behavior and paths.
- Depends on: PGIR-001
- Resource profile: `RP-CPU-M`
- Expected inputs: `SRC-DATASETS-AUTH-1`, registry declarations, imports, tests
- Expected outputs: canonical/component/generated/facade/legacy/experimental/artifact/declaration/duplicate/obsolete/unresolved inventory
- Allowed effects: documentation/evidence under owned paths
- Prohibited effects: inventing literal modules, code migration, new logic/semantic/proof/cache system
- Acceptance criteria: all A1 roles have one disposition and evidence path; parallel compiler/checkpoint meanings and unresolved roles are explicit
- Required proof or evaluation evidence: import/static reference scan and focused collection receipt
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `datasets-inventory`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-002)`
- Outputs: ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/inventory/modules.json
- Validation: `python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_ir_family_conformance.py ipfs_datasets_py/tests/unit/logic/formalization`
- Bundle: pgir/freeze/datasets-inventory
- Parallel lane: datasets-inventory
- Predicted files: ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/inventory/modules.json
- Conflict policy: read-only source scan; one inventory writer

## PGIR-003 Inventory canonical supervisor learning infrastructure

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: inventory
- Parent goal: PGIR-G010
- Subgoal: accelerator-module-classification
- Owning repository: ipfs_accelerate_py
- Owned paths: docs/architecture/proof_grounded_ir_learning/inventory/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `JDAO-PINSET-1` metadata only
- Data split identity: not-applicable/inventory
- Compiler identity: `COMPILER-CURRENT-1` adapter boundary only
- Decompiler identity: `DECOMPILER-CURRENT-1` adapter boundary only
- Model checkpoint identity: `MODEL-LEGACY-1` classification only
- Objective: Classify actual planning, provider, Lean proposal, tactician, hammer, prover, checker, resource, multi-supervisor, daemon, checkpoint, refill, promotion, experiment, and publication infrastructure.
- Depends on: PGIR-001
- Resource profile: `RP-CPU-M`
- Expected inputs: `SRC-ACCEL-AUTH-1`, package map, CLI schemas, focused tests
- Expected outputs: A2 inventory and reuse/gap decisions
- Allowed effects: documentation/evidence under owned paths
- Prohibited effects: adding a scheduler/supervisor/prover/cache; calling merge-train ML training
- Acceptance criteria: canonical provider/proof/checker trust boundaries and absent training/promotion surfaces are explicit
- Required proof or evaluation evidence: capability discovery and cold-import side-effect receipt
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `accelerator-inventory`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-003)`
- Outputs: docs/architecture/proof_grounded_ir_learning/inventory/supervisor.json
- Validation: `python -m pytest -q test/api/test_agent_supervisor_formal_verification_capabilities.py test/api/test_agent_supervisor_resource_scheduler.py test/api/test_agent_supervisor_implementation_supervisor_runner.py`
- Bundle: pgir/freeze/accelerator-inventory
- Parallel lane: accelerator-inventory
- Predicted files: docs/architecture/proof_grounded_ir_learning/inventory/supervisor.json
- Conflict policy: read-only source scan; one inventory writer

## PGIR-004 Pin and classify JusticeDAO releases

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G010
- Subgoal: hf-release-inventory
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/source_inventory/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `JDAO-PINSET-1` exact 21-repository population
- Data split identity: Hub-provided `train` only; explicitly not an admissible campaign split
- Compiler identity: not-applicable/release-inventory
- Decompiler identity: not-applicable/release-inventory
- Model checkpoint identity: `MODEL-LEGACY-1`
- Objective: Revalidate exact revisions/configs/splits/files/shards/schemas/counts/rights/cutoffs/jurisdictions/tool versions/gaps/Viewer status and separate source from derivative rows.
- Depends on: PGIR-001
- Resource profile: `RP-IO-PINNED`
- Expected inputs: allowlisted HF APIs and pinned files only
- Expected outputs: `IRSourceReleaseInventory@1`, rights/admission/quarantine receipts, source/derived count table
- Allowed effects: bounded downloads and owned immutable inventory artifacts
- Prohibited effects: trust-remote-code, arbitrary scripts, bulk unbounded download, silent repair, publication
- Acceptance criteria: all 21 pins revalidate; 2,174 patent source rows and 4,999 Dutch law source rows are not inflated by derivatives; broken/mismatched/unresolved releases fail closed
- Required proof or evaluation evidence: file/hash/Viewer/load receipts and rights decision per config
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `hf-pinset`; checkpoint per repository
- Rollback procedure: `ROLLBACK-DEFAULT`; quarantine partial downloads
- Result identity: `RESULT(PGIR-004)`
- Outputs: ipfs_datasets_py/data/ir_learning/source_inventory/release_inventory.json
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py`
- Bundle: pgir/freeze/hf-inventory
- Parallel lane: hf-inventory
- Predicted files: ipfs_datasets_py/data/ir_learning/source_inventory/release_inventory.json
- Conflict policy: exclusive release-inventory root; network allowlist is immutable

## PGIR-005 Seal focused pre-change test evidence and repair prerequisites

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G010
- Subgoal: baseline-test-receipts
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/baseline_tests/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: none; synthetic/current fixtures only
- Data split identity: fixture populations named in each receipt
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: `MODEL-LEGACY-1`
- Objective: Seal the actual 360/360 accelerator pass and 801 pass, 2 fail, 2 skip, 13 error dataset result; create bounded prerequisite work for semantic CID drift, missing Lean runtime export, and absent hammer environment lock.
- Depends on: PGIR-001, PGIR-002, PGIR-003
- Resource profile: `RP-PROVER`
- Expected inputs: exact commands, logs, environment/tool capability snapshot
- Expected outputs: typed baseline test receipts and prerequisite/no-go dispositions
- Allowed effects: owned evidence and generated repair tasks only
- Prohibited effects: rewriting snapshots, treating unavailable prover as pass, hiding failures
- Acceptance criteria: every selected category has a result; first failed assertion and setup cause are retained; repair tasks cannot weaken gates
- Required proof or evaluation evidence: pytest result CIDs and capability probes
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `baseline-tests`; one sublease per test group
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-005)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/baseline_tests/summary.json
- Validation: `python -m pytest -q test/api/test_agent_supervisor_proof_workflow_e2e.py test/api/test_agent_supervisor_autonomous_unstall.py`
- Bundle: pgir/freeze/baseline-tests
- Parallel lane: baseline-tests
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/baseline_tests/summary.json
- Conflict policy: evidence-only; repairs receive separate task revisions

## PGIR-006 Build the compiler/decompiler gap matrix

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: inventory
- Parent goal: PGIR-G010
- Subgoal: direction-gap-matrix
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/gap_matrix.*
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `JDAO-PINSET-1`
- Data split identity: not-applicable; availability only
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: `MODEL-LEGACY-1`
- Objective: For each requested direction record schemas, deterministic/learned implementations, examples, proof/counterexample/round-trip coverage, token/cosine/syntax/type/proof metrics, limitations, and remediation.
- Depends on: PGIR-002, PGIR-003, PGIR-004, PGIR-005
- Resource profile: `RP-CPU-M`
- Expected inputs: all inventory and baseline receipts
- Expected outputs: machine-readable A4 matrix and human summary
- Allowed effects: owned documentation only
- Prohibited effects: invented metrics, implicit zero, universal semantic claims
- Acceptance criteria: every source/typed/bridge/family/prover/CNL/trace direction has an explicit row and unknowns remain unknown
- Required proof or evaluation evidence: schema/path/receipt links for every non-unknown cell
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `gap-matrix`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-006)`
- Outputs: ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/gap_matrix.json
- Validation: `python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py`
- Bundle: pgir/freeze/gap-matrix
- Parallel lane: gap-matrix
- Predicted files: ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/gap_matrix.json
- Conflict policy: begins after all inventories seal

## PGIR-010 Extend canonical source and lineage contracts

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: semantic-contracts
- Parent goal: PGIR-G020
- Subgoal: source-lineage-schema
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/**, ipfs_datasets_py/tests/unit/logic/ir_core/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `JDAO-PINSET-1` schema fixtures only
- Data split identity: not-yet-created
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: none
- Objective: Extend existing `ir_core` identity/provenance/artifact registries with equivalent versioned source release, source record, derived artifact, lineage graph, and corpus manifest records.
- Depends on: PGIR-006
- Resource profile: `RP-CPU-M`
- Expected inputs: inventory, `SourceRef`, CID/canonical schemas, pinset
- Expected outputs: closed schemas/migrations/validators with all required source, temporal, rights, and lineage fields
- Allowed effects: canonical registry/contracts/tests only
- Prohibited effects: new canonicalizer, logic family, separate provenance system, float-valued durable identity
- Acceptance criteria: deterministic CIDs, strict unknown-field policy, source/derived kinds, rights validation, round-trip/migration/property tests
- Required proof or evaluation evidence: golden vectors, malformed/adversarial fixtures, schema registry receipt
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `ir-source-contract`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-010)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/, ipfs_datasets_py/tests/unit/logic/ir_core/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/ir_core`
- Bundle: pgir/freeze/source-contracts
- Parallel lane: source-contracts
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/, ipfs_datasets_py/tests/unit/logic/ir_core/
- Conflict policy: exclusive `ir_core` schema-registry lease

## PGIR-011 Build a safe pinned corpus and source/derivative manifest

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G020
- Subgoal: corpus-build
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/huggingface/, ipfs_datasets_py/data/ir_learning/corpora/, ipfs_datasets_py/tests/unit/logic/ir_learning/source/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: exact admitted/quarantined config revisions from `RESULT(PGIR-004)`; initial `JDAO-PINSET-1`
- Data split identity: not-yet-created; sealed corpus groups only
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: none
- Objective: Implement safe pinned ingestion and content-addressed corpus manifests, preserving one source lineage group across articles, vectors, BM25, graph, translations, proofs, and repeated states.
- Depends on: PGIR-004, PGIR-010
- Resource profile: `RP-IO-PINNED`
- Expected inputs: exact release inventory and source contracts
- Expected outputs: sealed corpus root, source/derived counts, rights manifest, quarantine manifest
- Allowed effects: bounded cache/download and immutable corpus artifacts
- Prohibited effects: remote code, broken release use, random split, counting derivatives as sources
- Acceptance criteria: deterministic replay; 2,174 patent documents remain 2,174 groups; rights-quarantined rows cannot enter training; malformed/oversized/path attacks fail
- Required proof or evaluation evidence: build replay, file hashes/CIDs, rights and count reconciliation receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `corpus-build`; shard checkpoints immutable
- Rollback procedure: `ROLLBACK-DEFAULT`; quarantine partial shard
- Result identity: `RESULT(PGIR-011)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/huggingface/, ipfs_datasets_py/data/ir_learning/corpora/, ipfs_datasets_py/tests/unit/logic/ir_learning/source/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/ir_learning/source`
- Bundle: pgir/freeze/corpus
- Parallel lane: corpus-builder
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/huggingface/, ipfs_datasets_py/data/ir_learning/corpora/, ipfs_datasets_py/tests/unit/logic/ir_learning/source/
- Conflict policy: one corpus root writer; shard workers produce immutable candidates

## PGIR-012 Implement lineage-safe multidimensional splits

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: semantic-contracts
- Parent goal: PGIR-G020
- Subgoal: split-and-leakage
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_eval_splits.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py, ipfs_datasets_py/data/ir_learning/splits/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: sealed corpus `RESULT(PGIR-011)`
- Data split identity: output `IRSplitManifest@1`
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: none
- Objective: Extend the existing split guard to group all source/derivative/theorem/notation/paraphrase siblings and create declared lineage, publication, jurisdiction, domain, family, notation, time, type, compiler, proof-library, premise, length, rare-operator, exception, and cross-reference holdouts.
- Depends on: PGIR-011
- Resource profile: `RP-CPU-M`
- Expected inputs: sealed corpus/lineage graph and duplicate detectors
- Expected outputs: deterministic split manifest, leakage report, frozen hidden-test commitment
- Allowed effects: owned split artifacts and tests
- Prohibited effects: random-row principal split, hidden-label exposure, post-freeze membership mutation
- Acceptance criteria: zero known cross-split lineage leakage; duplicate/near-duplicate and derivative audits pass; every holdout has counts and insufficiency status
- Required proof or evaluation evidence: adversarial seeded leakage fixtures and content-root replay
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `split-manifest`
- Rollback procedure: `ROLLBACK-DEFAULT`; revoke entire manifest root, never patch membership in place
- Result identity: `RESULT(PGIR-012)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_eval_splits.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py, ipfs_datasets_py/data/ir_learning/splits/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py`
- Bundle: pgir/freeze/splits
- Parallel lane: split-auditor
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_eval_splits.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py, ipfs_datasets_py/data/ir_learning/splits/
- Conflict policy: exclusive split authority and fencing token

## PGIR-013 Implement training-example and trace contracts

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: semantic-contracts
- Parent goal: PGIR-G030
- Subgoal: examples-label-authority
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/**, ipfs_datasets_py/tests/unit/logic/formalization/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: sealed corpus `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: none
- Objective: Add closed versioned example/compiler/decompiler/translation/proof/tactic/hard-negative/positive/round-trip records and label/evidence authority without replacing existing formalization contracts.
- Depends on: PGIR-010, PGIR-012
- Resource profile: `RP-CPU-M`
- Expected inputs: source/lineage/split roots and existing formalization/constraint receipts
- Expected outputs: strict schemas, admissibility/quarantine reasons, cross-statement proof binding checks
- Allowed effects: existing formalization contract namespace and tests
- Prohibited effects: authority increase by translation, model output as truth, proof attached to other statement, unresolved loss as exact
- Acceptance criteria: all required fields bind; closed authority vocabulary; every rejection class tested; unknown family/relationship fails closed
- Required proof or evaluation evidence: golden CIDs, property tests, malicious examples, proof-binding fixtures
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `training-example-contract`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-013)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/, ipfs_datasets_py/tests/unit/logic/formalization/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/formalization`
- Bundle: pgir/freeze/example-contracts
- Parallel lane: example-contracts
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/, ipfs_datasets_py/tests/unit/logic/formalization/
- Conflict policy: begins after split freeze; exclusive formalization schema lease

## PGIR-014 Freeze semantic campaign inputs

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G030
- Subgoal: shared-freeze-root
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/freeze/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: exact `RESULT(PGIR-011)`
- Data split identity: exact `RESULT(PGIR-012)`
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: none
- Objective: Validate and bind schema registry, corpus, rights, lineage, split, example contracts, current compiler/decompiler, source snapshots, and policy into one immutable campaign freeze root; revise all descendant tasks with exact output identities.
- Depends on: PGIR-006, PGIR-010, PGIR-011, PGIR-012, PGIR-013
- Resource profile: `RP-CPU-S`
- Expected inputs: all freeze-chain results
- Expected outputs: `IRCampaignInputRoot@1`, revised descendant task CIDs, no-go if any identity remains unresolved
- Allowed effects: owned freeze artifacts and task-plan revision proposal
- Prohibited effects: source/schema/split mutation, promotion, hidden test access
- Acceptance criteria: full referential integrity and reproducible CIDs; zero training task eligible on unresolved rights/leakage/compiler identity
- Required proof or evaluation evidence: independent manifest verifier and supervisor plan-admission receipt
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `campaign-freeze-root`
- Rollback procedure: `ROLLBACK-DEFAULT`; supersede, never overwrite, a freeze root
- Result identity: `RESULT(PGIR-014)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/freeze/
- Validation: `python -m pytest -q test/api/test_agent_supervisor_formal_plan_validator.py test/api/test_agent_supervisor_task_identity.py`
- Bundle: pgir/freeze/root
- Parallel lane: freeze-root
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/freeze/
- Conflict policy: global serial freeze barrier; no downstream fan-out before admission

## PGIR-020 Consolidate the canonical typed bridge

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: compiler
- Parent goal: PGIR-G040
- Subgoal: bridge-ir
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_contracts.py, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` fixtures only
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `COMPILER-CURRENT-1`
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: none
- Objective: Compose one rich bridge from existing typed contracts and adapter registry, explicitly carrying unsupported/family extensions, assumptions, source references, provenance, and all required constructs.
- Depends on: PGIR-014
- Resource profile: `RP-CPU-M`
- Expected inputs: gap matrix, existing bridge/legal/family contracts
- Expected outputs: canonical bridge schema/registry/migrations and adapter conformance
- Allowed effects: existing bridge/canonical contract paths
- Prohibited effects: new logic family, duplicate canonicalizer, collapsing family identity
- Acceptance criteria: every requested construct is represented or explicitly unsupported; deterministic canonical identity; family round-trip fixtures pass
- Required proof or evaluation evidence: schema golden vectors and family conformance receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `canonical-bridge-contract`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-020)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_contracts.py, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_ir_family_conformance.py ipfs_datasets_py/tests/unit/logic/legal_ir`
- Bundle: pgir/bridge/contracts
- Parallel lane: bridge-contract
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_contracts.py, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Conflict policy: exclusive bridge schema lease

## PGIR-021 Consolidate deterministic compiler pipeline

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: compiler
- Parent goal: PGIR-G040
- Subgoal: compiler-pipeline
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/integration/reasoning/, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` development partition only
- Compiler identity: output compiler CID from this task
- Decompiler identity: `DECOMPILER-CURRENT-1`
- Model checkpoint identity: none/deterministic
- Objective: Reproduce source selection/spans, typed family parse, elaboration, formalization, available domain slice adapter, bridge, and target stages with explicit identities and unsupported constructs.
- Depends on: PGIR-020
- Resource profile: `RP-CPU-M`
- Expected inputs: frozen bridge/contracts and current compiler surfaces
- Expected outputs: one canonical compiler API/pipeline and compatibility adapters
- Allowed effects: existing canonical compiler/integration adapter paths
- Prohibited effects: LLM baseline, stale trace reuse, deletion of compatibility without migration
- Acceptance criteria: deterministic syntax/type/source-map outputs; parallel compiler surfaces delegate to one authority; every stage replayable
- Required proof or evaluation evidence: golden compile traces, source-map and invalid-input tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `compiler-contract`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-021)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/integration/reasoning/, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/legal_ir ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py`
- Bundle: pgir/bridge/compiler
- Parallel lane: compiler
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/integration/reasoning/, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Conflict policy: exclusive compiler contract/tokenization inputs

## PGIR-022 Consolidate decompiler and translation preservation

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: decompiler
- Parent goal: PGIR-G040
- Subgoal: decompiler-translation
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_decompiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_roundtrip.py, ipfs_datasets_py/ipfs_datasets_py/logic/bridge/, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` development partition only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: output decompiler CID from this task
- Model checkpoint identity: none/deterministic
- Objective: Separate controlled semantic reconstruction from style paraphrase and enforce closed lossless/equisatisfiable/over-/under-approximation/heuristic/unsupported translation contracts without authority increase.
- Depends on: PGIR-020, PGIR-021
- Resource profile: `RP-CPU-M`
- Expected inputs: bridge and compiler identities, current decoders/receipts
- Expected outputs: canonical decompiler, translation receipt, semantic recompilation gate
- Allowed effects: existing bridge/decompiler/round-trip paths
- Prohibited effects: plausible prose as fidelity, heuristic translation as proof, undeclared loss
- Acceptance criteria: all required round-trip directions and equality criteria recorded; paraphrase fidelity requires recompilation/semantic check
- Required proof or evaluation evidence: semantic-difference, translation-class, and adversarial decompilation fixtures
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `decompiler-contract`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-022)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_decompiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_roundtrip.py, ipfs_datasets_py/ipfs_datasets_py/logic/bridge/, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/legal_ir ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py`
- Bundle: pgir/bridge/decompiler
- Parallel lane: decompiler
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_decompiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_roundtrip.py, ipfs_datasets_py/ipfs_datasets_py/logic/bridge/, ipfs_datasets_py/tests/unit/logic/legal_ir/
- Conflict policy: begins after compiler identity; shared bridge changes serialized

## PGIR-023 Measure deterministic compiler/decompiler baseline

- Status: completed
- Completion: evaluation-evidence
- Is schedulable: true
- Priority: P0
- Track: evaluation
- Parent goal: PGIR-G040
- Subgoal: deterministic-baseline
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/benchmarks/semantic_roundtrip/**, ipfs_datasets_py/data/ir_learning/evaluations/deterministic/**
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` frozen non-hidden partitions
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none/deterministic
- Objective: Measure parser/type acceptance, exact/canonical/AST/graph/source-span/semantic/proof/unsupported/latency metrics separately for compiler and decompiler.
- Depends on: PGIR-022
- Resource profile: `RP-PROVER`
- Expected inputs: frozen examples/splits and deterministic pipelines
- Expected outputs: content-addressed R1 baseline and per-family/domain strata
- Allowed effects: benchmark/evaluation artifacts only
- Prohibited effects: learned inference, hidden-test selection, missing metric as zero
- Acceptance criteria: all E1 metrics reported with denominators, unsupported/unknown strata, tool versions and resource use
- Required proof or evaluation evidence: paired trace CIDs and independent proof replay where available
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `evaluation:deterministic`; immutable shards
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-023)`
- Outputs: ipfs_datasets_py/benchmarks/semantic_roundtrip/, ipfs_datasets_py/data/ir_learning/evaluations/deterministic/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py`
- Bundle: pgir/baseline/deterministic
- Parallel lane: deterministic-eval
- Predicted files: ipfs_datasets_py/benchmarks/semantic_roundtrip/, ipfs_datasets_py/data/ir_learning/evaluations/deterministic/
- Conflict policy: evaluation shards read frozen inputs; one report reducer lease

## PGIR-030 Implement compatible learned architectures and frozen tokenizer

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: model
- Parent goal: PGIR-G050
- Subgoal: architecture-tokenizer
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_autoencoder.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` train/development only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: output architecture/tokenizer initialization root; `MODEL-LEGACY-1` warm-start only if compatibility and quarantine gates pass
- Objective: Extend the existing advisor into the smallest compatible shared-latent and shared-encoder/typed-head experiment arms; freeze canonical tokenizer/vocabulary and structured output heads.
- Depends on: PGIR-023
- Resource profile: `RP-GPU`
- Expected inputs: canonical traces, deterministic baseline, legacy artifact classification
- Expected outputs: architecture manifests, tokenizer/vocabulary CID, initialization checkpoints, parameter/resource estimates
- Allowed effects: existing optimizer/grammar-decoder implementation and tests
- Prohibited effects: new semantic authority, unfrozen vocabulary mutation, legacy checkpoint promotion, architecture winner assumption
- Acceptance criteria: both arms runnable; output heads/conditioning/uncertainty explicit; tokenizer canonicalization/source surface separation; import side-effect free
- Required proof or evaluation evidence: shape/gradient/serialization/token-class/golden tokenizer tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive `tokenizer` and per-arm checkpoint keys
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-030)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_autoencoder.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_modal_autoencoder.py`
- Bundle: pgir/model/architecture
- Parallel lane: model-architecture
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_autoencoder.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: serial tokenizer/vocabulary freeze; arm checkpoints separate

## PGIR-031 Add grammar, binder, type, and proof-state constrained decoding

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: model
- Parent goal: PGIR-G050
- Subgoal: constrained-decoding
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` train/development only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: `RESULT(PGIR-030)` initialization roots
- Objective: Add valid-token/grammar/binder/type/family masks, bounded beam search, parser pruning, and optional proof-state pruning so cheap invalid candidates do not consume proof budgets.
- Depends on: PGIR-030
- Resource profile: `RP-CPU-M`
- Expected inputs: tokenizer/vocabulary, parsers/type systems, model heads
- Expected outputs: constrained decoder contracts and rejection telemetry
- Allowed effects: existing grammar decoder and model tests
- Prohibited effects: parser/type bypass, unbounded beams, family operator leakage
- Acceptance criteria: invalid candidates fail before prover calls; constraints preserve valid gold paths; bounds and fallback explicit
- Required proof or evaluation evidence: mutation/property tests and proof-call count comparison
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `decoder-constraints`; no checkpoint weight mutation
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-031)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer`
- Bundle: pgir/model/decoding
- Parallel lane: constrained-decoding
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: serialized with tokenizer; no concurrent vocabulary mutation

## PGIR-032 Implement versioned composite loss and sampling contracts

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: model
- Parent goal: PGIR-G050
- Subgoal: losses-and-samplers
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` train/development only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: `RESULT(PGIR-030)` initialization roots
- Objective: Implement fixed-point `IRLossConfiguration@1`, masked per-token-class CE, normalized cosine, supervised contrastive, cycle, structural/relation/semantic/proof/source-span/calibration/regularization signals, static baseline weights, and reproducible samplers.
- Depends on: PGIR-030, PGIR-031
- Resource profile: `RP-GPU`
- Expected inputs: frozen examples/tokenizer/architecture and proof-label authority
- Expected outputs: exact loss config, precision/schedule/sampler/memory-bank identities and component metrics
- Allowed effects: existing optimizer loss/training modules and tests
- Prohibited effects: proof calls in ordinary gradient path, floats in durable weights, aggregate CE hiding token classes, all-record cosine maximization
- Acceptance criteria: each loss isolatable; padding/binders/operators/types/source/family/proof/tactic tokens reported; teacher forcing/free run distinct; adaptive weights bounded and optional
- Required proof or evaluation evidence: analytical/golden loss tests, nonfinite/gradient policy tests, sampler false-negative fixtures
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive `loss-configuration`; memory bank bound to checkpoint
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-032)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer`
- Bundle: pgir/model/loss
- Parallel lane: loss-config
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: follows decoder; exclusive loss config root

## PGIR-033 Implement latent diagnostics and calibration instrumentation

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P1
- Track: evaluation
- Parent goal: PGIR-G050
- Subgoal: latent-diagnostics
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_semantic_metrics.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_family_evaluator.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_metric_lineage.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_uncertainty.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_evaluation_artifacts.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` development/calibration only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: `RESULT(PGIR-030)` fixture checkpoints
- Objective: Measure singular values, effective rank, variance/norms, anisotropy, family/domain/length/jurisdiction/duplicate clustering, latent use, false neighborhoods, ECE/Brier/reliability and success-conditioned confidence.
- Depends on: PGIR-030
- Resource profile: `RP-CPU-M`
- Expected inputs: frozen representation batches and metric lineage
- Expected outputs: content-bound diagnostic/calibration reports and collapse triggers
- Allowed effects: existing evaluation/uncertainty modules and tests
- Prohibited effects: latent similarity as equivalence, confidence as authority, hidden-test tuning
- Acceptance criteria: family-balanced and OOD strata; degeneracies/unknown denominators explicit; deterministic metric lineage
- Required proof or evaluation evidence: synthetic collapse/anisotropy/memorization fixtures and metric golden vectors
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `latent-diagnostics`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-033)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_semantic_metrics.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_family_evaluator.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_metric_lineage.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_uncertainty.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_evaluation_artifacts.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer`
- Bundle: pgir/model/diagnostics
- Parallel lane: latent-diagnostics
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_semantic_metrics.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_family_evaluator.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_metric_lineage.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_uncertainty.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_evaluation_artifacts.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: read-only checkpoint evaluation; disjoint from loss implementation

## PGIR-040 Mine typed proof-aware positive pairs

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: curriculum
- Parent goal: PGIR-G060
- Subgoal: positive-pairs
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/, ipfs_datasets_py/data/ir_learning/pairs/positive/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` train only; lineage groups indivisible
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none; deterministic mining
- Objective: Generate exact/alpha/canonical/logical/equisatisfiable/proof/translation/paraphrase classes from verified identities and translations with explicit authority.
- Depends on: PGIR-032, PGIR-020
- Resource profile: `RP-PROVER`
- Expected inputs: example contracts, bridge, proof corpus, split/corpus roots
- Expected outputs: `IRPositivePair@1` shards with equivalence class and evidence
- Allowed effects: immutable pair shards and miner/tests
- Prohibited effects: equisatisfiable/paraphrase as exact, cross-split siblings, model-only proof labels
- Acceptance criteria: every pair has complete lineage/authority; duplicates filtered; independent verification required for logical/proof classes
- Required proof or evaluation evidence: reconstruction/kernel/translation receipts and negative authority tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, key per pair shard; reducer CAS
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-040)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/, ipfs_datasets_py/data/ir_learning/pairs/positive/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer`
- Bundle: pgir/curriculum/positives
- Parallel lane: positive-miner
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/, ipfs_datasets_py/data/ir_learning/pairs/positive/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: shards parallel; no shared source/split mutation

## PGIR-041 Mine and validate hard semantic negatives

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: curriculum
- Parent goal: PGIR-G060
- Subgoal: hard-negatives
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_hard_negatives.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_fuzzing.py, ipfs_datasets_py/data/ir_learning/pairs/negative/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` train only; lineage groups indivisible
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none; deterministic/solver validation
- Objective: Implement all specified minimal mutation classes, parse/type check them, obtain counterexample/non-equivalence/satisfiability/entailment evidence where practical, and enforce false-negative protection.
- Depends on: PGIR-032, PGIR-040
- Resource profile: `RP-PROVER`
- Expected inputs: examples, bridge, positive-equivalence index, admitted prover portfolio
- Expected outputs: `IRHardNegative@1` shards with confirmed-negative or unknown dispositions
- Allowed effects: miner/tests/immutable shards and bounded solver calls
- Prohibited effects: timeout/unavailable/unknown as negative, same proposition siblings as negatives, unchecked model labels
- Acceptance criteria: mutation class coverage; every retained negative has supporting evidence; unknowns segregated; minimality recorded
- Required proof or evaluation evidence: counterexample/model/entailment receipts and seeded false-negative fixtures
- Lease and checkpoint policy: `LEASE-DEFAULT`, proof-shard fence and reducer CAS
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-041)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_hard_negatives.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_fuzzing.py, ipfs_datasets_py/data/ir_learning/pairs/negative/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_hard_negatives.py`
- Bundle: pgir/curriculum/negatives
- Parallel lane: negative-miner
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_hard_negatives.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_fuzzing.py, ipfs_datasets_py/data/ir_learning/pairs/negative/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: shards parallel; positive-equivalence index read-only

## PGIR-050 Capture Lean-capable proposal and proof-attempt traces

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: proof-loop
- Parent goal: PGIR-G070
- Subgoal: lean-proposal-traces
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/proof/leanstral_proof_provider.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py, test/api/test_agent_supervisor_leanstral_proof_provider.py, test/api/test_agent_supervisor_leanstral_proof_gate.py, data/agent_supervisor/proof_grounded_ir_learning/proof_traces/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` proof-eligible shards only
- Data split identity: `RESULT(PGIR-012)` train/development only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: configured Lean-capable model revision per attempt; no default proof authority
- Objective: Extend current candidate provider/transport with content-addressed attempts binding obligation, state, premises, proposals, model/tool versions, parse/elaboration/prover/kernel outcomes, errors, counterexamples, timeout, and resources.
- Depends on: PGIR-014, PGIR-020
- Resource profile: `RP-PROVER`
- Expected inputs: provider capabilities, typed proof grounding, source/bridge identities
- Expected outputs: strict proof-attempt trace and non-authority receipts
- Allowed effects: canonical proof provider/trace adapter and immutable traces
- Prohibited effects: provider self-verification, model claim as proof, timeout as false, unbounded calls
- Acceptance criteria: all J2 fields bound; malformed/stale/wrong-statement/replayed attempts fail; candidate authority remains false
- Required proof or evaluation evidence: provider protocol, proof-gate and adversarial receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, provider-call and trace-shard keys
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-050)`
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/leanstral_proof_provider.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py, test/api/test_agent_supervisor_leanstral_proof_provider.py, test/api/test_agent_supervisor_leanstral_proof_gate.py, data/agent_supervisor/proof_grounded_ir_learning/proof_traces/
- Validation: `python -m pytest -q test/api/test_agent_supervisor_leanstral_proof_provider.py test/api/test_agent_supervisor_leanstral_proof_gate.py`
- Bundle: pgir/proof/lean-proposals
- Parallel lane: lean-proposals
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/leanstral_proof_provider.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py, test/api/test_agent_supervisor_leanstral_proof_provider.py, test/api/test_agent_supervisor_leanstral_proof_gate.py, data/agent_supervisor/proof_grounded_ir_learning/proof_traces/
- Conflict policy: trace shards immutable; provider capacity leased

## PGIR-051 Extend proof-state tactician and curriculum projection

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: proof-loop
- Parent goal: PGIR-G070
- Subgoal: tactician
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, ipfs_accelerate_py/agent_supervisor/proof/counterexample_guided_tactician.py, ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py, ipfs_accelerate_py/agent_supervisor/proof/proof_directed_retrieval.py, test/api/test_counterexample_guided_tactician.py, test/api/test_goal_directed_tactician_integration.py, test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` proof-eligible shards only
- Data split identity: `RESULT(PGIR-012)` train/development only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: tactic/premise model revision per trace; candidate-only
- Objective: Add proof-state classification, premise/tactic ranking, goal decomposition, branch/cost/failure prediction and typed curriculum projections while preserving lifecycle/lease authority.
- Depends on: PGIR-050
- Resource profile: `RP-PROVER`
- Expected inputs: content-addressed proof traces and current tactician lifecycle
- Expected outputs: tactic/premise traces, verified-success/parse-type/counterexample/timeout curriculum classes
- Allowed effects: existing tactician/retrieval modules and tests
- Prohibited effects: tactic success as source-faithfulness proof, timeout label corruption, bypass of kernel gate
- Acceptance criteria: top-k/Recall@k/cost metrics; bounded branching; restart/fencing; only validated traces upgrade curriculum authority
- Required proof or evaluation evidence: lifecycle/restart/adversarial and trace-classification tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, proof-plan shard fence
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-051)`
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, ipfs_accelerate_py/agent_supervisor/proof/counterexample_guided_tactician.py, ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py, ipfs_accelerate_py/agent_supervisor/proof/proof_directed_retrieval.py, test/api/test_counterexample_guided_tactician.py, test/api/test_goal_directed_tactician_integration.py, test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Validation: `python -m pytest -q test/api/test_counterexample_guided_tactician.py test/api/test_goal_directed_tactician_integration.py test/api/test_goal_tactician_supervisor_lifecycle.py test/api/test_goal_tactician_supervisor_restart.py`
- Bundle: pgir/proof/tactician
- Parallel lane: tactician
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, ipfs_accelerate_py/agent_supervisor/proof/counterexample_guided_tactician.py, ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py, ipfs_accelerate_py/agent_supervisor/proof/proof_directed_retrieval.py, test/api/test_counterexample_guided_tactician.py, test/api/test_goal_directed_tactician_integration.py, test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Conflict policy: one proof-plan lifecycle writer; ranked candidates immutable

## PGIR-052 Integrate hammer portfolio with independent checker authority

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: proof-loop
- Parent goal: PGIR-G070
- Subgoal: hammer-kernel
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_resources.py, ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py, ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py, test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_multi_prover_router.py, test/api/test_agent_supervisor_kernel_verification.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` proof-eligible shards only
- Data split identity: `RESULT(PGIR-012)` train/development only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: premise/tactic checkpoint identity per candidate; no authority
- Objective: Route ATP/SMT/model-checker/hammer candidates through existing datasets hammer and current multi-prover resource policy, requiring native reconstruction or independent kernel/checker acceptance for proof authority.
- Depends on: PGIR-050
- Resource profile: `RP-PROVER`
- Expected inputs: proof traces, hammer corpus/capabilities, checker environments
- Expected outputs: hammer/counterexample/checker traces and authoritative dispositions
- Allowed effects: existing integration/router/checker modules and tests
- Prohibited effects: new prover/cache, candidate certificate as proof, stale environment/statement receipt reuse
- Acceptance criteria: authority lattice enforced; wrong theorem/environment/version/timeout/admit/sorry fail closed; solver counterexamples retain scope/bounds
- Required proof or evaluation evidence: multi-prover, kernel, hammer reconstruction and adversarial receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, separate solver/checker shard fences
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-052)`
- Outputs: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_resources.py, ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py, ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py, test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_multi_prover_router.py, test/api/test_agent_supervisor_kernel_verification.py
- Validation: `python -m pytest -q test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py test/api/test_agent_supervisor_multi_prover_router.py test/api/test_agent_supervisor_kernel_verification.py`
- Bundle: pgir/proof/hammer-kernel
- Parallel lane: hammer-kernel
- Predicted files: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_resources.py, ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py, ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py, test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_multi_prover_router.py, test/api/test_agent_supervisor_kernel_verification.py
- Conflict policy: router/checker mutation serialized; proof shards parallel

## PGIR-053 Implement bounded expert iteration

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: proof-loop
- Parent goal: PGIR-G070
- Subgoal: expert-iteration
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/proof/, ipfs_accelerate_py/agent_supervisor/objectives/, test/api/test_agent_supervisor_proof_workflow_e2e.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` train/development only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: exact current/candidate checkpoint per round
- Objective: Implement the generate-parse/type-tactician/hammer-check-retain-refill-train-qualify loop with hard bounds on candidates, depth, calls, solver time, rounds, and repeated examples.
- Depends on: PGIR-040, PGIR-041, PGIR-051, PGIR-052, PGIR-062
- Resource profile: `RP-MIXED`
- Expected inputs: checked pair/negative roots, proof/tactic/hammer/checker traces, campaign runtime
- Expected outputs: content-addressed curriculum revisions and round receipts
- Allowed effects: existing proof/objective/refill control surfaces
- Prohibited effects: hidden-test feedback, unverified success retention, unbounded loop, checkpoint self-promotion
- Acceptance criteria: each terminal/timeout/unavailable class maps correctly; no-progress/repetition bounds deterministic; resume exact
- Required proof or evaluation evidence: expert-iteration fixture, crash/restart, authority and exhaustion tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, per-round curriculum/checkpoint/proof fences
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-053)`
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/, ipfs_accelerate_py/agent_supervisor/objectives/, test/api/test_agent_supervisor_proof_workflow_e2e.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: `python -m pytest -q test/api/test_agent_supervisor_proof_workflow_e2e.py test/api/test_agent_supervisor_self_improvement_refill.py`
- Bundle: pgir/proof/expert-iteration
- Parallel lane: expert-iteration
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/, ipfs_accelerate_py/agent_supervisor/objectives/, test/api/test_agent_supervisor_proof_workflow_e2e.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Conflict policy: round coordinator exclusive; stage workers use immutable inputs

## PGIR-060 Implement IR learning campaign contracts and APIs

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: campaign
- Parent goal: PGIR-G080
- Subgoal: campaign-work-graph
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/objectives/, ipfs_accelerate_py/agent_supervisor/planning/, test/api/test_agent_supervisor_formal_plan_compiler.py, test/api/test_agent_supervisor_formal_plan_validator.py, test/api/test_agent_supervisor_control_plane.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: explicit none/current/candidate per generated task
- Objective: Add `IRLearningCampaign@1` and create/plan/status/steer/refill/proof-replay/compare/promote/reject/report operations using current objective/planning/control contracts and all required work-graph roles.
- Depends on: PGIR-014
- Resource profile: `RP-CPU-M`
- Expected inputs: frozen input root, task metadata contract, existing planners
- Expected outputs: strict campaign/role/task schemas and deterministic dependency projection
- Allowed effects: existing objective/planning/control extension and tests
- Prohibited effects: semantic definitions in accelerator, new agent framework, prompt-selected authority/hidden labels
- Acceptance criteria: every task field required by this board is validated; imports side-effect free; task revision binds unresolved dependency outputs before lease
- Required proof or evaluation evidence: schema/parity/adversarial plan-admission tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `campaign-plan`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-060)`
- Outputs: ipfs_accelerate_py/agent_supervisor/objectives/, ipfs_accelerate_py/agent_supervisor/planning/, test/api/test_agent_supervisor_formal_plan_compiler.py, test/api/test_agent_supervisor_formal_plan_validator.py, test/api/test_agent_supervisor_control_plane.py
- Validation: `python -m pytest -q test/api/test_agent_supervisor_formal_plan_compiler.py test/api/test_agent_supervisor_formal_plan_validator.py test/api/test_agent_supervisor_control_plane.py`
- Bundle: pgir/campaign/contracts
- Parallel lane: campaign-contracts
- Predicted files: ipfs_accelerate_py/agent_supervisor/objectives/, ipfs_accelerate_py/agent_supervisor/planning/, test/api/test_agent_supervisor_formal_plan_compiler.py, test/api/test_agent_supervisor_formal_plan_validator.py, test/api/test_agent_supervisor_control_plane.py
- Conflict policy: campaign schema/control catalog exclusive

## PGIR-061 Extend resource admission and safe pipeline overlap

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: campaign
- Parent goal: PGIR-G080
- Subgoal: resource-scheduling
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/scheduler_metrics.py, test/api/test_agent_supervisor_resource_scheduler.py, test/api/test_agent_supervisor_provider_batch_scheduler.py, test/api/test_agent_supervisor_scheduler_metrics.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` identity only
- Data split identity: `RESULT(PGIR-012)` identity only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: candidate checkpoint resource claims per task
- Objective: Model CPU/GPU/prover/I/O/token/provider/network resource classes and overlap safe stages while blocking unsealed data, stale traces, checkpoint collisions, and incompatible tokenizer mutation.
- Depends on: PGIR-060
- Resource profile: `RP-CPU-M`
- Expected inputs: campaign work graph and current schedulers
- Expected outputs: stage admission profiles, backpressure/fairness/overlap receipts
- Allowed effects: current runtime schedulers/metrics and tests
- Prohibited effects: second scheduler, missing telemetry admission for production, concurrent shared authority mutation
- Acceptance criteria: safe overlap scenarios pass; prohibited overlap fails; no over-admission/starvation; cancellation/timeouts observable
- Required proof or evaluation evidence: deterministic resource simulations and concurrency tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, scheduler policy key; resource subleases fenced
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-061)`
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/scheduler_metrics.py, test/api/test_agent_supervisor_resource_scheduler.py, test/api/test_agent_supervisor_provider_batch_scheduler.py, test/api/test_agent_supervisor_scheduler_metrics.py
- Validation: `python -m pytest -q test/api/test_agent_supervisor_resource_scheduler.py test/api/test_agent_supervisor_provider_batch_scheduler.py test/api/test_agent_supervisor_scheduler_metrics.py`
- Bundle: pgir/campaign/resources
- Parallel lane: resource-scheduler
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/scheduler_metrics.py, test/api/test_agent_supervisor_resource_scheduler.py, test/api/test_agent_supervisor_provider_batch_scheduler.py, test/api/test_agent_supervisor_scheduler_metrics.py
- Conflict policy: extend existing scheduler only

## PGIR-062 Add checkpoint, resume, leases, fencing, and refill policy

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: campaign
- Parent goal: PGIR-G080
- Subgoal: durable-runtime
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/runtime/, ipfs_accelerate_py/agent_supervisor/merge/, ipfs_accelerate_py/agent_supervisor/rescue/, ipfs_accelerate_py/agent_supervisor/self_improvement/, ipfs_accelerate_py/agent_supervisor/todo_daemon/, test/api/test_agent_supervisor_fault_recovery_v2.py, test/api/test_agent_supervisor_daemon_recovery_lease.py, test/api/test_agent_supervisor_worktree_lifecycle.py, test/api/test_agent_supervisor_autonomous_unstall.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: strict training checkpoint binding defined by task
- Objective: Reuse current worktree/lease/CAS/artifact/recovery/refill mechanisms for corpus/split/tokenizer/checkpoint/run/proof/evaluation/promotion/publication ownership, compatible resume, deterministic curriculum priority/triggers, and loop bounds.
- Depends on: PGIR-060, PGIR-061
- Resource profile: `RP-CPU-M`
- Expected inputs: campaign/resource contracts and existing durable runtime
- Expected outputs: learning checkpoint/resume adapter, lease keys, refill scoring/limits, crash recovery receipts
- Allowed effects: existing runtime/merge/rescue/refill/daemon extensions and tests
- Prohibited effects: new scheduler/checkpoint store, overwrite without fence, incompatible resume, mutable promotion authority
- Acceptance criteria: checkpoint binds architecture/weights/optimizer/scheduler/tokenizer/vocab/cursor/corpus/split/curriculum/loss/random/env/code/compiler; all refill triggers bounded; restart exactly once
- Required proof or evaluation evidence: stale-fence, duplicate-writer, incompatible-resume, no-progress, crash/restart tests
- Lease and checkpoint policy: `LEASE-DEFAULT`; all named L3 resources get distinct keys
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-062)`
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/, ipfs_accelerate_py/agent_supervisor/merge/, ipfs_accelerate_py/agent_supervisor/rescue/, ipfs_accelerate_py/agent_supervisor/self_improvement/, ipfs_accelerate_py/agent_supervisor/todo_daemon/, test/api/test_agent_supervisor_fault_recovery_v2.py, test/api/test_agent_supervisor_daemon_recovery_lease.py, test/api/test_agent_supervisor_worktree_lifecycle.py, test/api/test_agent_supervisor_autonomous_unstall.py
- Validation: `python -m pytest -q test/api/test_agent_supervisor_fault_recovery_v2.py test/api/test_agent_supervisor_daemon_recovery_lease.py test/api/test_agent_supervisor_worktree_lifecycle.py test/api/test_agent_supervisor_autonomous_unstall.py`
- Bundle: pgir/campaign/durability
- Parallel lane: campaign-durability
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/, ipfs_accelerate_py/agent_supervisor/merge/, ipfs_accelerate_py/agent_supervisor/rescue/, ipfs_accelerate_py/agent_supervisor/self_improvement/, ipfs_accelerate_py/agent_supervisor/todo_daemon/, test/api/test_agent_supervisor_fault_recovery_v2.py, test/api/test_agent_supervisor_daemon_recovery_lease.py, test/api/test_agent_supervisor_worktree_lifecycle.py, test/api/test_agent_supervisor_autonomous_unstall.py
- Conflict policy: shared runtime mutations serialized; immutable shards parallel

## PGIR-070 Implement checkpoint lifecycle and promotion manifest

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G090
- Subgoal: checkpoint-promotion
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_autoencoder_checkpoint.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/checkpoints.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: output `IRCheckpointManifest@1`; legacy manifest remains separate
- Objective: Unify semantic checkpoint identity and closed lifecycle without conflating existing formalization advisor and modal state manifests; define all M1 fields and side outcomes.
- Depends on: PGIR-032, PGIR-062
- Resource profile: `RP-CPU-M`
- Expected inputs: loss/tokenizer/campaign checkpoint contracts
- Expected outputs: strict checkpoint manifest/lifecycle verifier and compatibility adapters
- Allowed effects: existing semantic checkpoint modules/tests
- Prohibited effects: loss-only promotion, self-promotion, ambiguous current pointer, incompatible manifest aliasing
- Acceptance criteria: all lifecycle transitions validated; torn/corrupt/stale/mismatched state quarantined; exact artifact identities complete
- Required proof or evaluation evidence: golden manifests, transition/adversarial/restart tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive checkpoint-write and promotion keys
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-070)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_autoencoder_checkpoint.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/checkpoints.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_modal_autoencoder_checkpoint.py`
- Bundle: pgir/qualification/checkpoints
- Parallel lane: checkpoint-lifecycle
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_autoencoder_checkpoint.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/checkpoints.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: semantic owner defines manifest; accelerator only executes lifecycle

## PGIR-071 Implement comprehensive evaluation and statistical gates

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: evaluation
- Parent goal: PGIR-G090
- Subgoal: evaluation-suite
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_semantic_metrics.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_family_evaluator.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_metric_lineage.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_uncertainty.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_evaluation_artifacts.py, ipfs_datasets_py/benchmarks/semantic_roundtrip/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)` frozen development/calibration/test/OOD; hidden labels evaluator-only
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: exact candidate and baseline manifest per comparison
- Objective: Add `IREvaluationSuite@1` covering N1-N8 separately for compiler/decompiler with tokenizer-comparability rules, paired bootstrap CIs, noninferiority margins, and significance corrections.
- Depends on: PGIR-023, PGIR-033, PGIR-040, PGIR-041, PGIR-052, PGIR-070
- Resource profile: `RP-MIXED`
- Expected inputs: frozen heldouts, traces, checkpoints, deterministic baseline
- Expected outputs: token/latent/retrieval/structural/semantic/proof/readability/calibration/OOD/statistical reports
- Allowed effects: existing evaluation/benchmark paths and immutable reports
- Prohibited effects: hidden-test tuning, readability overriding semantics, noisy point-estimate promotion, incomparable tokenizer CE claim
- Acceptance criteria: every listed N metric has denominator/CI/strata or explicit unsupported reason; compiler/decompiler separate; false-neighbor analysis retained
- Required proof or evaluation evidence: metric goldens, paired comparison fixtures, independent proof receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, per-evaluation-shard fence and report reducer CAS
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-071)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_semantic_metrics.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_family_evaluator.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_metric_lineage.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_uncertainty.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_evaluation_artifacts.py, ipfs_datasets_py/benchmarks/semantic_roundtrip/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py`
- Bundle: pgir/qualification/evaluation
- Parallel lane: evaluation-suite
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_semantic_metrics.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_family_evaluator.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_metric_lineage.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_uncertainty.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_evaluation_artifacts.py, ipfs_datasets_py/benchmarks/semantic_roundtrip/, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/
- Conflict policy: metric schema lease; shards read immutable checkpoints

## PGIR-072 Implement deterministic promotion comparison and policy admission

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G090
- Subgoal: promotion-gates
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/validation/, ipfs_accelerate_py/agent_supervisor/control/, ipfs_accelerate_py/agent_supervisor/merge/, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_control_transactions.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: `RESULT(PGIR-070)` manifests being compared
- Objective: Execute M2/M3 gates with current validation/control/lease/CAS policy, fresh proof evidence, optional human approval, and compare-and-swap promoted pointer.
- Depends on: PGIR-060, PGIR-062, PGIR-070, PGIR-071
- Resource profile: `RP-CPU-M`
- Expected inputs: verified manifests/reports/policy/current pointer
- Expected outputs: deterministic promote/reject/regressed/inconclusive decision and audit receipt
- Allowed effects: current validation/control/merge coordination extensions; CAS pointer only when admitted
- Prohibited effects: model self-promotion, test-set selection, lowered semantic/proof minima, overwrite without lease
- Acceptance criteria: all M2 gates represented and non-compensable; identical evidence/policy yields identical decision; stale CAS loses
- Required proof or evaluation evidence: policy/adversarial/concurrency/human-approval fixtures
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive `promotion-pointer`
- Rollback procedure: `ROLLBACK-DEFAULT`; CAS restore prior pointer only with new decision
- Result identity: `RESULT(PGIR-072)`
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/, ipfs_accelerate_py/agent_supervisor/control/, ipfs_accelerate_py/agent_supervisor/merge/, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_control_transactions.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Validation: `python -m pytest -q test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_control_transactions.py test/api/test_agent_supervisor_parallel_acceptance_flow.py`
- Bundle: pgir/qualification/promotion
- Parallel lane: promotion-gate
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/, ipfs_accelerate_py/agent_supervisor/control/, ipfs_accelerate_py/agent_supervisor/merge/, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_control_transactions.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Conflict policy: promotion authority serialized and independent of evaluator/model

## PGIR-080 Publish stable semantic APIs

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P1
- Track: api
- Parent goal: PGIR-G100
- Subgoal: datasets-public-api
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/logic/, ipfs_datasets_py/tests/integration/logic/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: `RESULT(PGIR-070)` schema
- Objective: Expose the O1-equivalent corpus/split/example/compile/decompile/translate/pair/evaluate/verify/publish APIs over canonical implementations without import side effects.
- Depends on: PGIR-020, PGIR-021, PGIR-022, PGIR-041, PGIR-070, PGIR-071
- Resource profile: `RP-CPU-M`
- Expected inputs: admitted semantic contracts/implementations
- Expected outputs: stable reviewed exports, docs, compatibility adapters and parity tests
- Allowed effects: existing logic package exports/adapters/tests
- Prohibited effects: daemon start on import, operational scheduling semantics, duplicate implementation behind facade
- Acceptance criteria: exact signatures/discovery/versioning; all APIs delegate to canonical owners; cold import no process/network
- Required proof or evaluation evidence: API parity, import isolation, compatibility and end-to-end tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, key `datasets-public-api`
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-080)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/, ipfs_datasets_py/tests/integration/logic/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_ir_compatibility_exports.py`
- Bundle: pgir/api/datasets
- Parallel lane: datasets-api
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/, ipfs_datasets_py/tests/integration/logic/
- Conflict policy: public export changes serialized after implementations settle

## PGIR-081 Publish stable operational campaign APIs and prompt handoff

- Status: todo
- Completion: validated-implementation
- Is schedulable: true
- Priority: P1
- Track: api
- Parent goal: PGIR-G100
- Subgoal: accelerator-public-api
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/control/, ipfs_accelerate_py/agent_supervisor/objectives/, ipfs_accelerate_py/agent_supervisor/planning/, ipfs_accelerate_py/agent_supervisor/runtime/, ipfs_accelerate_py/agent_supervisor/proof/, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_prompt_plan_admission.py, test/api/test_agent_supervisor_v2_public_api.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` allowlisted identities
- Data split identity: `RESULT(PGIR-012)` immutable membership
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: `RESULT(PGIR-070)`
- Objective: Expose O2-equivalent create/plan/start/resume/status/steer/refill/proof-replay/compare/promote/reject/report operations and validate O3 prompts without expanding authority.
- Depends on: PGIR-060, PGIR-062, PGIR-072, PGIR-080
- Resource profile: `RP-CPU-M`
- Expected inputs: campaign/control/promotion contracts
- Expected outputs: Python/CLI/MCP-parity operational APIs and prompt policy
- Allowed effects: existing control/catalog/domain packages and tests
- Prohibited effects: secret/hidden-label access, prompt-selected authority/data/promotion, semantic redefinition
- Acceptance criteria: one transport-neutral contract; auth/lease/idempotency/effects enforced; resume exact; cold discovery side-effect free
- Required proof or evaluation evidence: Python/CLI/MCP parity, authorization, adversarial prompt and restart tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, control catalog and lifecycle keys
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-081)`
- Outputs: ipfs_accelerate_py/agent_supervisor/control/, ipfs_accelerate_py/agent_supervisor/objectives/, ipfs_accelerate_py/agent_supervisor/planning/, ipfs_accelerate_py/agent_supervisor/runtime/, ipfs_accelerate_py/agent_supervisor/proof/, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_prompt_plan_admission.py, test/api/test_agent_supervisor_v2_public_api.py
- Validation: `python -m pytest -q test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_prompt_plan_admission.py test/api/test_agent_supervisor_v2_public_api.py`
- Bundle: pgir/api/accelerator
- Parallel lane: accelerator-api
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/, ipfs_accelerate_py/agent_supervisor/objectives/, ipfs_accelerate_py/agent_supervisor/planning/, ipfs_accelerate_py/agent_supervisor/runtime/, ipfs_accelerate_py/agent_supervisor/proof/, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_prompt_plan_admission.py, test/api/test_agent_supervisor_v2_public_api.py
- Conflict policy: control catalog mutation serialized

## PGIR-090 Implement append-only IR release packaging

- Status: completed
- Completion: validated-implementation
- Is schedulable: true
- Priority: P1
- Track: publication
- Parent goal: PGIR-G100
- Subgoal: dataset-checkpoint-publication
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/ipfs_datasets_py/huggingface/, ipfs_datasets_py/tests/unit/logic/ir_learning/publication/, ipfs_datasets_py/data/ir_learning/releases/
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` and exact output release revision only after upload
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: admitted `RESULT(PGIR-070)` only
- Objective: Extend current HF read/release/publisher components for separate P1 configs, complete dataset/checkpoint cards, P4 evidence, immutable roots, dry-run and append-only publication.
- Depends on: PGIR-072, PGIR-080
- Resource profile: `RP-IO-PINNED`
- Expected inputs: qualified release/checkpoint/evaluation/proof manifests and publication policy
- Expected outputs: local release package, card/config manifests, dry-run/upload receipt
- Allowed effects: local packaging; remote upload only with explicit publication lease and qualification/human authority
- Prohibited effects: ambiguous overwrite, heterogeneous auto-detected schema, unrestricted publication, secret exposure
- Acceptance criteria: all configs/cards/evidence complete; source/derived counts distinct; re-run idempotent; remote revision captured
- Required proof or evaluation evidence: package validation, dry-run, partial-upload/retry/idempotency tests
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive `hf-publication:<repo>` fence
- Rollback procedure: `ROLLBACK-DEFAULT`; never delete published version, publish revocation/supersession record
- Result identity: `RESULT(PGIR-090)`
- Outputs: ipfs_datasets_py/ipfs_datasets_py/huggingface/, ipfs_datasets_py/tests/unit/logic/ir_learning/publication/, ipfs_datasets_py/data/ir_learning/releases/
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/ir_learning/publication`
- Bundle: pgir/publication/package
- Parallel lane: release-packager
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/huggingface/, ipfs_datasets_py/tests/unit/logic/ir_learning/publication/, ipfs_datasets_py/data/ir_learning/releases/
- Conflict policy: one release root and publication pointer writer

## PGIR-100 Add integrated dataset, proof, training, and recovery security

- Status: todo
- Completion: validated-implementation
- Is schedulable: true
- Priority: P0
- Track: security
- Parent goal: PGIR-G100
- Subgoal: fail-closed-security
- Owning repository: ipfs_accelerate_py
- Owned paths: ipfs_accelerate_py/agent_supervisor/validation/, ipfs_accelerate_py/agent_supervisor/rescue/, ipfs_accelerate_py/agent_supervisor/proof/, ipfs_accelerate_py/agent_supervisor/todo_daemon/, test/security/, test/api/test_agent_supervisor_fault_recovery_v2.py, test/api/test_agent_supervisor_process_tree_fencing.py
- Base source revisions: `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` hostile and admitted fixtures
- Data split identity: `RESULT(PGIR-012)` including hidden-label policy
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: `RESULT(PGIR-070)` hostile/admitted fixtures
- Objective: Enforce Q1-Q4 across dataset intake, proof authority, training state, leases, checkpoints, promotion and upload; inject failures at every material stage and prove safe restart.
- Depends on: PGIR-062, PGIR-070, PGIR-081, PGIR-090
- Resource profile: `RP-MIXED`
- Expected inputs: all semantic/operational security policies and hostile fixtures
- Expected outputs: integrated fail-closed gates, fault matrix, recovery evidence
- Allowed effects: existing validation/rescue/proof/daemon extensions and tests
- Prohibited effects: remote code, policy prompt injection, hidden labels, forged proof/promotion, unsafe cleanup
- Acceptance criteria: every listed Q rejection/failure injected; partial checkpoints rejected; immutable evidence preserved; duplicate accepted work zero
- Required proof or evaluation evidence: adversarial/property/crash/restart/concurrency receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, fault-scoped leases; no production pointer mutation in tests
- Rollback procedure: `ROLLBACK-DEFAULT`
- Result identity: `RESULT(PGIR-100)`
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/, ipfs_accelerate_py/agent_supervisor/rescue/, ipfs_accelerate_py/agent_supervisor/proof/, ipfs_accelerate_py/agent_supervisor/todo_daemon/, test/security/, test/api/test_agent_supervisor_fault_recovery_v2.py, test/api/test_agent_supervisor_process_tree_fencing.py
- Validation: `python -m pytest -q test/security test/api/test_agent_supervisor_fault_recovery_v2.py test/api/test_agent_supervisor_process_tree_fencing.py`
- Bundle: pgir/security/recovery
- Parallel lane: security-recovery
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/, ipfs_accelerate_py/agent_supervisor/rescue/, ipfs_accelerate_py/agent_supervisor/proof/, ipfs_accelerate_py/agent_supervisor/todo_daemon/, test/security/, test/api/test_agent_supervisor_fault_recovery_v2.py, test/api/test_agent_supervisor_process_tree_fencing.py
- Conflict policy: hostile tests isolated; production state/pointers protected

## PGIR-110 Run R1-R6 controlled campaign

- Status: todo
- Completion: evaluation-evidence
- Is schedulable: true
- Priority: P0
- Track: experiments
- Parent goal: PGIR-G110
- Subgoal: controlled-comparisons
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/experiments/**
- Base source revisions: exact post-implementation repository commits plus `SRCSET-1` ancestry
- Source dataset revisions: `RESULT(PGIR-011)`
- Data split identity: immutable `RESULT(PGIR-012)` identical across R1-R6
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: one content-addressed manifest per R1-R6 arm/seed
- Objective: Execute deterministic, CE-only, CE+cosine, contrastive, full multi-task, and proof-grounded curriculum arms under identical frozen heldouts and bounded resources.
- Depends on: PGIR-033, PGIR-041, PGIR-053, PGIR-062, PGIR-071, PGIR-081, PGIR-100
- Resource profile: `RP-MIXED`
- Expected inputs: frozen campaign, architectures/losses/pairs/proof loop/evaluator/security
- Expected outputs: all arm checkpoints, manifests, actual metrics/CIs/costs/failures and comparison report
- Allowed effects: isolated training/proof/evaluation checkpoints and immutable experiment artifacts
- Prohibited effects: hidden-test tuning, best-test selection, failed experiment deletion, shared checkpoint writes, threshold weakening
- Acceptance criteria: same heldouts/seeds policy; every R metric reported; bounded exhaustion typed; no fabricated target attainment
- Required proof or evaluation evidence: training/checkpoint/proof/evaluation/resource receipts and paired statistical report
- Lease and checkpoint policy: `LEASE-DEFAULT`, separate arm/seed/checkpoint/proof/evaluation leases; reducer CAS
- Rollback procedure: `ROLLBACK-DEFAULT`; reject/quarantine failed arm, preserve artifacts
- Result identity: `RESULT(PGIR-110)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/experiments/
- Validation: `python -m pytest -q test/api/test_agent_supervisor_proof_workflow_e2e.py test/api/test_agent_supervisor_scheduler.py`
- Bundle: pgir/experiments/r1-r6
- Parallel lane: experiment-orchestrator
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/experiments/
- Conflict policy: arms parallel only after freeze; promotion/test reducer independent

## PGIR-111 Qualify, publish or reject, and issue the next board

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G110
- Subgoal: final-decision-report
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Base source revisions: exact final repository/source ancestry from `RESULT(PGIR-110)`
- Source dataset revisions: `RESULT(PGIR-011)` and any append-only output revision from `RESULT(PGIR-090)`
- Data split identity: `RESULT(PGIR-012)`
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: all R-arm manifests and selected candidate if any
- Objective: Apply all final criteria, produce the required 32-section factual report, deterministic promote/reject/no-go/resource-exhausted decision, authorized append-only publication if qualified, and exact next improvement board.
- Depends on: PGIR-072, PGIR-090, PGIR-100, PGIR-110
- Resource profile: `RP-CPU-M`
- Expected inputs: every accepted task result, experiment comparisons, current promotion/publication authorities
- Expected outputs: final report/decision/limitations/publication receipts/next board
- Allowed effects: qualification artifacts; promotion/publication only under current independent authority and leases
- Prohibited effects: universal understanding claim, missing-failure suppression, model self-evaluation/promotion, unauthorized upload
- Acceptance criteria: all 16 final acceptance criteria and 32 report sections resolved with evidence or explicit no-go; exact required qualified-claim text used only if gates pass
- Required proof or evaluation evidence: manifest/evaluation/proof/promotion/publication verifiers and complete result graph
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive final-decision/promotion/publication keys
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-111)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Validation: `python -m pytest -q test/api/test_agent_supervisor_goal_completion.py test/api/test_agent_supervisor_proof_goal_completion.py`
- Bundle: pgir/qualification/final
- Parallel lane: final-qualifier
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Conflict policy: one independent qualification/promotion authority; evaluator/model cannot hold it

## PGIR-112 Resolve 1 preflight-conflicting backlogged worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: f3f4420cab79db4b6f696585de85d53fbc16b5a3
- Dedupe key: reconciliation_guardrail:preflight_merge_conflict
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-112-reconciliation-f3f4420cab79.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by preflight_merge_conflict. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-112-reconciliation-f3f4420cab79.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-113 Resolve 1 dirty backlogged worktrees blocked by unsupported_status

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: d68dc99c4973e7ab181751b002522fb231d9c7f6
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-113-reconciliation-d68dc99c4973.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by unsupported_status. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-113-reconciliation-d68dc99c4973.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-114 Resolve validation retry-budget failure for PGIR-030

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: PGIR-023
- Outputs: ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_autoencoder.py, ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py, ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/, /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-114-pgir-030-retry-budget.md

- Acceptance: Retry-budget guardrail filed this from repeated validation failures in PGIR-030. Use evidence in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-114-pgir-030-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release PGIR-030 from strategy blocked_tasks.

## PGIR-115 Resolve dirty main checkout blocking 1 worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: e2940f4df61491aed9a68997a1292c79799d36ea
- Dedupe key: reconciliation_guardrail:main_checkout_dirty
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-115-reconciliation-e2940f4df614.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by main_checkout_dirty. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-115-reconciliation-e2940f4df614.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-116 Resolve 1 dirty backlogged worktrees blocked by unsupported_status

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: 33f7191f76900019322b08895de7c2ebb9a98774
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-116-reconciliation-33f7191f7690.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by unsupported_status. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-17-pgir-116-reconciliation-33f7191f7690.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-117 Resolve 1 dirty backlogged worktrees blocked by unsupported_status

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: 41d8412c5bf982feac6f41b72bae85f485ec03b9
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-117-reconciliation-41d8412c5bf9.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by unsupported_status. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-117-reconciliation-41d8412c5bf9.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-118 Resolve 1 dirty backlogged worktrees blocked by unsupported_status

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: e9e580d7cef5d0772e1895fcc201243316fbb94a
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-118-reconciliation-e9e580d7cef5.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by unsupported_status. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-118-reconciliation-e9e580d7cef5.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-119 Resolve 1 dirty backlogged worktrees blocked by unsupported_status

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: fe270214fc72a4ca0eca4b0e8e1bbfaa72705443
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-119-reconciliation-fe270214fc72.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by unsupported_status. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-119-reconciliation-fe270214fc72.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-120 Resolve 1 dirty backlogged worktrees blocked by unsupported_status

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: cc926a3c3ee86dfd1b97a1ae63663b07bb259a9d
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-120-reconciliation-cc926a3c3ee8.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by unsupported_status. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-120-reconciliation-cc926a3c3ee8.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-121 Resolve 1 dirty backlogged worktrees blocked by submodule_gitlink_diverged

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P2
- Track: ops
- Fingerprint: e2fea239d87284c4682d9643d136eb867410380f
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:submodule_gitlink_diverged
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-121-reconciliation-e2fea239d872.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by submodule_gitlink_diverged. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-121-reconciliation-e2fea239d872.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## PGIR-122 Resolve 1 dirty backlogged worktrees blocked by unsupported_status

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: 18ab237d43cf4a71c16c74605e3919ee39de2a9b
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:unsupported_status
- Depends on:
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: test -f /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-122-reconciliation-18ab237d43cf.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by unsupported_status. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-18-pgir-122-reconciliation-18ab237d43cf.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.
