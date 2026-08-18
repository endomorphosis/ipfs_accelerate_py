# Proof-Grounded IR Learning Fabric Next Improvement Board

This board is issued by `RESULT(PGIR-111)`. It is the exact next
training and data-improvement board after the first campaign's
`no_go` decision `baguqeeraejs56hwzs3bqtgzoayrc2fxwgfnhcsxjthi4dh7gh64wptlkfhwa`.

It does not replace the protected original board. Workers must keep
`docs/architecture/proof_grounded_ir_learning.todo.md`,
`docs/architecture/proof_grounded_ir_learning.objectives.md`, and
`data/agent_supervisor/proof_grounded_ir_learning/justice_dao_pinset.yaml`
unchanged unless a later admitted plan revision says otherwise.

The freeze chain remains binding. No learned-model, pair-mining,
proof-curriculum, training, promotion, or publication task may be
leased before a superseding `IRCampaignInputRoot@1` admits rights,
materialized corpus rows, complete required holdouts, and a current
tokenizer or an explicit deterministic-only restriction.

## Frozen identities inherited from RESULT(PGIR-111)

- `SRCSET-1` remains (`df93e91e6338c84a17c3208ef68b88de8566f78c`, `8d46a6d25dd006c8cab3c9d9612707d2a014e79c`).
- `JDAO-PINSET-1` remains SHA-256 `8e3a4b1bd81639393ddda35e5dfb3b95f9e7320afa898bde0b3eb9a0317a6b76` and still admits zero training repositories.
- `RESULT(PGIR-014)` `baguqeerai2ipwhyywztjob62ju5pokmm4o6unqqee3poyrabj37aby6fuoca` remains the current no-go freeze.
- Hidden-test commitment `sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded` stays sealed.
- `MODEL-LEGACY-1` remains artifact-only and never promotion authority.

## Why the previous campaign cannot be promoted

- `corpus_not_materialized`
- `historical_semantic_baseline_not_currently_qualified`
- `no_candidate_checkpoint`
- `no_learned_tokenizer_admitted`
- `no_rights_admitted_training_rows`
- `publication_not_authorized`
- `required_holdouts_insufficient`

No worker may treat historical `RESULT(PGIR-023)` fixture scores, the
PGIR-090 local package, or `MODEL-LEGACY-1` as current qualification.

## PGIR-200 Admit or permanently quarantine JusticeDAO source rights

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G020
- Subgoal: rights-admission
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/corpora/**
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Re-evaluate source and transformation rights for the 7,173 quarantined source rows and emit either a rights-admitted row set or a permanent quarantine with explicit residual gaps.
- Depends on: PGIR-111
- Resource profile: RP-IO-PINNED
- Expected inputs: JDAO-PINSET-1, RESULT(PGIR-004), RESULT(PGIR-011), RESULT(PGIR-111)
- Expected outputs: updated rights/quarantine manifests and an admitted-row count that is either positive or explicitly permanently zero
- Allowed effects: owned corpus rights/quarantine artifacts only
- Prohibited effects: silent un-quarantine, trust-remote-code, training, publication
- Acceptance criteria: every quarantined row has a fresh rights decision; training_admitted_rows is either >0 with cited licenses or remains 0 with a permanent no-go reason
- Required proof or evaluation evidence: rights receipts, license/cutoff/jurisdiction bindings, and a replayable admitted-row count
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `corpus-rights`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-200)`
- Outputs: ipfs_datasets_py/data/ir_learning/corpora/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py
- Bundle: pgir/next/rights
- Parallel lane: rights-admission
- Predicted files: ipfs_datasets_py/data/ir_learning/corpora/
- Conflict policy: exclusive corpus-rights writer

## PGIR-201 Materialize the sealed corpus after rights admission

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G020
- Subgoal: corpus-materialization
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/corpora/**
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Materialize only rights-admitted source rows into the sealed corpus root. Keep source and derived counts distinct and leave derivatives linked to their source CID groups.
- Depends on: PGIR-200
- Resource profile: RP-IO-PINNED
- Expected inputs: RESULT(PGIR-200) rights decision and RESULT(PGIR-011) manifests
- Expected outputs: a corpus root with materialized=true only when admitted rows exist; otherwise a documented still-unmaterialized no-go
- Allowed effects: owned corpus materialization artifacts
- Prohibited effects: inflating source counts with derivatives, hidden-test access, training
- Acceptance criteria: materialized flag matches the admitted-row set; source_count remains 7173 or a cited superseding count; derived_count stays separate
- Required proof or evaluation evidence: corpus_root, corpus_manifest, and load receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `corpus-materialize`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-201)`
- Outputs: ipfs_datasets_py/data/ir_learning/corpora/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py
- Bundle: pgir/next/corpus
- Parallel lane: corpus-materialize
- Predicted files: ipfs_datasets_py/data/ir_learning/corpora/
- Conflict policy: exclusive corpus-root writer

## PGIR-202 Populate the thirteen insufficient holdouts

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G020
- Subgoal: holdout-completion
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/splits/**
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Populate compiler, cross_reference, domain, exception, length, lineage, notation, premise, proof_library, publication, rare_operator, time, and type holdouts, or document why a named holdout remains impossible.
- Depends on: PGIR-201
- Resource profile: RP-CPU-M
- Expected inputs: RESULT(PGIR-012), RESULT(PGIR-201), hidden-test commitment
- Expected outputs: a superseding split root whose required holdouts are populated or explicitly permanently insufficient
- Allowed effects: owned split/holdout artifacts
- Prohibited effects: opening hidden tests, random-row splitting as the principal method, leakage-group splits
- Acceptance criteria: leakage audit still passes; hidden-test commitment unchanged; every previously insufficient holdout is populated or has a permanent no-go reason
- Required proof or evaluation evidence: holdout_report, leakage_report, and split_root identities
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `split-holdouts`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-202)`
- Outputs: ipfs_datasets_py/data/ir_learning/splits/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py
- Bundle: pgir/next/holdouts
- Parallel lane: holdout-completion
- Predicted files: ipfs_datasets_py/data/ir_learning/splits/
- Conflict policy: exclusive split-root writer

## PGIR-203 Admit a learned tokenizer or restrict the campaign to deterministic-only

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: model
- Parent goal: PGIR-G050
- Subgoal: tokenizer-admission
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/freeze/**
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Either freeze a compatible learned tokenizer/vocabulary CID or issue an explicit deterministic-only campaign restriction that keeps R2-R6 ineligible.
- Depends on: PGIR-202
- Resource profile: RP-CPU-M
- Expected inputs: RESULT(PGIR-030) architecture surfaces and the current tokenizer freeze policy
- Expected outputs: a superseding IRTokenizerFreezePolicy@1 that is either admitted or permanently deterministic-only
- Allowed effects: owned tokenizer-policy artifacts under a new freeze location
- Prohibited effects: mutating the current freeze in place, promoting MODEL-LEGACY-1, unfrozen vocabulary mutation
- Acceptance criteria: unknown tokens still fail closed; learned training remains unauthorized until a tokenizer is admitted
- Required proof or evaluation evidence: tokenizer policy CID and golden token-class receipts
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `tokenizer`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-203)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/freeze/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_modal_autoencoder.py
- Bundle: pgir/next/tokenizer
- Parallel lane: tokenizer-admission
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/freeze/
- Conflict policy: serial tokenizer freeze

## PGIR-204 Requalify or replace the historical R1 semantic baseline

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: evaluation
- Parent goal: PGIR-G040
- Subgoal: current-input-baseline
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/evaluations/deterministic/**
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Re-run or replace RESULT(PGIR-023) on current-input admitted rows and declared non-hidden partitions so the deterministic baseline is either currently qualified or explicitly retired.
- Depends on: PGIR-202
- Resource profile: RP-PROVER
- Expected inputs: RESULT(PGIR-021), RESULT(PGIR-022), RESULT(PGIR-202)
- Expected outputs: a current-input R1 report with E1 metrics, denominators, and tool versions, or a retirement receipt
- Allowed effects: owned deterministic evaluation artifacts
- Prohibited effects: hidden-test selection, missing metric as zero, treating historical fixture scores as current
- Acceptance criteria: either a current-input qualified R1 CID exists or the historical baseline is retired by CID
- Required proof or evaluation evidence: recipe, identities, strata, tool versions, and independent replay
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `evaluation:deterministic`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-204)`
- Outputs: ipfs_datasets_py/data/ir_learning/evaluations/deterministic/
- Validation: python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py
- Bundle: pgir/next/r1
- Parallel lane: deterministic-requalify
- Predicted files: ipfs_datasets_py/data/ir_learning/evaluations/deterministic/
- Conflict policy: one report reducer lease

## PGIR-205 Issue a superseding campaign input freeze

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G030
- Subgoal: superseding-freeze
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/freeze/**
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Bind the new rights, corpus, split, tokenizer, and baseline identities into a superseding IRCampaignInputRoot@1 whose previous_root_cid is the current no-go freeze.
- Depends on: PGIR-200, PGIR-201, PGIR-202, PGIR-203, PGIR-204
- Resource profile: RP-CPU-S
- Expected inputs: RESULT(PGIR-014) and all PGIR-200..204 outputs
- Expected outputs: a new freeze root, descendant task revisions, and a go or documented no-go
- Allowed effects: a separately located freeze; never overwrite the current root
- Prohibited effects: in-place freeze mutation, hidden-test access, promotion
- Acceptance criteria: previous_root_cid equals the current freeze; training tasks are eligible only when rights, corpus, holdouts, and tokenizer gates pass
- Required proof or evaluation evidence: independent freeze verifier and plan-admission receipt
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `campaign-freeze-root`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-205)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/freeze/
- Validation: python -m pytest -q test/api/test_agent_supervisor_formal_plan_validator.py test/api/test_agent_supervisor_task_identity.py
- Bundle: pgir/next/freeze
- Parallel lane: freeze-root
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/freeze/
- Conflict policy: global serial freeze barrier

## PGIR-206 Re-run R1-R6 on the superseding freeze

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: experiments
- Parent goal: PGIR-G110
- Subgoal: controlled-comparisons-v2
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/experiments/**
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Execute deterministic, CE-only, CE+cosine, contrastive, full multi-task, and proof-grounded arms on identical frozen heldouts only after the superseding freeze authorizes descendant execution.
- Depends on: PGIR-205
- Resource profile: RP-MIXED
- Expected inputs: superseding freeze, architectures, losses, pairs, proof loop, evaluator, security
- Expected outputs: arm checkpoints, actual metrics/CIs/costs/failures, and a comparison report
- Allowed effects: isolated training/proof/evaluation artifacts under a new campaign location
- Prohibited effects: hidden-test tuning, best-test selection, fabricated target attainment, shared checkpoint writes
- Acceptance criteria: same heldouts/seeds; every R metric reported; bounded exhaustion typed; no invented scores
- Required proof or evaluation evidence: training/checkpoint/proof/evaluation/resource receipts and paired statistical report
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `campaign:reducer`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-206)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/experiments/
- Validation: python -m pytest -q test/api/test_agent_supervisor_proof_workflow_e2e.py test/api/test_agent_supervisor_scheduler.py
- Bundle: pgir/next/r1-r6
- Parallel lane: experiment-orchestrator
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/experiments/
- Conflict policy: promotion/test reducer independent of trainers

## PGIR-207 Re-qualify, publish or reject, and issue the following board

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G110
- Subgoal: final-decision-report-v2
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Base source revisions: exact ancestry from `RESULT(PGIR-111)` plus `SRCSET-1`
- Source dataset revisions: `RESULT(PGIR-011)` until a superseding rights-admitted corpus exists
- Data split identity: `RESULT(PGIR-012)` until a superseding holdout root exists
- Compiler identity: `RESULT(PGIR-021)`
- Decompiler identity: `RESULT(PGIR-022)`
- Model checkpoint identity: none until a later admitted candidate exists
- Objective: Apply the same 16 final criteria and 32 report sections to RESULT(PGIR-206). Emit promote, reject, no-go, or resource-exhausted. Publish only if independently authorized. Issue the next board.
- Depends on: PGIR-072, PGIR-090, PGIR-100, PGIR-206
- Resource profile: RP-CPU-M
- Expected inputs: every accepted successor result, experiment comparisons, current promotion/publication authorities
- Expected outputs: successor final report, decision, publication receipt, and next board
- Allowed effects: qualification artifacts; promotion/publication only under current independent authority
- Prohibited effects: universal understanding claim, missing-failure suppression, model self-promotion, unauthorized upload
- Acceptance criteria: all 16 criteria and 32 sections resolved with evidence or explicit no-go; exact qualified-claim text used only if gates pass
- Required proof or evaluation evidence: manifest/evaluation/proof/promotion/publication verifiers and complete result graph
- Lease and checkpoint policy: `LEASE-DEFAULT`, exclusive key `final-decision`
- Rollback procedure: `ROLLBACK-DEFAULT`; append correction/supersession, never rewrite released evidence
- Result identity: `RESULT(PGIR-207)`
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_goal_completion.py test/api/test_agent_supervisor_proof_goal_completion.py
- Bundle: pgir/next/qualification
- Parallel lane: final-qualifier
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/qualification/, docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Conflict policy: one independent qualification/promotion authority; evaluator/model cannot hold it
