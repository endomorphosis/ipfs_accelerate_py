# Proof-Grounded IR Learning Fabric current-supervisor successor board

This is the executable `successor-v1` projection of `next.todo.md`. The four
completed anchors resolve dependencies on the protected historical board.
Every open task writes a new generation and must not overwrite historical
evidence.

## PGIR-072 Historical evaluation-contract completion anchor

- Status: completed
- Completion: historical-result-anchor
- Is schedulable: true
- Priority: P0
- Parent goal: PGIR-G090
- Goal id: PGIR-G090
- Board namespace: proof-grounded-ir-learning-successor-v1
- Objective: Bind the accepted historical evaluation-contract result for the PGIR-207 dependency.
- Depends on: none
- Expected outputs: immutable RESULT(PGIR-072) reference
- Acceptance: the protected original board records PGIR-072 completed
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: python scripts/validate_proof_grounded_ir_learning_successor_board.py --check-anchors
- Predicted files: docs/architecture/proof_grounded_ir_learning.todo.md
- Bundle: pgir/successor/anchors
- Conflict policy: read-only historical anchor

## PGIR-090 Historical publication-package completion anchor

- Status: completed
- Completion: historical-result-anchor
- Is schedulable: true
- Priority: P0
- Parent goal: PGIR-G100
- Goal id: PGIR-G100
- Board namespace: proof-grounded-ir-learning-successor-v1
- Objective: Bind the accepted historical local-package result without granting publication authority.
- Depends on: none
- Expected outputs: immutable RESULT(PGIR-090) reference
- Acceptance: the protected original board records PGIR-090 completed
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: python scripts/validate_proof_grounded_ir_learning_successor_board.py --check-anchors
- Predicted files: docs/architecture/proof_grounded_ir_learning.todo.md
- Bundle: pgir/successor/anchors
- Conflict policy: read-only historical anchor

## PGIR-100 Historical security-validation completion anchor

- Status: completed
- Completion: historical-result-anchor
- Is schedulable: true
- Priority: P0
- Parent goal: PGIR-G100
- Goal id: PGIR-G100
- Board namespace: proof-grounded-ir-learning-successor-v1
- Objective: Bind the accepted historical security and recovery result for the final qualifier.
- Depends on: none
- Expected outputs: immutable RESULT(PGIR-100) reference
- Acceptance: the protected original board records PGIR-100 completed
- Outputs: docs/architecture/proof_grounded_ir_learning.todo.md
- Validation: python scripts/validate_proof_grounded_ir_learning_successor_board.py --check-anchors
- Predicted files: docs/architecture/proof_grounded_ir_learning.todo.md
- Bundle: pgir/successor/anchors
- Conflict policy: read-only historical anchor

## PGIR-111 Historical no-go qualification completion anchor

- Status: completed
- Completion: historical-result-anchor
- Is schedulable: true
- Priority: P0
- Parent goal: PGIR-G110
- Goal id: PGIR-G110
- Board namespace: proof-grounded-ir-learning-successor-v1
- Objective: Bind the protected RESULT(PGIR-111) no-go decision as the predecessor of this successor generation.
- Depends on: none
- Expected outputs: decision CID baguqeeraejs56hwzs3bqtgzoayrc2fxwgfnhcsxjthi4dh7gh64wptlkfhwa
- Acceptance: the protected final report remains no_go and names PGIR-200 through PGIR-207
- Outputs: docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Validation: python scripts/validate_proof_grounded_ir_learning_successor_board.py --check-anchors
- Predicted files: docs/architecture/proof_grounded_ir_learning/final_report.md, docs/architecture/proof_grounded_ir_learning/next.todo.md
- Bundle: pgir/successor/anchors
- Conflict policy: read-only historical anchor

## PGIR-200 Admit or permanently quarantine JusticeDAO source rights

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G020
- Goal id: PGIR-G020
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: rights-admission
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/corpora/successor-v1/**
- Base source revisions: accelerator 22173f9cf4f357ab20040024f87af53c1cd89c9a; datasets c30ccbec997868b061c4cadac38d30468c46ea2d; historical RESULT(PGIR-111)
- Objective: Re-evaluate source and transformation rights for every quarantined source row and emit either an exact rights-admitted row set or a permanent quarantine with explicit residual gaps.
- Depends on: PGIR-111
- Resource class: network
- Expected inputs: JDAO-PINSET-1, historical corpus manifests, exact pinned source revisions, RESULT(PGIR-111)
- Expected outputs: successor-v1 rights, quarantine, source-release, and replay receipts with an admitted-row count that is positive or explicitly permanently zero
- Allowed effects: successor-v1 corpus rights and quarantine artifacts only
- Prohibited effects: silent un-quarantine, trust-remote-code, training, publication, legal-rights inference from repository visibility alone
- Acceptance: every quarantined source release and row has a fresh cited disposition; training_admitted_rows is positive with exact license authority or zero with a permanent no-go reason
- Required proof or evaluation evidence: license and source receipts, cutoff and jurisdiction bindings, row-count replay, current repository-forest identity
- Result identity: RESULT(PGIR-200) in successor-v1
- Outputs: ipfs_datasets_py/data/ir_learning/corpora/successor-v1/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py ipfs_datasets_py/tests/unit/logic/ir_learning/source/test_corpus_build.py ipfs_datasets_py/tests/unit/logic/ir_learning/source/test_successor_rights.py
- Bundle: pgir/successor/rights
- Parallel lane: rights-admission
- Predicted files: ipfs_datasets_py/data/ir_learning/corpora/successor-v1/
- Submodules: ipfs_datasets_py
- Exclusive group: corpus-rights-successor-v1
- Conflict policy: exclusive successor corpus-rights writer

## PGIR-201 Materialize the sealed corpus after rights admission

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G020
- Goal id: PGIR-G020
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: corpus-materialization
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/corpora/successor-v1/**
- Base source revisions: current successor branch plus RESULT(PGIR-200)
- Objective: Materialize only rights-admitted source rows into a new sealed corpus root while keeping source and derived counts distinct and retaining lineage to source CID groups.
- Depends on: PGIR-200
- Resource class: io-medium
- Expected inputs: RESULT(PGIR-200), historical corpus and lineage manifests
- Expected outputs: a successor-v1 corpus root and load receipt; when no rows are admitted, a replayable still-unmaterialized no-go
- Allowed effects: successor-v1 corpus materialization artifacts only
- Prohibited effects: derivative count inflation, hidden-test access, training, mutation of the historical corpus root
- Acceptance: materialized matches the admitted-row set; source and derived counts remain separate; every materialized row has admitted rights and lineage
- Required proof or evaluation evidence: corpus root, manifest, lineage graph, deterministic load and count receipts
- Result identity: RESULT(PGIR-201) in successor-v1
- Outputs: ipfs_datasets_py/data/ir_learning/corpora/successor-v1/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py
- Bundle: pgir/successor/corpus
- Parallel lane: corpus-materialize
- Predicted files: ipfs_datasets_py/data/ir_learning/corpora/successor-v1/
- Submodules: ipfs_datasets_py
- Exclusive group: corpus-root-successor-v1
- Conflict policy: exclusive successor corpus-root writer

## PGIR-202 Populate the thirteen insufficient holdouts

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: source-curation
- Parent goal: PGIR-G020
- Goal id: PGIR-G020
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: holdout-completion
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/splits/successor-v1/**
- Base source revisions: current successor branch plus RESULT(PGIR-201)
- Objective: Populate compiler, cross_reference, domain, exception, length, lineage, notation, premise, proof_library, publication, rare_operator, time, and type holdouts, or bind a permanent reason for each impossible holdout.
- Depends on: PGIR-201
- Resource class: cpu-medium
- Expected inputs: RESULT(PGIR-201), historical split root, unchanged hidden-test commitment
- Expected outputs: a successor split root whose required holdouts are populated or explicitly permanently insufficient
- Allowed effects: successor-v1 split and holdout artifacts only
- Prohibited effects: opening hidden tests, principal random-row splitting, leakage-group splitting, historical-root mutation
- Acceptance: leakage still passes; the hidden-test commitment is unchanged; every named holdout is populated or has a permanent no-go reason
- Required proof or evaluation evidence: holdout, leakage, split-root, and deterministic replay receipts
- Result identity: RESULT(PGIR-202) in successor-v1
- Outputs: ipfs_datasets_py/data/ir_learning/splits/successor-v1/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py
- Bundle: pgir/successor/holdouts
- Parallel lane: holdout-completion
- Predicted files: ipfs_datasets_py/data/ir_learning/splits/successor-v1/
- Submodules: ipfs_datasets_py
- Exclusive group: split-holdouts-successor-v1
- Conflict policy: exclusive successor split-root writer

## PGIR-203 Admit a learned tokenizer or restrict the campaign to deterministic-only

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: model
- Parent goal: PGIR-G050
- Goal id: PGIR-G050
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: tokenizer-admission
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/**
- Base source revisions: current successor branch plus RESULT(PGIR-202)
- Objective: Freeze a compatible learned tokenizer and vocabulary CID or issue an explicit deterministic-only restriction that keeps R2-R6 ineligible.
- Depends on: PGIR-202
- Resource class: cpu-medium
- Expected inputs: RESULT(PGIR-202), historical architecture surfaces, historical tokenizer policy
- Expected outputs: a successor-v1 IRTokenizerFreezePolicy that is admitted or permanently deterministic-only
- Allowed effects: successor-v1 tokenizer evidence only
- Prohibited effects: mutation of the historical freeze, MODEL-LEGACY-1 promotion, unfrozen vocabulary mutation
- Acceptance: unknown tokens fail closed and learned training stays unauthorized unless exact tokenizer and golden token-class evidence pass
- Required proof or evaluation evidence: tokenizer policy CID, vocabulary identity, golden token-class receipts
- Result identity: RESULT(PGIR-203) in successor-v1
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/tokenizer/
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_modal_autoencoder.py
- Bundle: pgir/successor/tokenizer
- Parallel lane: tokenizer-admission
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/tokenizer/
- Submodules: ipfs_datasets_py
- Exclusive group: tokenizer-successor-v1
- Conflict policy: serial successor tokenizer freeze

## PGIR-204 Requalify or replace the historical R1 semantic baseline

- Status: completed
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: evaluation
- Parent goal: PGIR-G040
- Goal id: PGIR-G040
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: current-input-baseline
- Owning repository: ipfs_datasets_py
- Owned paths: ipfs_datasets_py/data/ir_learning/evaluations/deterministic/successor-v1/**
- Base source revisions: current successor branch plus RESULT(PGIR-202)
- Objective: Re-run or replace the historical R1 result on current admitted rows and declared non-hidden partitions so it is currently qualified or explicitly retired.
- Depends on: PGIR-202
- Resource class: cpu-large
- Expected inputs: current compiler and decompiler, RESULT(PGIR-202), historical R1 identities
- Expected outputs: a current-input R1 report with denominators and tool versions, or an immutable retirement receipt
- Allowed effects: successor-v1 deterministic evaluation artifacts only
- Prohibited effects: hidden-test selection, missing metric as zero, historical fixture scores presented as current
- Acceptance: a current-input qualified R1 CID exists or the historical baseline is retired by CID
- Required proof or evaluation evidence: recipe, identities, strata, tool versions, independent replay
- Result identity: RESULT(PGIR-204) in successor-v1
- Outputs: ipfs_datasets_py/data/ir_learning/evaluations/deterministic/successor-v1/
- Validation: python -m pytest -q ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py
- Bundle: pgir/successor/r1
- Parallel lane: deterministic-requalify
- Predicted files: ipfs_datasets_py/data/ir_learning/evaluations/deterministic/successor-v1/
- Submodules: ipfs_datasets_py
- Exclusive group: deterministic-evaluation-successor-v1
- Conflict policy: one successor report reducer lease

## PGIR-205 Issue a superseding campaign input freeze

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G030
- Goal id: PGIR-G030
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: superseding-freeze
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/**
- Base source revisions: current successor branch plus RESULT(PGIR-200) through RESULT(PGIR-204) and RESULT(PGIR-208)
- Objective: Bind new rights, corpus, split, tokenizer, and baseline identities into a superseding campaign input root whose previous_root_cid is the historical no-go freeze.
- Depends on: PGIR-200, PGIR-201, PGIR-202, PGIR-203, PGIR-204, PGIR-208
- Resource class: cpu-small
- Expected inputs: historical RESULT(PGIR-014), RESULT(PGIR-200) through RESULT(PGIR-204), RESULT(PGIR-208)
- Expected outputs: a successor-v1 freeze root, descendant task revisions, plan-admission receipt, and go or documented no-go
- Allowed effects: successor-v1 freeze artifacts only
- Prohibited effects: historical freeze mutation, hidden-test access, promotion
- Acceptance: previous_root_cid binds the historical freeze; learned tasks are eligible only if rights, corpus, holdouts, tokenizer, and current-baseline gates pass
- Required proof or evaluation evidence: portable independent verifier, repository-forest binding, plan-admission receipt
- Result identity: RESULT(PGIR-205) in successor-v1
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/
- Validation: python -m pytest -q test/api/test_agent_supervisor_formal_plan_validator.py test/api/test_agent_supervisor_task_identity.py
- Bundle: pgir/successor/freeze
- Parallel lane: freeze-root
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/
- Submodules: ipfs_datasets_py
- Exclusive group: campaign-freeze-successor-v1
- Conflict policy: global serial successor freeze barrier

## PGIR-206 Re-run R1-R6 on the superseding freeze

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: experiments
- Parent goal: PGIR-G110
- Goal id: PGIR-G110
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: controlled-comparisons-v2
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1/**
- Base source revisions: current successor branch plus RESULT(PGIR-205)
- Objective: Run deterministic, CE-only, CE-plus-cosine, contrastive, full multi-task, and proof-grounded arms on identical frozen holdouts only if the superseding freeze authorizes execution.
- Depends on: PGIR-205
- Resource class: gpu-large
- Expected inputs: RESULT(PGIR-205), admitted architectures, losses, pairs, proof loop, evaluator, security policy
- Expected outputs: tracked arm checkpoints, actual metrics and uncertainty, costs, failures, receipts, comparison, and RESULT(PGIR-206); when denied, complete typed not_run evidence
- Allowed effects: successor-v1 isolated training, proof, checkpoint, and evaluation artifacts only
- Prohibited effects: hidden-test tuning, best-test selection, fabricated target attainment, shared checkpoint writes, execution under a no-go freeze
- Acceptance: identical heldouts and declared seeds; every metric family resolved; bounded exhaustion typed; every manifest-referenced JSON tracked and independently replayable
- Required proof or evaluation evidence: training, checkpoint, proof, evaluation, resource, reducer-CAS, and paired-comparison receipts
- Result identity: RESULT(PGIR-206) in successor-v1
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1/
- Validation: python -m pytest -q test/api/test_agent_supervisor_proof_workflow_e2e.py test/api/test_agent_supervisor_scheduler.py
- Bundle: pgir/successor/r1-r6
- Parallel lane: experiment-orchestrator
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1/
- Submodules: ipfs_datasets_py
- Exclusive group: campaign-reducer-successor-v1
- Conflict policy: successor reducer independent of trainers

## PGIR-207 Re-qualify, publish or reject, and issue the following board

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G110
- Goal id: PGIR-G110
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: final-decision-report-v2
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/**, docs/architecture/proof_grounded_ir_learning/successor-v1/**
- Base source revisions: current successor branch plus every accepted successor result
- Objective: Apply all sixteen final criteria and thirty-two report sections to RESULT(PGIR-206), then promote, reject, no-go, or declare resource exhaustion and issue the following board.
- Depends on: PGIR-072, PGIR-090, PGIR-100, PGIR-206
- Resource class: cpu-medium
- Expected inputs: accepted successor results, experiment comparison, current promotion and publication authorities
- Expected outputs: tracked successor-v1 decision, manifest, promotion and publication receipts, final report, and next board
- Allowed effects: successor-v1 qualification artifacts; promotion or publication only under independent current authority
- Prohibited effects: universal-understanding claims, failure suppression, model self-promotion, unauthorized upload, historical report mutation
- Acceptance: all criteria and report sections resolve with evidence or explicit no-go; the exact qualified claim is emitted only if every required gate passes; a clean recursive checkout replays the result
- Required proof or evaluation evidence: manifest, evaluation, proof, promotion, publication, and complete result-graph verification
- Result identity: RESULT(PGIR-207) in successor-v1
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/, docs/architecture/proof_grounded_ir_learning/successor-v1/
- Validation: python -m pytest -q test/api/test_agent_supervisor_goal_completion.py test/api/test_agent_supervisor_proof_goal_completion.py
- Bundle: pgir/successor/qualification
- Parallel lane: final-qualifier
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/, docs/architecture/proof_grounded_ir_learning/successor-v1/
- Submodules: ipfs_datasets_py
- Exclusive group: final-decision-successor-v1
- Conflict policy: one independent successor qualification and promotion authority

## PGIR-208 Seal and adjudicate the PGIR-200 through PGIR-202 source chain

- Status: todo
- Completion: supervisor-evidence
- Is schedulable: true
- Priority: P0
- Track: qualification
- Parent goal: PGIR-G030
- Goal id: PGIR-G030
- Board namespace: proof-grounded-ir-learning-successor-v1
- Subgoal: source-chain-acceptance
- Owning repository: ipfs_accelerate_py
- Owned paths: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance/**, scripts/verify_proof_grounded_ir_learning_successor_source_chain.py
- Base source revisions: current successor branch plus RESULT(PGIR-200), RESULT(PGIR-201), RESULT(PGIR-202); accelerator implementations 511ae84626a38dc43ed2851ca4a16c67ff1ac4ca, b189f26e316d1bf6f7760bcfcef3e1d705011a6d, f38fffb9b96ac06d7894ece8eb6030dc2b35bb83; datasets implementations 0566a833e795b0f0596251c2e7e8ca7d8ec27836, 8cc72c77736d3ff2db7cc2530e619bf09b5be027, 8736a0023d5d3afe4d0e5b044a3e4480966a8bf7
- Objective: Emit one tracked outer acceptance and adjudication seal plus a portable verifier for the immutable rights, corpus, and split results without changing any nested source-chain artifact.
- Depends on: PGIR-200, PGIR-201, PGIR-202
- Resource class: network
- Expected inputs: all RESULT(PGIR-200) through RESULT(PGIR-202) payloads and Git objects, the historical source inventory/corpus/lineage/split roots, exact 21 cited source revisions, and supervisor merge/validation receipts as non-authoritative adjudication inputs
- Expected outputs: a canonical source-chain acceptance receipt and portable verifier that close every referenced payload under byte identity, content CID, result identity, replay command, and recursive repository-forest identity
- Allowed effects: successor-v1 outer freeze evidence and its verifier only
- Prohibited effects: mutation of corpora/successor-v1 or splits/successor-v1, silent un-quarantine, hidden-test access, treating ignored runtime state as completion authority, claiming offline replay when network responses are not retained, publication, training, promotion
- Acceptance: all fourteen immutable PGIR-200 through PGIR-202 JSON payloads and their historical inputs are closed under exact byte hashes and CIDs; 21/21 citations, 7,173 candidate rows, zero admitted/materialized rows, all thirteen named permanent holdout no-gos, leakage, and the unchanged public hidden-test commitment replay; implementation, merge, completion, nested gitlink, commit, and tree identities form a recursive forest; the prior submodule compare-and-swap races are explicitly adjudicated; a clean recursive checkout either replays successfully or emits a typed unpublished-ref portability blocker that keeps PGIR-205 fail-closed
- Required proof or evaluation evidence: canonical receipt CID, closed artifact population, recursive repository-forest manifest, exact network replay timestamp and response hashes, portable verifier source identity, fresh recursive checkout receipt, and 34 focused source/corpus/split tests
- Result identity: RESULT(PGIR-208) in successor-v1
- Outputs: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance/, scripts/verify_proof_grounded_ir_learning_successor_source_chain.py
- Validation: python scripts/verify_proof_grounded_ir_learning_successor_source_chain.py --network && python -m pytest -q ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py ipfs_datasets_py/tests/unit/logic/ir_learning/source/test_corpus_build.py ipfs_datasets_py/tests/unit/logic/ir_learning/source/test_successor_rights.py ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py
- Bundle: pgir/successor/source-chain-acceptance
- Parallel lane: source-chain-acceptance
- Predicted files: data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance/, scripts/verify_proof_grounded_ir_learning_successor_source_chain.py
- Submodules: ipfs_datasets_py
- Exclusive group: source-chain-acceptance-successor-v1
- Conflict policy: immutable nested inputs and one exclusive outer acceptance writer
