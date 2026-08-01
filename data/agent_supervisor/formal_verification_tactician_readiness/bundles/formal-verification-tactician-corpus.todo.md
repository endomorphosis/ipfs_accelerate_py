# Objective Bundle: formal-verification-tactician/corpus

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-004 Close formal verification tactician readiness gap: Establish the goal/proof-gap/counterexample golden corpus

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: tactician-corpus
- Depends on: FVT-001
- Outputs: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Validation: python -m pytest test/api/test_formal_verification_tactician_corpus_contract.py -q
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-30-fvt-004-objective-gap-270347a3f9bc.md
- Bundle: formal-verification-tactician/corpus
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-corpus.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 3
- Parallel lane: formal-verification-tactician/corpus
- Conflict policy: Own new corpus contracts and fixtures; do not tune production behavior to fixture strings or label injected expected results as live verification.
- Predicted files: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Changed paths:
- AST symbols: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Interfaces: ProofTacticianCorpus@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G020
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/ef44886470eb200892e9b6f998a1f2735e35b58812ffd7c7f8f00cb9627157a0
- Canonical task CID: baguqeera55ciqzdq5mqarexjw34zripsonpdlnmicl75pr7y6aglsytrk6qa
- Semantic identity: objective-evidence-obligation/v1/b880d3a0fb92f7c9ed7431ccb9ccf4d69cedfd2a6c47e2267727b54217d89e68
- Acceptance subset: The corpus covers missing loop invariant, callee contract/frame, lease safety, fairness ambiguity, impossible target/core, SMT model, runtime trace, protocol attack, hypertrace, kernel rejection, bridge lemma, and legal evidence routing, fixtures bind licenses/provenance and expected authority without embedding private witnesses.
- Preconditions: objective goal FVT-G020 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, satisfy evidence requirement: test/api/test_formal_verification_tactician_corpus_contract.py
- Evidence subset: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/FVT-G020
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/b880d3a0fb92f7c9ed7431ccb9ccf4d69cedfd2a6c47e2267727b54217d89e68
- Missing evidence: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Embedding query: Define solvable, mutated, impossible, ambiguous, unsupported, and unavailable cases that measure end-goal formalization, proof-gap recovery, proof-chain authority, counterexample replay/minimization, and honest failure.
- AST query: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Surplus group: objective/FVT-G020
- Merge key: 1ddc4248c45a4149
- Merge family: objective/FVT-G020
- Merge role: aggregate
- Work item count: 2
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: a13598b9cf18aa46
- Acceptance: Objective scan filed this gap for FVT-G020. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-30-fvt-004-objective-gap-270347a3f9bc.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
