# LFV-024 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 2eb0b3e03cb62eac531ac74ed42b09841cc253ce
Goal id: LFV-G047
Goal title: Implement Tamarin and ProVerif protocol backends
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P1
Track: prover
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 23
Bundle: logic-formal-verification/protocols
Parallel lane: logic-formal-verification/protocols
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Generalize reviewed supervisor/domain protocol models into deterministic Tamarin and ProVerif compilers, runners, result parsers, and attack-trace receipts.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py, ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py
Conflict policy: Own the new protocol backend modules/test; port generic logic without modifying reviewed domain fixtures, installers, public API, or supervisor routing.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py, ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py, ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py
Interfaces: TamarinBackend@1, ProVerifBackend@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/b9afb8dc43fe872f348b9077712ff0928baedd7e440dfeb980baf0999bb385b5
Acceptance subset: Compilers disclose the Dolev-Yao/symbolic-model ceiling, equational theory, and claim support, tool versions and Maude/opam dependencies bind receipts, attack traces normalize and replay, disagreement and inconclusive results quarantine, missing tools are explicit.
Preconditions: objective goal LFV-G047 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py, satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py, ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py
Dependencies: LFV-G013, LFV-G014, LFV-G029
Resource class: cpu-proof-solver
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-solver
Merge fate: objective/LFV-G047
Rejection reasons: none (accepted)

## Goal

Generalize reviewed supervisor/domain protocol models into deterministic Tamarin and ProVerif compilers, runners, result parsers, and attack-trace receipts.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py
- ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py
- ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
