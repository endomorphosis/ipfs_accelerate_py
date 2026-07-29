# LFV-008 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 003218cab9fc6b358c7b04e5ab9dbc697407ebae
Goal id: LFV-G015
Goal title: Reconcile datasets and supervisor provider contracts
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P0
Track: provider-contract
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 7
Bundle: logic-formal-verification/foundation
Parallel lane: logic-formal-verification/foundation
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Define a stable wire/provider contract usable by datasets logic and thin supervisor adapters without creating a cyclic import or a fifth provider abstraction.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py, ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py, test/api/test_logic_provider_contract.py
Conflict policy: Own the new provider contract, supervisor facade, and contract test; do not register concrete providers or edit routing policy.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py, ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py, test/api/test_logic_provider_contract.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py, ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py, test/api/test_logic_provider_contract.py
Interfaces: LogicProvider@1, SupervisorLogicProviderFacade@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/934af1dfc3866cfef90902390f94471ec0cd6a3a78966bcbd12602070548cb98
Acceptance subset: Requests/responses round trip canonically across the submodule boundary, provider discovery stays lazy, cancellations and resources are representable, supervisor compatibility is additive, dataset code never imports the parent package.
Preconditions: objective goal LFV-G015 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py, satisfy evidence requirement: ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py, satisfy evidence requirement: test/api/test_logic_provider_contract.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py, ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py, test/api/test_logic_provider_contract.py
Dependencies: LFV-G005, LFV-G012, LFV-G013
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G015
Rejection reasons: none (accepted)

## Goal

Define a stable wire/provider contract usable by datasets logic and thin supervisor adapters without creating a cyclic import or a fifth provider abstraction.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py
- ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py
- test/api/test_logic_provider_contract.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
