# LFV-036 Objective Goal Gap

Date: 2026-07-29
Fingerprint: b43e03c61ed072542b6588fccc4589a99a9aea34
Goal id: LFV-G070
Goal title: Expose the stable Python software-verification API
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P1
Track: python-api
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 35
Bundle: logic-formal-verification/api
Parallel lane: logic-formal-verification/api
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Add lightweight generic family/provider discovery, compilation, checking, monitoring, portfolio, counterexample, receipt, advisor, and attestation operations while preserving legacy imports.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_verification_api.py
Conflict policy: Single owner for `logic.api`, package exports, submodule registry, and the new verification facade; do not edit CLI/MCP or supervisor routers.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/api.py, ipfs_datasets_py/ipfs_datasets_py/logic/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/submodule_registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/registry.py, ipfs_datasets_py/tests/unit/logic/test_verification_api.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_verification_api.py
Interfaces: LogicVerificationAPI@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/59ad2b3cc265db9ff2774e31d84e86992287285dc79faff3a47af6c7c1d84932
Acceptance subset: API imports quietly without optional tools, discovery is declarative, responses expose typed status/authority/assumptions/bounds/translations/witnesses/cache provenance, absent features are explicit, legacy API behavior stays green.
Preconditions: objective goal LFV-G070 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/test_verification_api.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_verification_api.py
Dependencies: LFV-G011, LFV-G026, LFV-G027, LFV-G043, LFV-G044, LFV-G045, LFV-G046, LFV-G047, LFV-G048, LFV-G050, LFV-G061, LFV-G062, LFV-G063
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G070
Rejection reasons: none (accepted)

## Goal

Add lightweight generic family/provider discovery, compilation, checking, monitoring, portfolio, counterexample, receipt, advisor, and attestation operations while preserving legacy imports.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py
- ipfs_datasets_py/tests/unit/logic/test_verification_api.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
