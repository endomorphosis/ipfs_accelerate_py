# LFV-038 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 054dc8ddf20fcd4da86563562f84c2230d24fb6a
Goal id: LFV-G071
Goal title: Expose equivalent CLI and MCP verification operations
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P1
Track: cli-mcp
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 37
Bundle: logic-formal-verification/api
Parallel lane: logic-formal-verification/api
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Add bounded machine-readable CLI and MCP operations for the stable verification API with capability inspection and receipt retrieval.
AST query: ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py, ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py
Conflict policy: Single owner for CLI/MCP registration and tests; reuse the Python facade and preserve existing command/tool names and behavior.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/cli.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_tools/__init__.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/__init__.py, ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py, ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py
Interfaces: LogicVerificationCLI@1, LogicVerificationMCP@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/4282e827d5bb39161bf7b8e6b9b431d06a89b7f8b35118533fc8ddc09d3cfa43
Acceptance subset: CLI/MCP cover list, capability, compile, check, monitor, portfolio, counterexample, receipt, advisor, and attestation operations, schemas match Python, inputs/outputs are bounded, errors and unavailable tools are stable and secret safe.
Preconditions: objective goal LFV-G071 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py, ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py
Dependencies: LFV-G070
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G071
Rejection reasons: none (accepted)

## Goal

Add bounded machine-readable CLI and MCP operations for the stable verification API with capability inspection and receipt retrieval.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py
- ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
