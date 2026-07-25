# ASI-085 Unified Control Completion-Gate Evidence Map

- Date: 2026-07-25
- Task: ASI-085
- Goal: ASI-G070 — Unified Python, CLI, and MCP supervisor control
- Parent: ASI-G000
- Child goals: ASI-G103, ASI-G104, ASI-G105
- Requirements: `031486194157679117987393491870400400279`,
  `184125100306462690646212311073240043804`,
  `186773143401179107362964063059661378722`
- Source gap fingerprint: `d45c96ecbac5d3e37f6a636b340916a408163c07`
- Evidence obligation:
  `objective-work/v1/c52d21e406da75742aa64ccac6f570d34b854c50`
- Todo vector: `c52d21e406da7574`
- Merge family: ASI-G070
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- Lifecycle: provisionally complete and supervisor-actionable

## Gap disposition

ASI-085 closes the parent-gate implementation and audit gap without treating
this document, transport metadata, or any one operational witness as
completion authority:

1. Producing-task closure is exact.
   `UNIFIED_CONTROL_PRODUCING_TASK_IDS` fixes ASI-002, ASI-018, ASI-019,
   ASI-020, and ASI-021 as the unified-control producer population, and
   `evaluate_unified_control_completion` requires each task exactly once in a
   successful terminal state in addition to the caller's completion
   assertion. Missing, duplicate, foreign, or incomplete producers fail
   closed.
2. Descendant closure is exact and recursive.
   `UNIFIED_CONTROL_CHILD_GOAL_IDS` fixes ASI-G103, ASI-G104, and ASI-G105;
   each must be `verified_complete` with a fresh passing gate bound to the
   current repository tree. Every descendant proof requirement remains
   freshly proved, conclusive, uncontradicted, and satisfied at its required
   assurance. A parity witness cannot replace mutation or discovery proof,
   and no child can directly complete its parent.
3. `UNIFIED_CONTROL_ACCEPTANCE_CRITERIA` fixes the parent acceptance
   population to the five literal ASI-G070 clauses. Callers cannot narrow or
   replace it. Every submitted validation participates in the decision; one
   failed, stale, malformed, contradictory, or foreign-bound sibling
   invalidates the submission even if another receipt for the same criterion
   passes.
4. Coverage is implementation- and receipt-bound. There is exactly one
   coverage row per literal criterion on the current tree. Each row names a
   concrete implementation binding and the provenance identity of the
   submitted fresh passing validation for that exact criterion. A summary,
   requirement ID, child status, discovery record, or detached receipt
   identity cannot fill a coverage gap.
5. Analyzer health is a separate authority input. It must explicitly report
   healthy and safe for completion reasoning and bind the repository, tree,
   `UNIFIED_CONTROL_OBJECTIVE_ID`, `UNIFIED_CONTROL_OBJECTIVE_REVISION`,
   `UNIFIED_CONTROL_COMPLETION_ANALYZER_VERSION`, and
   `UNIFIED_CONTROL_COMPLETION_CONFIGURATION_REVISION`. CLI output, MCP
   discovery, provider health, an operational parity record, and this audit
   index cannot substitute.
6. Exhaustion is configured and independent.
   `UNIFIED_CONTROL_REQUIRED_EXHAUSTIVE_RECEIPTS` fixes the count at two. Each
   supplied member must be independently identified, fresh, healthy,
   completion-safe, exhaustive, identically bound to analyzer health, and
   unique by member, evidence-channel, and receipt identity. Caller-supplied
   under-counts or duplicate, stale, unhealthy, unsafe, non-exhaustive, or
   foreign-bound members fail closed.
7. Completion remains two phase. A fully passing active evaluation may move
   only to provisional completion. Verification requires a later separate
   evaluation while every producer, child, validation, mapping, analyzer, and
   exhaustion receipt remains valid. Later invalidation reopens a verified
   parent.

The three existing child goals are the stable minimal partition:
ASI-G103 owns shared operation/schema and cross-surface behavior parity,
ASI-G104 owns mutation authority and exactly-once effects, and ASI-G105 owns
side-effect-free discovery. No additional child goal is needed.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Shared operations have schema and behavior parity across Python, CLI, and MCP | The closed `Operation` vocabulary, canonical `OperationRequest`/`OperationResult` schemas, operation-specific schema population, direct `SupervisorControlService.execute` delegation by `run_agent_cli` and `execute_agent_supervisor_operation`, and `ControlSurfaceParityEvidence` exact three-surface cases | shared-schema, exact Python/CLI/MCP record, direct-dispatch, operation-matrix, schema-drift, and G103 completion-gate tests in `test_agent_supervisor_control_plane.py`, `test_unified_cli_agent_supervisor.py`, and `test_agent_supervisor_tools.py` |
| read operations are bounded | `ControlBounds`, request validation, bounded repository adapters and result construction, and the finite CLI watch loop | bounded-read and stable-error tests in `test_agent_supervisor_control_plane.py`, plus bounded-watch and unbounded-watch rejection tests in `test_unified_cli_agent_supervisor.py` |
| mutations require authorization, explicit roots, dry-run/preview, idempotency, lease/fencing, and audit receipts | Canonical request cross-field validation, the single `SupervisorControlService` preflight boundary, `ControlMutationGuardEvidence`, dry-run proposal-only execution, scoped idempotent replay, live lease/fence enforcement, exact effect binding, and `ControlAuditReceipt` | lifecycle dry-run/applied/replay/fencing tests in `test_agent_supervisor_control_lifecycle.py`, service mutation tests in `test_agent_supervisor_control_plane.py`, and the closed unsafe CLI/MCP rejection matrices |
| lifecycle state and errors are consistent | Typed lifecycle commands pass through the same request/result envelope and service dispatch path; stable `ControlError` codes, audit IDs, lifecycle bindings, and replay records are serialized by `OperationResult.to_record` on every surface | lifecycle command, stable error, parity matrix, CLI canonical-envelope, and MCP exact-result tests across all four ASI-085 validation modules |
| tool discovery has no provider or process-start side effects | Static deterministic `ControlDiscoveryManifest` producers plus `ControlDiscoverySafetyEvidence` runtime observations of provider imports, process starts, service resolution, operation vocabulary, and schema population for Python, CLI, and MCP | deterministic Python/CLI discovery, no-service-resolution registration, observed three-surface discovery safety, strict content-addressed evidence, and G105 completion-gate tests |

The parent completion matrix additionally exercises incomplete producing
tasks; missing, duplicate, foreign, reopened, proofless, stale, or
foreign-tree children; failed, stale, missing, duplicate, detached, or
unmapped criterion receipts; unsafe or mismatched analyzer evidence; and
under-count, duplicate-member, duplicate-receipt, duplicate-channel,
unhealthy, unsafe, non-exhaustive, stale, or foreign-bound quorum members.
`test_g070_parent_completion_requires_closed_current_tree_proof_packet`
constructs the fully bound active-to-provisional packet and then performs the
separate provisional-to-verified evaluation.
`test_g070_parent_completion_rejects_each_narrowed_or_unhealthy_input`
mutates one authority input at a time, including incomplete producers, every
invalid criterion/coverage case, stale or contradicted descendant proof,
unsafe or foreign analyzer bindings, caller-lowered quorum size, and
duplicate, stale, detached, unhealthy, unsafe, or non-exhaustive members.

## Validation observation

The mandatory current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
```

This discovery record is an audit and provenance index, not a completion
receipt. It intentionally claims no final repository-tree identity, analyzer
execution, exhaustion vote, or lifecycle transition. The submitting runner's
fresh passing post-change execution is the validation receipt. ASI-G070 and
ASI-G000 remain supervisor-actionable until all exact producing tasks are
terminal-successful, all three child goals remain verified with fresh
current-tree proof, every parent criterion has a fresh passing mapped
validation, analyzer health is explicitly healthy, completion-safe, and fully
bound, two independent fresh healthy exhaustive receipts pass, and the later
separate provisional-to-verified evaluation succeeds.
