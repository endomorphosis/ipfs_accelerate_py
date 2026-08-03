# Objective Bundle: cvefixes-security-ir/integration

Source todo: docs/architecture/cvefixes_security_ir.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## CVESIR-009 Close objective gap: End-to-end conformance, rollback, and operator rollout

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rollout
- Depends on: CVESIR-007, CVESIR-001, CVESIR-003, CVESIR-015, CVESIR-005
- Outputs: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Validation: python -m pytest test/api/test_agent_supervisor_cve_security_e2e.py -q
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/hallucinate_app/ipfs_accelerate_py/data/agent_supervisor/cvefixes_security_ir/discovery/2026-07-29-cvesir-009-objective-gap-65248628dd64.md
- Bundle: cvefixes-security-ir/integration
- Bundle shard: data/agent_supervisor/cvefixes_security_ir/bundles/cvefixes-security-ir-integration.todo.md
- Bundle strategy: explicit
- Graph parents: CVESIR-G000
- Graph depth: 1
- Objective heap index: 8
- Parallel lane: cvefixes-security-ir/integration
- Conflict policy: Own E2E fixture and operator guide; do not fabricate external publication success.
- Predicted files: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Changed paths:
- AST symbols: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Interfaces:
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: CVESIR-G180
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/101841b4985d9d646cec8fe84d071485e002ead766fbdc59a8fd285215c3ef7b
- Canonical task CID: baguqeeracamedneylwowi3hmr7ue2byuqxqaf2wxm355ywni7uufefod555q
- Semantic identity: objective-evidence-obligation/v1/2eecabc338e941699129201a46e15d961f03ea7ea8363ec057bde69ff62f556a
- Acceptance subset: Hermetic fixture passes all cases, live-Hub smoke is opt-in, shadow/assist/enforce/rollback modes are documented, rollback pins prior Security IR root without weakening checks.
- Preconditions: objective goal CVESIR-G180 is schedulable
- Effects: satisfy evidence requirement: test/api/test_agent_supervisor_cve_security_e2e.py, satisfy evidence requirement: docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Evidence subset: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Resource class: cpu-medium
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-medium
- Merge fate: objective/CVESIR-G180
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/2eecabc338e941699129201a46e15d961f03ea7ea8363ec057bde69ff62f556a
- Missing evidence: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Embedding query: Prove end-to-end vulnerable/fixed, intent-only, code-only, deny, allow, unknown, conflict, stale, injection, rollback, and pinned-release behavior and document operations.
- AST query: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Surplus group: objective/CVESIR-G180
- Merge key: a71d41454c32dd35
- Merge family: objective/CVESIR-G180
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
- Todo vector key: 1c68e021a41dffce
- Acceptance: Objective scan filed this gap for CVESIR-G180. Use evidence in /home/barberb/lift_coding/hallucinate_app/ipfs_accelerate_py/data/agent_supervisor/cvefixes_security_ir/discovery/2026-07-29-cvesir-009-objective-gap-65248628dd64.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
