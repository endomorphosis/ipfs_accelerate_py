# Codebase-proof program

Proof-carrying control for code change: property catalog → claims → obligations
→ trust-aware prove/cache → queries → obligation-first context → edit packets →
re-proof and policy gates.

## Machine boards (do not rename task headers)

- Objectives: [`../../agent_supervisor_codebase_proof.objectives.md`](../../agent_supervisor_codebase_proof.objectives.md)
- Taskboard: [`../../agent_supervisor_codebase_proof.todo.md`](../../agent_supervisor_codebase_proof.todo.md)
- Context plan: [`../../AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md`](../../AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md)
- SRT bridge notes: [`../../agent_supervisor_codebase_proof_srt_bridge.md`](../../agent_supervisor_codebase_proof_srt_bridge.md)
- ZK policy: [`../../agent_supervisor_codebase_proof_zk_policy.md`](../../agent_supervisor_codebase_proof_zk_policy.md)

## Code homes (semantic)

Prefer `proof/`, `context/`, and `planning/` domain packages (see
[package map](../../PACKAGE_MAP.md)). Implementation modules include
`code_property_catalog`, `code_claim_contracts`, `code_proof_*`,
`code_edit_*`, `proof_attestation`.
