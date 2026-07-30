# Objective Bundle: cvefixes-security-ir/schema

Source todo: docs/architecture/cvefixes_security_ir.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## CVESIR-017 Close objective gap: Canonical derived dataset schemas and identities

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: schema
- Depends on:
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py -q
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/hallucinate_app/ipfs_accelerate_py/data/agent_supervisor/cvefixes_security_ir/discovery/2026-07-29-cvesir-017-objective-gap-b8ca84201d54.md
- Bundle: cvefixes-security-ir/schema
- Bundle shard: data/agent_supervisor/cvefixes_security_ir/bundles/cvefixes-security-ir-schema.todo.md
- Bundle strategy: explicit
- Graph parents: CVESIR-G000
- Graph depth: 1
- Objective heap index: 16
- Parallel lane: cvefixes-security-ir/schema
- Conflict policy: Own new schema module and tests; reuse ir_core canonical identities.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Changed paths:
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Interfaces:
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: CVESIR-G020
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/ce0ee91e2ab5f0da7a4f9166263b76533651a347d6df12c015ca9ea6ab410407
- Canonical task CID: baguqeerazyhoshrkwxynu6spsftcmo3wkm3fdi2h23prfqavzkpknk2baqdq
- Semantic identity: objective-evidence-obligation/v1/26678703f9fea694ab6720cc601665b729c139d805ee7a57b82ab90d8fc045e1
- Acceptance subset: Canonical round-trip and CID stability pass, parent/source/config identities are mandatory, NaN, unknown fields, duplicate IDs, and authority broadening fail closed.
- Preconditions: objective goal CVESIR-G020 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Resource class: cpu-medium
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-medium
- Merge fate: objective/CVESIR-G020
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/26678703f9fea694ab6720cc601665b729c139d805ee7a57b82ab90d8fc045e1
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Embedding query: Define immutable schemas and canonical IDs for source records, code units, graph nodes/edges, policy candidates, formal views, evaluations, and release manifests.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Surplus group: objective/CVESIR-G020
- Merge key: 315957ead1ba1d4d
- Merge family: objective/CVESIR-G020
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
- Todo vector key: 644269ce925d0f5c
- Acceptance: Objective scan filed this gap for CVESIR-G020. Use evidence in /home/barberb/lift_coding/hallucinate_app/ipfs_accelerate_py/data/agent_supervisor/cvefixes_security_ir/discovery/2026-07-29-cvesir-017-objective-gap-b8ca84201d54.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
