# CVEfixes Security IR Objective Heap

Ultimate goal: convert a pinned CVEfixes snapshot into a reviewed,
content-addressed Security IR GraphRAG dataset and make the agent supervisor
fail closed when Intent IR or generated code conflicts with its authoritative
security policies.

Program invariants:

- Source code and text are inert untrusted data.
- Retrieval, similarity, model output, formalization, and tests do not grant
  execution authority.
- Derived candidates remain distinct from reviewed and authoritative policy.
- Deny overrides allow; unknown/conflict/stale rejects in enforcement mode.
- Every projection binds the pinned source revision and parent content IDs.
- Existing Security IR, result, proof, and execution-permit contracts are
  extended rather than duplicated.
- Public release excludes unrestricted full third-party bodies by default.
- Protected operator documents are the plan, this heap, and the task board.

## CVESIR-G000 Pinned CVE security reasoning and enforcement loop

- Status: active
- Parent:
- Depends on: CVESIR-G095, CVESIR-G180
- Fib priority: 89
- Priority: P0
- Track: integration
- Bundle: cvefixes-security-ir/integration
- Goal: Complete the pinned source-to-Hub-to-Security-IR-to-supervisor loop with current publication and enforcement receipts.
- Evidence: docs/architecture/cvefixes_security_ir_completion_receipt.json
- Outputs: docs/architecture/cvefixes_security_ir_completion_receipt.json
- Validation: python -m pytest test/api/test_agent_supervisor_cve_security_e2e.py -q
- Acceptance: Receipt binds the source revision, derived Hub revision, release root, Security IR root, current tree, evaluation summary, and passing end-to-end gates; no candidate result is represented as authority.
- Conflict policy: Final integration receipt only after every child goal and external publication receipt is current.
- Goal completion schema version: 1
- Completion confidence: 0.083333
- Uncovered criteria: ["Receipt binds the source revision, derived Hub revision, release root, Security IR root, current tree, evaluation summary, and passing end-to-end gates","no candidate result is represented as authority."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G010 Pinned CVEfixes source snapshot and safe row adapter

- Status: active
- Parent: CVESIR-G000
- Depends on:
- Fib priority: 55
- Priority: P0
- Track: source
- Bundle: cvefixes-security-ir/source
- Goal: Implement a read-only source profile pinned to hitoshura25/cvefixes revision d4f5c4ea65329d9ccbb8a3b3149e5d06eda5edb2, three Parquet shards, 12,987 rows, and the reviewed 23-column contract.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/source_snapshot.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_source_snapshot.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/source_snapshot.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_source_snapshot.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_source_snapshot.py -q
- Acceptance: Pin and shard hashes are verified; rows are bounded and inert; serialized descriptions are parsed without eval; malformed and drifted inputs fail closed; tokens never enter artifacts.
- Conflict policy: Own source snapshot and its tests only; do not implement graph or policy projection.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Pin and shard hashes are verified","rows are bounded and inert","serialized descriptions are parsed without eval","malformed and drifted inputs fail closed","tokens never enter artifacts."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G020 Canonical derived dataset schemas and identities

- Status: active
- Parent: CVESIR-G000
- Depends on:
- Fib priority: 55
- Priority: P0
- Track: schema
- Bundle: cvefixes-security-ir/schema
- Goal: Define immutable schemas and canonical IDs for source records, code units, graph nodes/edges, policy candidates, formal views, evaluations, and release manifests.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py -q
- Acceptance: Canonical round-trip and CID stability pass; parent/source/config identities are mandatory; NaN, unknown fields, duplicate IDs, and authority broadening fail closed.
- Conflict policy: Own new schema module and tests; reuse ir_core canonical identities.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Canonical round-trip and CID stability pass","parent/source/config identities are mandatory","NaN, unknown fields, duplicate IDs, and authority broadening fail closed."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G030 Licensing, privacy, poisoning, and release policy

- Status: active
- Parent: CVESIR-G000
- Depends on:
- Fib priority: 34
- Priority: P0
- Track: governance
- Bundle: cvefixes-security-ir/source
- Goal: Define public/internal body profiles, license provenance, PII/secret scanning, prompt-injection treatment, redaction receipts, and publication admission.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py -q
- Acceptance: Default public profile excludes unrestricted full bodies; provenance is retained; source instructions remain inert; detected secrets, unsafe paths, unreviewed licenses, and policy drift block release.
- Conflict policy: Own release policy only; do not upload or change shared Security IR authority.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Default public profile excludes unrestricted full bodies","provenance is retained","source instructions remain inert","detected secrets, unsafe paths, unreviewed licenses, and policy drift block release."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G040 Vulnerable/fixed semantic projector

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G010, CVESIR-G020, CVESIR-G030
- Fib priority: 34
- Priority: P0
- Track: projection
- Bundle: cvefixes-security-ir/graphrag
- Goal: Project rows to file/hunk/symbol code units, paired vulnerable/fixed facts, preconditions, actions, effects, mitigations, and explicit unsupported diagnostics.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/projector.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_projector.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/projector.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_projector.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_projector.py -q
- Acceptance: Pairing and provenance are loss-aware; deterministic facts are distinct from model candidates; fixed code is negative evidence; unsupported languages and ambiguous hunks are retained rather than fabricated or dropped.
- Conflict policy: Own projector and language adapter registry; no graph index or SecurityIR conversion.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Pairing and provenance are loss-aware","deterministic facts are distinct from model candidates","fixed code is negative evidence","unsupported languages and ambiguous hunks are retained rather than fabricated or dropped."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G050 Typed GraphRAG graph builder

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G040
- Fib priority: 21
- Priority: P0
- Track: graph
- Bundle: cvefixes-security-ir/graphrag
- Goal: Build the reviewed node/edge ontology, integrity-bound graph tables, adjacency indexes, and deterministic graph root.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py -q
- Acceptance: Node and edge types/directions validate; all edges bind sources and endpoints; similarity edges are marked non-authoritative; graph rebuild is deterministic and detects tampering.
- Conflict policy: Own graph materialization; reuse shared GraphRAG primitives where compatible.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Node and edge types/directions validate","all edges bind sources and endpoints","similarity edges are marked non-authoritative","graph rebuild is deterministic and detects tampering."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G060 Bounded lexical, vector, and graph retrieval

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G050
- Fib priority: 13
- Priority: P1
- Track: retrieval
- Bundle: cvefixes-security-ir/graphrag
- Goal: Add bounded hybrid retrieval over CWE, language, code facts, actions, effects, policies, and graph neighborhoods with partition and authority filters.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py -q
- Acceptance: Queries cap shards/nodes/results; filters cannot broaden authority; indexes bind model/config/graph roots; tampering and split crossing fail closed; retrieval never returns a grant.
- Conflict policy: Own CVEfixes retrieval; embeddings route through existing accelerator ports.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Queries cap shards/nodes/results","filters cannot broaden authority","indexes bind model/config/graph roots","tampering and split crossing fail closed","retrieval never returns a grant."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G070 Leakage-safe evaluation and promotion gates

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G040, CVESIR-G050, CVESIR-G060, CVESIR-G110, CVESIR-G120
- Fib priority: 21
- Priority: P0
- Track: evaluation
- Bundle: cvefixes-security-ir/release
- Goal: Implement repository/CVE/body-isolated splits, vulnerable-positive/fixed-negative metrics, calibration, adversarial injection tests, and explicit review/promotion decisions.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py -q
- Acceptance: Leakage checks cover repo, commit, body hash, and near duplicates; metrics are stratified; fixed negatives cannot inherit vulnerable labels; thresholds are measured; failed gates cannot promote candidates.
- Conflict policy: Own evaluation and split policy; do not rewrite source, graph, or Security IR.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Leakage checks cover repo, commit, body hash, and near duplicates","metrics are stratified","fixed negatives cannot inherit vulnerable labels","thresholds are measured","failed gates cannot promote candidates."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G080 Reproducible Hugging Face release builder

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G070
- Fib priority: 13
- Priority: P0
- Track: release
- Bundle: cvefixes-security-ir/release
- Goal: Build deterministic Parquet configs, dataset card, manifest, evaluation report, and bounded query client for sofiyapervane/cvefixes-security-ir-graphrag.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py -q
- Acceptance: Bounded shards and schemas validate; card documents source/license/limitations; release root is stable; validate-only requires no credentials; no secrets/caches/internal bodies enter staging.
- Conflict policy: Own local release packaging; do not perform the Hub mutation.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Bounded shards and schemas validate","card documents source/license/limitations","release root is stable","validate-only requires no credentials","no secrets/caches/internal bodies enter staging."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G090 Idempotent publish and remote verification command

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G080
- Fib priority: 8
- Priority: P1
- Track: publication-tooling
- Bundle: cvefixes-security-ir/release
- Goal: Provide an authenticated, secret-safe, idempotent upload command and remote Dataset Viewer verification that emits a proposed publication receipt.
- Evidence: ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py
- Outputs: ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py -q
- Acceptance: Dry-run is default; target/source/release tuple is idempotent; credentials never persist; remote revision and shard/schema verification are mandatory before a receipt is proposed.
- Conflict policy: Own publication command and tests; tests must not mutate the Hub.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Dry-run is default","target/source/release tuple is idempotent","credentials never persist","remote revision and shard/schema verification are mandatory before a receipt is proposed."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G095 Publish and pin the derived Hugging Face dataset

- Status: provisionally_complete
- Parent: CVESIR-G000
- Depends on: CVESIR-G090
- Fib priority: 5
- Priority: P0
- Track: external-publication
- Bundle: cvefixes-security-ir/external
- Goal: Upload the validated release to sofiyapervane/cvefixes-security-ir-graphrag, resolve the immutable Hub commit, and verify it through the Dataset Viewer.
- Evidence: data/agent_supervisor/cvefixes_security_ir/external/hf_publication_receipt.json
- Outputs: data/agent_supervisor/cvefixes_security_ir/external/hf_publication_receipt.json
- Validation: python ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py --verify-receipt data/agent_supervisor/cvefixes_security_ir/external/hf_publication_receipt.json
- Acceptance: Receipt binds authenticated principal, target repo, source revision, derived release root, Hub commit, remote manifest/shards/schema, timestamp, and verification result without containing credentials.
- Completion authority: external
- External authority blockers: authenticated Hugging Face upload and remote commit resolution
- Conflict policy: Operator/external action only after G090; implementation workers must not fabricate or self-approve the receipt.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["data/agent_supervisor/cvefixes_security_ir/external/hf_publication_receipt.json"]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []
- External completion receipt CIDs: []
- External completion validation: [{"evidence_term":"data/agent_supervisor/cvefixes_security_ir/external/hf_publication_receipt.json","goal_id":"CVESIR-G095","reason_codes":["external_authority_not_supplied"],"receipt_cid":"","requirement_cid":"","schema":"ipfs_accelerate_py.agent_supervisor.external-completion-validation.v1","valid":false}]
- State transitioned at: 2026-07-29T01:57:34.774348+00:00
- State transition reason: Produce completion evidence for: data/agent_supervisor/cvefixes_security_ir/external/hf_publication_receipt.json; Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree.; Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one.; Require an explicitly healthy analyzer that is safe for completion reasoning.; Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.; Task completion is provisional until every criterion has valid evidence.
- Completion evidence records: []
- Provisional at: 2026-07-29T01:57:34.774348+00:00

## CVESIR-G100 CVE/CWE Security IR vocabulary

- Status: active
- Parent: CVESIR-G000
- Depends on:
- Fib priority: 34
- Priority: P0
- Track: security-vocabulary
- Bundle: cvefixes-security-ir/security-ir
- Goal: Define versioned CVE/CWE action, precondition, effect, mitigation, language, and scope vocabulary compatible with Security IR policy attributes.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/vocabulary.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_vocabulary.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/vocabulary.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_vocabulary.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_vocabulary.py -q
- Acceptance: Terms are typed/versioned/canonical; unknown and wildcard-broadened terms fail closed; CWE is classification rather than sufficient policy authority; aliases cannot merge distinct scopes.
- Conflict policy: Own CVEfixes vocabulary; do not change shared exchange/xaman vocabularies.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Terms are typed/versioned/canonical","unknown and wildcard-broadened terms fail closed","CWE is classification rather than sufficient policy authority","aliases cannot merge distinct scopes."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G110 CVE candidates to canonical Security IR adapter

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G020, CVESIR-G040, CVESIR-G100
- Fib priority: 34
- Priority: P0
- Track: security-adapter
- Bundle: cvefixes-security-ir/security-ir
- Goal: Convert grounded candidate/reviewed records into canonical SecuritySource, Resource, Policy(DENY), ThreatAssumption, SecurityClaim, and optional StateMachine declarations without importing results into declarations.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py -q
- Acceptance: Mapping is loss-aware and round-trippable; source and review state are mandatory; candidates cannot claim authoritative result state; wildcard/generalized scopes require explicit review.
- Conflict policy: Extend canonical Security IR and ir_core contracts; no parallel authority/schema.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Mapping is loss-aware and round-trippable","source and review state are mandatory","candidates cannot claim authoritative result state","wildcard/generalized scopes require explicit review."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G120 Forbidden-logic formalization views

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G110
- Fib priority: 21
- Priority: P0
- Track: formalization
- Bundle: cvefixes-security-ir/security-ir
- Goal: Produce typed deontic prohibitions, threat premises, transition views, claims, and proof obligations through the existing Security IR formalization adapter.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py -q
- Acceptance: Typed symbols bind exact scope; deny maps to prohibition; unsupported semantics emit diagnostics; formulas and obligations are non-authoritative; vulnerable/fixed controls demonstrate polarity.
- Conflict policy: Extend formalization_adapter views via a domain adapter; do not create another prover.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Typed symbols bind exact scope","deny maps to prohibition","unsupported semantics emit diagnostics","formulas and obligations are non-authoritative","vulnerable/fixed controls demonstrate polarity."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G130 Pinned Hugging Face Security IR source adapter

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G080, CVESIR-G110
- Fib priority: 13
- Priority: P0
- Track: hf-integration
- Bundle: cvefixes-security-ir/security-ir
- Goal: Load the derived dataset by exact repo and revision, verify its manifest/shards/row identities, and expose bounded canonical Security IR declarations and policy lookup.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py -q
- Acceptance: Floating revisions, manifest drift, missing shards, row tampering, unknown schema, and candidate-as-authority fail closed; offline cache preserves revision identity.
- Conflict policy: Own CVEfixes HF adapter and package exports; reuse existing huggingface snapshot contracts.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Floating revisions, manifest drift, missing shards, row tampering, unknown schema, and candidate-as-authority fail closed","offline cache preserves revision identity."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G140 Generated-code security fact contract and extractors

- Status: active
- Parent: CVESIR-G000
- Depends on:
- Fib priority: 55
- Priority: P0
- Track: code-facts
- Bundle: cvefixes-security-ir/supervisor
- Goal: Define canonical code-security facts and deterministic changed-diff extractors for actions, targets, data flow, effects, capabilities, guards, language, and source scope.
- Evidence: ipfs_accelerate_py/agent_supervisor/code_security_facts.py, test/api/test_agent_supervisor_code_security_facts.py
- Outputs: ipfs_accelerate_py/agent_supervisor/code_security_facts.py, test/api/test_agent_supervisor_code_security_facts.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_security_facts.py -q
- Acceptance: Facts bind tree/blob/diff/AST identities; only changed scope is attributed; unsupported/ambiguous extraction is explicit; source strings cannot inject facts; extractors never grant authority.
- Conflict policy: Own new fact module and tests; do not edit decision runtime yet.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Facts bind tree/blob/diff/AST identities","only changed scope is attributed","unsupported/ambiguous extraction is explicit","source strings cannot inject facts","extractors never grant authority."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G150 Intent/code facts to exact SecurityAuthorizationRequest

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G120, CVESIR-G130, CVESIR-G140
- Fib priority: 34
- Priority: P0
- Track: comparison
- Bundle: cvefixes-security-ir/supervisor
- Goal: Map pinned Intent IR and generated code facts independently to exact security requests and correlate undeclared, broadened, or contradictory effects.
- Evidence: ipfs_accelerate_py/agent_supervisor/cve_security_gate.py, test/api/test_agent_supervisor_cve_security_gate.py
- Outputs: ipfs_accelerate_py/agent_supervisor/cve_security_gate.py, test/api/test_agent_supervisor_cve_security_gate.py
- Validation: python -m pytest test/api/test_agent_supervisor_cve_security_gate.py -q
- Acceptance: Every request binds principal/action/tool/target/data_flow/effect/state/authority and Security IR root; intent pass cannot mask code fail; ambiguous mappings are unknown; exact matching follows existing adapter contracts.
- Conflict policy: Own new gate module; call existing intent/security adapters rather than duplicate them.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Every request binds principal/action/tool/target/data_flow/effect/state/authority and Security IR root","intent pass cannot mask code fail","ambiguous mappings are unknown","exact matching follows existing adapter contracts."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G160 Decision-runtime, permit, and merge enforcement

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G150
- Fib priority: 34
- Priority: P0
- Track: enforcement
- Bundle: cvefixes-security-ir/enforcement
- Goal: Wire the CVE security gate into plan admission, pre-execution permits, post-generation validation, merge admission, and merged-tree revalidation.
- Evidence: test/api/test_agent_supervisor_cve_security_enforcement.py
- Outputs: ipfs_accelerate_py/agent_supervisor/security_constraint_adapter.py, ipfs_accelerate_py/agent_supervisor/ir_constraint_compiler.py, ipfs_accelerate_py/agent_supervisor/execution_permit.py, test/api/test_agent_supervisor_cve_security_enforcement.py
- Validation: python -m pytest test/api/test_agent_supervisor_security_constraints.py test/api/test_agent_supervisor_ir_constraint_compiler.py test/api/test_agent_supervisor_cve_security_enforcement.py -q
- Acceptance: Deny/conflict/unknown/stale reject; allow requires existing authority; generated undeclared effects reject; root or tree drift invalidates receipts; no permit or merge path bypasses the gate.
- Conflict policy: Sole task allowed to edit the three shared enforcement files; preserve backward-compatible existing decisions and deny-overrides.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Deny/conflict/unknown/stale reject","allow requires existing authority","generated undeclared effects reject","root or tree drift invalidates receipts","no permit or merge path bypasses the gate."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G170 Bounded security decision receipts and observability

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G150, CVESIR-G160
- Fib priority: 13
- Priority: P1
- Track: receipts
- Bundle: cvefixes-security-ir/enforcement
- Goal: Emit explainable bounded receipts linking intent/code facts, matching policies, CVE/CWE/source CIDs, roots, reason codes, counterexamples, and enforcement stage.
- Evidence: ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py, test/api/test_agent_supervisor_cve_security_receipts.py
- Outputs: ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py, test/api/test_agent_supervisor_cve_security_receipts.py
- Validation: python -m pytest test/api/test_agent_supervisor_cve_security_receipts.py -q
- Acceptance: Receipts are canonical, bounded, redacted, stage/tree/root-bound, and distinguish evidence from authority; cache keys invalidate on every declared dependency.
- Conflict policy: Own receipt module and tests; extend event fields without logging code bodies or secrets.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Receipts are canonical, bounded, redacted, stage/tree/root-bound, and distinguish evidence from authority","cache keys invalidate on every declared dependency."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## CVESIR-G180 End-to-end conformance, rollback, and operator rollout

- Status: active
- Parent: CVESIR-G000
- Depends on: CVESIR-G070, CVESIR-G090, CVESIR-G130, CVESIR-G160, CVESIR-G170
- Fib priority: 21
- Priority: P0
- Track: rollout
- Bundle: cvefixes-security-ir/integration
- Goal: Prove end-to-end vulnerable/fixed, intent-only, code-only, deny, allow, unknown, conflict, stale, injection, rollback, and pinned-release behavior and document operations.
- Evidence: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Outputs: test/api/test_agent_supervisor_cve_security_e2e.py, docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md
- Validation: python -m pytest test/api/test_agent_supervisor_cve_security_e2e.py -q
- Acceptance: Hermetic fixture passes all cases; live-Hub smoke is opt-in; shadow/assist/enforce/rollback modes are documented; rollback pins prior Security IR root without weakening checks.
- Conflict policy: Own E2E fixture and operator guide; do not fabricate external publication success.
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Hermetic fixture passes all cases","live-Hub smoke is opt-in","shadow/assist/enforce/rollback modes are documented","rollback pins prior Security IR root without weakening checks."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []
