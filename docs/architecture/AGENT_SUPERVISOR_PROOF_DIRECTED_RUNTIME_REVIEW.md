# Agent Supervisor Proof-Directed Runtime Review

Date: 2026-07-26

## Outcome

The agent supervisor already contains most of the mechanisms needed for a
content-addressed, proof-carrying runtime, but they are not yet composed on the
live planning and implementation path.

The target system should compile each proposed decision against four pinned
inputs:

1. Intent: the applicable IntentIR action contract, including goals,
   preconditions, postconditions, invariants, guards, failures, and
   verification requirements.
2. Law: the applicable LegalIR norms, exceptions, jurisdiction, effective
   interval, conflicts, and proof obligations.
3. Program behavior: the exact clean or dirty repository snapshot, AST,
   symbols, interfaces, call/effect graph, tests, and proposed effect manifest.
4. Security: the applicable SecurityIR principals, assets, resources, zones,
   channels, policies, state transitions, threat assumptions, and claims.

Those inputs should be compiled into an immutable proof-dependency graph. A
decision context should contain the small invariant decision core plus the
authoritative transitive dependency closure for that decision, not the full
legal corpus, repository, skill corpus, task board, or conversation.

Retrieval is useful for finding candidate nodes. It is not authority. Exact
pinned declarations, formalization artifacts, proof results, and runtime
monitors remain the only authority-bearing inputs.

## Current implementation review

### What is already strong

The supervisor has mature, reusable foundations:

- `context_contracts.py` defines bounded, canonical `ContextBudget`,
  `ContextReference`, `ContextCapsule`, and `ContextDeltaCapsule` contracts.
- `context_compiler.py` provides required-versus-optional selection,
  provider-aware token budgets, stable prefixes, value-of-information ranking,
  content-addressed expansion handles, and delta retries.
- `analysis_pipeline.py`, `analysis_ast_index.py`, and
  `analysis_retrieval.py` provide cached incremental AST analysis and bounded
  multi-signal retrieval.
- `proof_scope_index.py` provides a reverse index from proof inputs to
  obligations and receipts, including transitive dependent lookup.
- `proof_context.py`, `proof_carrying_planner.py`,
  `formal_plan_compiler.py`, and `formal_plan_context.py` provide bounded proof
  contexts and content-addressed planning artifacts.
- `authorization_logic.py` defines canonical principals, grants, policies,
  requests, and decisions.
- `code_evidence_graph.py` and `CodeImpactIndex` represent authoritative code,
  proof, validation, merge, and completion relationships. GraphRAG annotations
  are correctly prohibited from manufacturing proof edges.
- `artifact_store.py`, `runtime_cas.py`, and the cache coordinator provide
  bounded content-addressed storage, dependency metadata, and reuse.
- The optional `ipfs_datasets_py` analysis and logic providers are lazy and
  preserve proposal-only authority.

These pieces make this an integration and enforcement project, not a rewrite.

### The live-path gap

The normal implementation path in
`todo_daemon/implementation_daemon.py` does not use those foundations as one
decision system:

- `_compile_implementation_context` builds the required capsule from task
  title, generic implementation rules, dependencies, expected outputs, and
  acceptance text.
- Its only optional evidence is the rendered todo vector context from
  `_render_todo_vector_context`.
- It does not query `AnalysisASTIndex`, `BoundedGraphRAGRetriever`,
  `CodeImpactIndex`, `ProofScopeIndex`, or the proof-carrying planner while
  compiling the implementation context.
- It does not load or pin IntentIR, LegalIR, or SecurityIR roots.
- `_implementation_prompt_policy_appendix` appends some edit policy after the
  compiled capsule. That text is outside the capsule's content identity,
  budget, required-coverage witness, and retry identity.
- `_implementation_repository_and_tree_ids` binds the Git common directory
  and `HEAD` commit. The implementation actor edits a working tree, so an
  uncommitted or untracked input can affect behavior without changing that
  `tree_id`.
- Retry capsules are usefully delta-based, but invalidation observes the task
  and repository/tree identifiers rather than IR roots and the exact proof
  dependency closure.

The task proposal router has the same structural limitation: it
context-compiles task and plan prose, but it does not compile a decision from
the four authoritative domains.

The result is content-addressed prompt construction, not yet a
content-addressed proof-carrying runtime.

## Reusable `ipfs_datasets_py` machinery

### Shared IR and formalization core

The following should remain the canonical cross-repository foundation:

- `logic/ir_core/identity.py`: canonical CIDv1 identities over canonical
  bytes.
- `logic/ir_core/canonical.py`: declared canonical collection semantics.
- `logic/ir_core/provenance.py`: source, span, producer, and configuration
  bindings.
- `logic/ir_core/claims.py`: assumptions, proof obligations, and IR claims.
- `logic/ir_core/protocols.py`: bounded requests/results and exact result
  authority.
- `logic/formalization/compiler.py` and `logic/formalization/views.py`:
  content-addressed formalization artifacts and typed logical views.

The supervisor should bridge these identities into its artifact store by
verifying that the CID and supervisor digest name the same canonical bytes. It
must not silently recanonicalize a foreign artifact and retain the old
identity.

### IntentIR

`logic/intent_ir/schema.py` is a source-grounded declaration of procedure,
capability, or policy intent. It explicitly does not authorize or execute a
procedure. Its statements and action/control graph are suitable for compiling:

- goal and action selection;
- precondition, guard, and invariant checks;
- expected effect and postcondition manifests;
- failure and retry branches;
- verification obligations; and
- plan conformance.

`logic/intent_ir/formalize/compiler.py` and `obligations.py` provide
deterministic formal views and authority-bound proof packets. The supervisor
should consume those artifacts, not infer an intent contract from retrieved
skill prose at dispatch time.

The SkillCenter BM25, vector, and GraphRAG indexes are appropriate for
discovering candidate skills or premises. Their hits correctly carry
`authority=context_only` and `proof_authority=False`.

### LegalIR

`logic/legal_ir/adapter.py` converts reviewed legal material into the shared
formalization model and deontic, temporal, frame, first-order, and
knowledge-graph views.

LegalIR needs an additional supervisor-facing applicability layer. Selection
must be deterministic over pinned jurisdiction, subject, action, resource, and
effective-time facts. Semantic similarity may nominate a provision, but it
cannot establish applicability, resolve an exception, or silently choose
between conflicting norms.

The output should be applicable obligations, prohibitions, permissions,
exceptions, conflicts, and unresolved legal questions. LegalIR constrains a
decision; it does not directly create an execution grant.

### SecurityIR

`logic/security_ir/model.py` defines immutable, declaration-only security
principals, assets, zones, channels, resources, policies, state machines,
threat assumptions, and claims.

`logic/security_ir/formalization_adapter.py` lowers those declarations and
claims into shared formal views and proof obligations. A new supervisor adapter
must also compile applicable SecurityIR policy into the existing
`authorization_logic.py` request/decision model. It should preserve deny
overrides, unknown/conflict states, state-machine preconditions, and exact
resource/effect scope.

SecurityIR declarations are policy inputs. Only a verified authorization
decision bound to the exact action can become part of an execution permit.

### AST, vector, knowledge graph, and GraphRAG

- The supervisor's incremental `AnalysisASTIndex` is the most direct source of
  code symbols and file relationships for implementation decisions.
- `CodeImpactIndex` should supply the transitive changed-symbol, changed-path,
  obligation, and validation closure.
- `ipfs_datasets_py.knowledge_graphs.query.semantic_traversal` provides useful
  bounded embedding-guided beam traversal.
- The IntentIR retrieval facade provides pinned graph snapshots, partition
  isolation, explicit budgets, and context-only results.
- The SkillCenter release machinery provides CID-keyed rows, BM25 lookup,
  centroid-organized vector shard metadata, and bounded remote graph
  navigation.

These retrieval systems should produce seed candidates and a retrieval receipt
that binds the query, index roots, embedding model/configuration, graph root,
budgets, returned candidates, omissions, and truncation. Mandatory authority
and proof dependencies must then be completed by deterministic graph closure.

## Target architecture

```text
task / tool request / proposed patch
                 |
                 v
        canonical DecisionRequest
                 |
        +--------+---------+----------------+----------------+
        |                  |                |                |
   pinned IntentIR    pinned LegalIR   program/AST root  pinned SecurityIR
        |                  |                |                |
        +------------------+--------+-------+----------------+
                                  |
                     SemanticProofDependencyGraph
                                  |
             retrieval seeds + authoritative dependency closure
                                  |
                    DecisionContextCompiler + witness
                                  |
                  constrained plan / action candidate
                                  |
       intent conformance + legal constraints + security authorization
                                  |
                        exact ExecutionPermit
                                  |
               tool/edit/validation/merge enforcement
                                  |
                  effects, proofs, receipts, invalidation
```

### 1. Canonical decision envelope

Add an immutable `DecisionRequest` containing:

- decision kind and stage;
- principal, lease, fencing epoch, and requested authority;
- task/objective and acceptance identities;
- exact action, tool arguments, target resources, and expected effects;
- repository identity and a working-tree content root that covers tracked,
  modified, staged, deleted, and untracked inputs in scope;
- pinned IntentIR, LegalIR, SecurityIR, AST/program graph, tool catalog, and
  policy roots;
- time, jurisdiction, environment, capability, and model/toolchain facts when
  they affect applicability;
- hard token, byte, graph-hop, retrieval, proof, latency, and expansion
  budgets.

Any field that can change the decision belongs in its content identity.

### 2. Pinned IR registry and adapters

Add a lazy, capability-negotiated registry that:

- accepts exact CIDs/digests and supported schema versions;
- obtains bounded artifacts through local storage or the existing
  `ipfs_datasets_py` analysis transport;
- verifies canonical bytes, CID, schema, producer, provenance, review/trust
  state, and declared authority;
- compiles each IR family to normalized dependency nodes, constraints, and
  proof obligations;
- preserves unsupported, partial, stale, ambiguous, and contradictory states;
- has deterministic local fixtures and fails closed when a required IR or
  adapter is unavailable; and
- performs no eager dataset, model, graph, or prover import during package
  import or capability discovery.

### 3. Unified semantic proof dependency graph

Extend or layer over `CodeEvidenceGraph` with typed nodes for:

- decisions, plans, actions, effects, tools, and resources;
- intent goals, action contracts, preconditions, guards, invariants,
  postconditions, failures, and verifications;
- legal norms, applicability facts, exceptions, conflicts, and effective
  intervals;
- security principals, assets, zones, channels, policies, state transitions,
  threats, and claims;
- repository snapshots, files, AST symbols, interfaces, calls, data flows,
  tests, and validators;
- assumptions, premises, proof obligations, proof attempts, proof receipts,
  runtime monitors, and authorization decisions.

Use explicit edges such as `requires`, `constrained_by`, `applies_to`,
`exception_to`, `conflicts_with`, `authorizes`, `denies`, `implements`,
`affects`, `depends_on`, `proven_by`, `monitored_by`, `invalidates`, and
`sourced_from`.

Every node and edge must carry content identity, provenance, authority class,
trust state, and version. Proposal-only edges can rank or annotate, but cannot
enter an authority-bearing closure.

### 4. Proof-directed retrieval

The retrieval algorithm should:

1. derive exact seeds from the decision envelope;
2. optionally add bounded BM25, vector, AST, and GraphRAG candidates;
3. validate every candidate against the pinned snapshot and partition;
4. walk mandatory typed edges to a fixed-point dependency closure;
5. join active proof and authorization receipts from `ProofScopeIndex`;
6. expose contradictions, missing proofs, uncertain applicability, omitted
   optional evidence, and budget exhaustion;
7. store large bodies as content-addressed expansion handles; and
8. emit a canonical retrieval/closure receipt.

Approximate retrieval is acceptable for optional explanatory evidence. It is
not acceptable for deciding that no law, policy, invariant, denial, or proof
dependency applies.

### 5. Minimal decision context and completeness witness

Add a `DecisionContextCompiler` that uses the existing `ContextCompiler` but
changes its evidence source and required core.

The invariant core should contain:

- exact decision and pinned-root identities;
- the selected IntentIR goal/action contract;
- applicable legal and security constraints, conflicts, and unknowns;
- authorization status;
- required preconditions, invariants, effects, postconditions, and validation;
- proof obligations and current proof/monitor status;
- exact read/write/tool/resource scope; and
- acceptance and failure behavior.

The compiler should inline compact authoritative summaries and use expansion
handles for larger bodies and proofs. It must emit a `ContextCompletenessWitness`
mapping every mandatory dependency node to an included reference or verified
expansion handle, including the paths that made it mandatory.

For a decision `d`, let `C(d)` be its mandatory authoritative proof closure.
The desired invariant is:

```text
context_tokens(d) <= fixed_core + O(size(C(d))) + selected_optional_evidence
```

Corpus size may affect index lookup cost, but it must not directly affect
prompt size. If the mandatory closure cannot fit, the compiler must split the
decision, request a named expansion, or return a typed budget failure. It must
never truncate a mandatory constraint.

### 6. Constraint compilation and execution permits

Compile the selected action or plan through three independent checks:

- Intent conformance: the plan advances the selected goal and respects its
  preconditions, guards, invariants, declared effects, failures, and
  verification contract.
- Legal constraints: all pinned applicable obligations, prohibitions,
  permissions, exceptions, conflicts, and unresolved applicability conditions
  are represented. Unknown or contradictory mandatory law fails closed or
  requires an explicitly authorized review path.
- Security authorization: the exact principal/action/resource/effect request
  is evaluated under the compiled SecurityIR policy and current state.

Only after all required checks and proofs pass should the runtime issue an
immutable `ExecutionPermit` bound to:

- the exact `DecisionRequest`;
- the plan/action and expected effect manifest;
- all IR, repository, policy, capability, and tool roots;
- the exact dependency closure and context witness;
- proof, monitor, legal-applicability, intent-conformance, and authorization
  decisions;
- lease, fencing epoch, expiry, idempotency key, and permitted operation count.

The permit must be checked immediately before every mutation-capable tool
call, file change transaction, task-board mutation, commit, and merge. A
changed argument, target, worktree, IR root, policy, lease, or effect invalidates
it.

### 7. Incremental invalidation and revalidation

After a proposed or observed change:

- update the working-tree root and incremental AST/effect graph;
- compute changed semantic nodes;
- walk reverse dependency edges through the generalized `ProofScopeIndex`;
- invalidate only dependent contexts, plans, permits, caches, proofs, and
  validation receipts;
- retain unaffected content-addressed artifacts;
- recompile the affected plan suffix and decision context;
- rerun the minimum authoritative proof/validation closure; and
- require a merged-tree permit and evidence assembly before completion.

Event checkpoints must bind the same roots so restart cannot revive a permit
or context from a stale IR or dirty worktree.

## Authority and failure rules

The integrated runtime should enforce these rules globally:

1. Intent describes desired behavior; it never grants authority.
2. Retrieval similarity, graph proximity, model confidence, and generated
   formal logic are proposal-tier only.
3. A formalization artifact is a faithful compiled view, not proof that its
   claims are true.
4. LegalIR contributes applicable constraints and obligations; it does not
   bypass SecurityIR authorization.
5. SecurityIR policy compilation does not discharge intent, legal, program,
   or proof obligations.
6. Proof authority must be exact, producer-owned, current-root-bound, and
   valid for the obligation it discharges.
7. Unknown schema, missing root, ambiguous applicability, contradiction,
   budget exhaustion in mandatory closure, stale proof, or changed effect
   fails closed.
8. Runtime monitors may satisfy only obligations whose policy explicitly
   permits monitor authority.
9. Text appended outside a context capsule cannot impose or broaden
   authoritative execution scope.
10. Conversation history is evidence only when explicitly promoted into a
    pinned source artifact with provenance and an allowed authority class.

## Measurement and rollout

The new runtime should begin in shadow mode beside the current implementation
path. A paired benchmark should compare the same decisions using current
context construction and proof-directed contexts.

Required measurements include:

- provider input tokens per accepted criterion and per decision;
- mandatory closure nodes/bytes versus total corpus/repository nodes/bytes;
- context growth while irrelevant corpus, skills, code, and conversation
  history are multiplied;
- exact warm reuse and invalidation precision;
- retrieval seed recall and authoritative closure completeness;
- proof, policy, and intent omission rate;
- stale context, stale permit, and unauthorized mutation escapes;
- first-valid-plan rate, retry tokens, changed-file precision, and validation
  cost;
- latency and storage bounds for cold, warm, incremental, degraded, and
  recovery paths.

Promotion gates should require:

- zero seeded authority, stale-root, path/effect escape, and mandatory
  dependency omission failures;
- identical decisions when irrelevant corpora are enlarged by at least 10x,
  with context growth bounded to fixed index metadata;
- context tokens strongly explained by mandatory closure size and not total
  corpus/repository size;
- exact invalidation of all and only transitive dependents on controlled
  changes;
- deterministic local fallback and fail-closed behavior without
  `ipfs_datasets_py`, vector indexes, a model, or a prover;
- no eager provider/model/dataset imports on package import or discovery; and
- automatic rollback to shadow on any stale binding or safety regression.

Adversarial fixtures should cover retrieved prompt injection, forged CIDs,
canonicalization mismatch, schema drift, cross-partition graph leakage,
similar-but-inapplicable law, expired law, legal exceptions and conflicts,
SecurityIR deny/unknown cases, intent-policy confusion, dirty/untracked file
changes, changed tool arguments after approval, stale leases, proof replay,
graph truncation, poisoned embeddings, unavailable indexes, and restart during
an authorized transaction.

## Delivery sequence

The ASI objective and task boards contain the executable decomposition:

1. Define decision and cross-repository identity contracts.
2. Add the lazy pinned IR registry and three trust-preserving adapters.
3. Bind dirty worktree, AST, behavior, tool, and effect identities.
4. Build the unified dependency graph and generalized reverse proof scope.
5. Add proof-directed retrieval and the completeness-bearing context compiler.
6. Compile intent, legal, and security constraints into plan checks.
7. Issue and enforce exact action permits.
8. Wire the runtime into proposal, implementation, retry, validation, commit,
   and merge paths.
9. Add incremental invalidation, recovery, paired measurement, adversarial
   gates, and shadow-to-automatic rollout.

This sequence deliberately reuses the existing context, analysis, CAS, proof,
authorization, control, and validation contracts. It should add adapters and
one orchestration spine rather than parallel replacement subsystems.
