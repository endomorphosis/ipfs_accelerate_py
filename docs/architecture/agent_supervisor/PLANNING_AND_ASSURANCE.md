# Supervisor planning and assurance pipeline

**Status:** Current
**Audience:** Developers, architects, and implementation agents that plan work,
compile context, analyze repositories, route provers, or interpret evidence for
merge and completion gates
**Scope:** How an objective gap advances through planning, context compilation,
repository/AST/GraphRAG analysis, expected-versus-observed program contracts,
obligations, solver and prover routing, trust-aware caching, edit packets, and
typed completion evidence—while keeping every evidence class distinct
**Non-goals:** Transport-neutral control operations and authorization
([`CONTROL_PLANE.md`](CONTROL_PLANE.md) / DOC-011); multi-lane execution, merge
trains, and rescue loops
([`EXECUTION_AND_RECOVERY.md`](EXECUTION_AND_RECOVERY.md) / DOC-013); sealed delivery
plans and objective heaps as runtime authority; inventing new assurance levels
beyond the live lattice in `formal_verification_contracts`
**Last verified:** `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03);
package layout, claim/evidence tiers, repository forest, program contracts,
context capsules, plan branches, and proof-cache trust boundary checked against
the tree

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Planning package | `ipfs_accelerate_py/agent_supervisor/planning/` | Formal + adaptive plan compile/validate/replan |
| Plan branch IR | `planning.plan_evaluator.PlanBranch` | Schema-validated candidate implementation |
| Task proposal routing | `planning.task_proposal_router` | LLM/analysis proposals → admitted branches |
| Formal plan compiler | `planning.formal_plan_compiler` | Graph + admission projections |
| Formal plan validator | `planning.formal_plan_validator` | Preconditions, findings, countermodels |
| Proof-carrying planner | `planning.proof_carrying_planner` | Plan stages bound to proof obligations |
| Context package | `ipfs_accelerate_py/agent_supervisor/context/` | Capsules and decision runtime |
| Context capsule | `context.context_contracts.ContextCapsule` | Mandatory goal/authority/scope/acceptance |
| Context compiler | `context.context_compiler` | Obligation-first compilation; no silent truncation of required core |
| Analysis package | `ipfs_accelerate_py/agent_supervisor/analysis/` | AST, retrieval, consensus, contracts |
| Analysis pipeline | `analysis.analysis_pipeline` | Integrated orchestration; consensus ≠ completion |
| AST index | `analysis.analysis_ast_index` | Symbol/AST observations |
| Bounded GraphRAG | `analysis.analysis_retrieval.BoundedGraphRAGRetriever`, `analysis.symbolic_contract_graph` | Context-only retrieval candidates |
| Semantic dependency graph | `analysis.semantic_dependency_graph` | Authority-closure rules; model provenance non-authoritative |
| Program contract IR | `program_contracts.py` | Expected vs observed records; distinct kinds |
| Program AST adapters | `program_ast_adapters.py` | Polyglot → `ASTBlobRecord` + observational sidecar |
| Program assurance | `program_assurance_contracts.py` | Assurance claims with stale/authority fail-closed rules |
| Repository forest | `repository_forest.py` | Multi-checkout observation authority unit |
| Forest manifest | `repository_forest_manifest.py` | Portable forest identity helpers |
| Corpus index | `repository_corpus_index.py` | Corpus scanning over forest roots |
| Repository authority | `analysis.repository_authority` | Per-root read/write authority binding |
| Proof package | `ipfs_accelerate_py/agent_supervisor/proof/` | Obligations, provers, cache, attestation |
| Claim / evidence tiers | `proof.code_claim_contracts.EvidenceTier`, `CodeClaimRecord` | Closed ladder; no self-upgrade |
| Assurance lattice | `proof.formal_verification_contracts.AssuranceLevel` | `unverified` … `attested` |
| Proof receipt | `proof.formal_verification_contracts.ProofReceipt` | Typed evidence → re-derived assurance |
| Obligations | `proof.code_proof_obligations`, `CodeProofObligation` | Reviewed property binding |
| Property catalog | `proof.code_property_catalog` | Reviewed properties only |
| Query API | `proof.code_proof_query` | open / satisfied / refuted / impact / proof_delta |
| Re-proof | `proof.code_proof_reproof` | Cache-aware re-evaluation; miss ≠ refutation |
| Trust-aware proof cache | `proof.formal_verification_cache.TrustAwareProofCache` | Hit re-derives assurance; not a trust root |
| Multi-prover router | `proof.multi_prover_router` | Capability-gated routing |
| Kernel verification | `proof.kernel_verification` | Kernel-checked receipts only |
| Attestation / ZK | `proof.proof_attestation`, `proof.provekit_setup`, `proof.ipfs_datasets_zk_attestation` | Simulated ZK never reaches `attested` |
| Edit packets | `proof.code_edit_packet`, `proof.code_edit_materialize` | Implementable packets + validation commands |
| Optional datasets analysis | `integrations/ipfs_datasets_analysis_provider.py`, `ipfs_datasets_program_analysis_provider.py` | Nominate evidence; cannot self-certify success |
| Validation package | `ipfs_accelerate_py/agent_supervisor/validation/` | Deterministic proposal gates |
| External completion | `core/external_completion.py` | External operational receipts (distinct class) |
| Claim contract narrative | [agent_supervisor_code_claim_evidence_contract.md](../agent_supervisor_code_claim_evidence_contract.md) | CBP-025 detail |
| Philosophy / pillars | [AGENT_SUPERVISOR_PHILOSOPHY.md](../AGENT_SUPERVISOR_PHILOSOPHY.md) | Evidence tiers doctrine |
| Package map | [PACKAGE_MAP.md](PACKAGE_MAP.md) | DAG ownership |
| Focused tests | `test/api/test_agent_supervisor_code_claim_evidence_contract.py`, `test_agent_supervisor_formal_planning_contracts.py`, `test_agent_supervisor_context_compiler.py`, `test_agent_supervisor_code_proof_*` | Deterministic gates for this plane |

---

## 1. Purpose

This guide is the maintained product narrative for the **planning and
assurance pipeline**: the path from an objective gap to typed evidence that
policy may use—without collapsing weaker signals into stronger ones.

It answers:

1. What stages exist between “we need work” and “evidence is admissible”?
2. Which packages own each stage?
3. How are **evidence tiers** and **assurance levels** related—and where may
   they never promote?
4. Why optional analysis/dataset providers cannot certify completion?

Readers who need control-plane authorization, lease fencing, or merge-train
mechanics should use sibling guides. This page stops at **evidence production
and classification**, then points at validation and authoritative completion
as the next rungs on the authority ladder.

---

## 2. Context and component map

Planning and assurance span several domain packages on the acyclic DAG.
Higher layers may import lower ones; analysis findings never import daemons to
mint completion.

```text
 Objective gap / open obligations
              │
              ▼
         planning/          PlanBranch candidates, formal plan IR,
                            adaptive refine, proof-carrying stages
              │
              ▼
         context/           ContextCapsule (goal · authority · scope · acceptance)
                            obligation-first compile; proof_delta retries
              │
              ▼
    repository_forest +     Portable multi-root identity + authority
    repository_corpus_index
              │
              ▼
         analysis/          Lexical · AST · GraphRAG · consensus
                            (observations / context only)
              │
              ▼
  program_contracts +       Expected vs observed IR
  program_ast_adapters      (implementation never self-oracles)
              │
              ▼
         proof/             Catalog → obligations → provers → receipts
                            TrustAwareProofCache · edit packets
              │
              ▼
      validation/ (+ gates) Deterministic commands / scope policy
              │
              ▼
  completion evidence       Distinct from merge landings and provider prose
  (authoritative completion elsewhere)
```

| Plane | Packages / modules | Answers | Trust role |
| --- | --- | --- | --- |
| **Planning** | `planning/` | Which candidate branches and plan stages address the gap? | Proposal only |
| **Context** | `context/` | What must the agent or prover see without dropping the invariant core? | Budgeted input; never authority |
| **Repository identity** | `repository_forest.py`, `repository_corpus_index.py` | Which Git roots and trees may be observed, under what authority? | Binding, not proof |
| **Analysis** | `analysis/`, optional datasets adapters | What lexical/AST/GraphRAG facts exist in scope? | Observation / context |
| **Contract IR** | `program_contracts.py`, `program_ast_adapters.py` | Expected vs observed shapes; mismatch sets | Structured observation |
| **Proof / claim** | `proof/` | Obligations, solver candidates, kernel receipts, attestations | Typed evidence ladder |
| **Validation** | `validation/` | Did deterministic checks pass for an admitted proposal? | Gate rung (not kernel) |

Conceptual boxes (for example “human review”) are out of band unless bound by
an explicit receipt or external completion authority.

---

## 3. Assurance pipeline (end-to-end flow)

### 3.1 Happy path

```text
1. Gap         Open objective / claim / obligation set for a bound tree
2. Plan        Produce PlanBranch / formal plan stages (models propose)
3. Context     Compile ContextCapsule; preserve goal·authority·scope·acceptance
4. Forest      Bind repository forest (commit, tree, gitlinks, dirty overlay, authority)
5. Analyze     Lexical + AST + optional GraphRAG over admitted roots only
6. Contract    Compare expected ProgramContractIR vs observed implementation facts
7. Obligation  Compile CodeProofObligation from reviewed properties + scopes
8. Lookup      TrustAwareProofCache: hit re-derives assurance; miss → open
9. Prove       Route solvers / kernels / attestation by required AssuranceLevel
10. Packet     Materialize CodeEditPacket when repair/implement is admitted
11. Validate   Deterministic validation commands (not model prose)
12. Evidence   ProofReceipt / claim lifecycle / completion-class receipts stay typed
```

### 3.2 Failure path (fail closed)

| Failure | Behavior | Must not become |
| --- | --- | --- |
| Missing forest root / path escape | Reject scope | Silent parent-path authority |
| GraphRAG/backend unavailable | Context-only miss or diagnostic | Kernel or completion authority |
| Expected/observed conflict | Report conflict set | Auto-resolve toward implementation |
| Cache miss | Claim status `open` | `refuted` |
| Stale selectors (tree, toolchain, policy, …) | Status `stale`; re-prove | Silent hit reuse |
| Solver candidate only | Cap at `solver_checked` | Merge as `kernel_verified` |
| Simulated ZK / hash placeholder | Non-attested | `attested` |
| Optional datasets provider success text | Analysis nomination | Self-certified task completion |
| LLM plan or patch prose | Proposal | Admission or assurance upgrade |

### 3.3 Interfaces in the packet vocabulary

| Interface | Home | Role in the pipeline |
| --- | --- | --- |
| **PlanBranch@1** (conceptual wire name; type `PlanBranch`) | `planning.plan_evaluator` | One schema-validated candidate: predicted files/symbols, validation commands, cost/risk, source |
| **ContextCapsule@1** | `context.context_contracts` | Bounded stage context; invariant core always present |
| **ProgramContractIR@1** | `program_contracts.py` | Versioned expected/observed program contract IR |
| **ProofReceipt@1** | `formal_verification_contracts.ProofReceipt` | Content-bound proof outcome with re-derived assurance |

These records are **not** interchangeable. A plan branch is not a proof
receipt; a context capsule is not acceptance; a program contract observation
is not a kernel proof.

---

## 4. Stage detail

### 4.1 Planning: models propose, plans stay candidates

**Owner:** `planning/`

- `PlanBranch` records candidate implementations with predicted files, symbols,
  dependencies, validation commands, and optional validation-proof hooks.
- `task_proposal_router` turns structured model/analysis output into
  schema-validated branches; malformed JSON fails closed
  (`PlanBranchValidationError`).
- Formal planning (`formal_plan_compiler`, `formal_plan_validator`,
  `formal_replanner`, `proof_carrying_planner`) binds plan stages to
  preconditions and counterexample-guided repair.
- Adaptive modules (`adaptive_planner`, `adaptive_goal_refiner`) may refine
  goals from **existing** evidence; they must not treat proposals as
  completions.

**Invariant:** planning output is a **proposal**. Admission requires later
validation and, when policy demands, fresh proof evidence.

### 4.2 Context compilation: obligation-first capsules

**Owner:** `context/`

`ContextCapsule` requires separate mandatory fields:

| Field | Why it is mandatory |
| --- | --- |
| `goal` | Work identity cannot be inferred from optional evidence |
| `authority` | Who/what may act; not discoverable from file hits |
| `scope` | Repository/tree/path bounds for analysis and edits |
| `acceptance` | What would count as success for this stage |

Optional evidence and expansion handles may be budgeted. Required core fields
must not be truncated into expansion references
(`context_compiler`, `RequiredContextBudgetEvidence` patterns). Context
compilation **grants visibility**, never merge or completion authority.

Proof-directed and decision-runtime helpers assemble obligation-first prompts
and `proof_delta` retries so cache hits shrink token use without inventing
assurance.

### 4.3 Repository authority forest

**Owner:** `repository_forest.py` (+ `repository_forest_manifest.py`,
`repository_corpus_index.py`, `analysis.repository_authority`)

A **repository forest** is the unit of observation authority for multi-checkout
assurance:

- Each descriptor is derived from its **own Git root**. Sibling roots never
  share Git or mutation authority merely by co-location under a parent path.
- Portable identity binds commit, tree, recursive gitlink closure, dirty
  overlay digest, ignore policy, case/Unicode policy, and read/write authority.
- Host locators and credentials stay outside portable CIDs so equivalent
  relocations replay with the same forest identity.
- Root and path resolution is **fail-closed**: missing roots, escaping paths,
  and symlink escapes reject rather than silently broaden scope.

Analysis and proof that claim to bind a tree must cite forest or repository
identifiers consistent with this unit. Co-location is never shared mutation
authority.

### 4.4 Analysis: lexical, AST, and GraphRAG remain observations

**Owner:** `analysis/` (+ optional `integrations/ipfs_datasets_*`)

| Channel | Typical modules | Evidence tier ceiling |
| --- | --- | --- |
| Lexical / index retrieval | `analysis_retrieval`, corpus index | `query_fact` / observation |
| AST / symbol facts | `analysis_ast_index`, `program_ast_adapters` | `observation` |
| GraphRAG / knowledge-graph candidates | `BoundedGraphRAGRetriever`, `symbolic_contract_graph`, exact datasets adapters when probed | `graphrag_fact` (unverified assurance ceiling) |
| Consensus across channels | `analysis_consensus`, `analysis_pipeline` | Still non-completion |

Rules enforced in code and tests:

1. Retrieval candidates **cannot claim proof authority**
   (`RetrievalBindingError` paths in `analysis_retrieval`).
2. GraphRAG edges are **context-only**; they are never mandatory for authority
   closure (`symbolic_contract_graph`).
3. Model or untrusted provenance cannot create authoritative dependency-graph
   nodes or edges (`semantic_dependency_graph`).
4. Analysis consensus **cannot create completion authority**
   (`analysis_pipeline`).
5. Capability probes for exact datasets GraphRAG/Cypher modules report
   availability only; constructing a capability matrix **never** claims proof
   or completion authority.

Optional `ipfs_datasets_*` providers may nominate analysis results when
installed and probed. They **cannot self-certify success**: provider status
text, simulated surfaces, and degradation evidence are diagnostic.

### 4.5 Program contracts: expected ≠ observed

**Owner:** `program_contracts.py`, `program_ast_adapters.py`,
`program_assurance_contracts.py`

Program contract IR keeps **expectations** and **observations** as distinct
record kinds so implementation behavior can never silently become its own
validation oracle.

Expectation source precedence (closed, ordered):

1. Reviewed MCP++/MCP IDL, JSON Schema, typed interfaces, protocol specs
2. Public signatures, type annotations, stable exports
3. Executable contract and conformance tests
4. Normative documentation
5. Compatibility manifests and generated SDKs

Implementation observations may only populate **observed** contracts.
Conflicting expectations are reported, not silently merged toward code.
Large bodies live in content-addressed artifacts; records carry compact facts
and references. Identities are derived from canonical DAG-JSON and are never
caller-supplied.

`program_ast_adapters` adapt Python, JS/TS, JSON/schema, and Markdown into the
canonical `ASTBlobRecord` plus an **observational** sidecar (dynamic edges stay
ambiguous rather than promoted to resolved authority).

### 4.6 Proof path: catalog → obligation → prove → claim lifecycle

**Owner:** `proof/`

```text
Reviewed property catalog
        │
        ▼
CodeProofObligation  (scope · premises · required assurance · cache keys)
        │
        ├── query / GraphRAG   → EvidenceTier.query_fact / graphrag_fact
        ├── test/runtime/static → observation
        ├── ATP/SMT/solver     → solver_candidate
        ├── kernel check       → kernel_proof
        └── crypto attestation → cryptographic_attestation
        │
        ▼
ProofReceipt  (assess_assurance / derive_assurance re-derives level)
        │
        ▼
CodeClaimRecord lifecycle:
  unknown | open | satisfied | refuted | unsupported | not_measured | stale
```

| Stage | Module family | Notes |
| --- | --- | --- |
| Catalog | `code_property_catalog` | Arbitrary NL claims fail closed |
| Obligation compile | `code_proof_obligations` | Binds property, tree, scopes, invalidators |
| Query | `code_proof_query` | open/satisfied/refuted/impact/proof_delta |
| Provider / matrix | `formal_verification_provider`, `multi_prover_router`, `prover_matrix_registry` | Capability-gated |
| Kernel | `kernel_verification` | Only path to independent kernel tier |
| Attestation | `proof_attestation`, ProveKit gates | Requires prior kernel-verified receipt |
| Edit materialization | `code_edit_packet`, `code_edit_materialize` | Packets carry validation commands, not self-proof |
| Re-proof | `code_proof_reproof` | Invalidation selectors drive stale |

Authoritative assurance is always a **projection of typed `ProofEvidence`**
via `assess_assurance` / `derive_assurance`. Claim records bind and lifecycle
those results; they do not create a second lattice.

### 4.7 Validation and completion evidence (downstream rungs)

**Validation** (`validation/`) selects and runs deterministic commands before a
patch is accepted. Model prose is never a substitute for validation commands.

**Completion receipts** are a separate evidence class from:

- analysis hits,
- solver candidates,
- kernel proofs,
- merge landings, and
- external operational completion authorities (`core/external_completion`).

Authoritative task completion (merge ≠ acceptance) is owned by execution
guides and `todo_daemon/authoritative_completion.py`. This planning/assurance
guide only requires that those receipts **remain distinct** in the type system
and in operator language.

---

## 5. Evidence tiers and assurance levels

These two ladders are related but not identical. **Evidence tier** classifies
*what kind of artifact supports a claim*. **Assurance level** is the policy
lattice used by gates after re-derivation from typed evidence.

### 5.1 Evidence tier table

| Evidence tier | Typical producers | Independent assurance ceiling | May independently mint kernel/attested? |
| --- | --- | --- | --- |
| `query_fact` | Repository queries, lexical hits | `unverified` | No |
| `graphrag_fact` | GraphRAG / enrichment projections | `unverified` | No |
| `observation` | Bounded tests, runtime traces, static analysis, AST facts | `candidate` | No |
| `solver_candidate` | ATP / SMT / solver / model-check candidates; LLM draft artifacts mapped as candidates | `solver_checked` | No |
| `kernel_proof` | Independent kernel-checked verification | `kernel_verified` | Kernel only (not attested) |
| `cryptographic_attestation` | Real attestation over a kernel receipt | `attested` | Yes (attested requires kernel base) |

Source of truth: `proof.code_claim_contracts.EvidenceTier`.

### 5.2 Assurance level lattice

| Level | Rank | Meaning |
| --- | --- | --- |
| `unverified` | 0 | No independent verification of the claim |
| `candidate` | 1 | Bounded observation or draft-quality support |
| `solver_checked` | 2 | Solver/model-check result without small trusted kernel |
| `kernel_verified` | 3 | Accepted by independent kernel verification |
| `attested` | 4 | Cryptographic attestation over a kernel-verified receipt |

Source of truth: `proof.formal_verification_contracts.AssuranceLevel`.

### 5.3 Non-upgrade (trust tier) rules

1. **No silent promotion.** Renaming, caching, or consensus cannot move a
   weaker evidence tier into a stronger one.
2. **Query / GraphRAG / observations** cannot independently mint
   `kernel_verified` or `attested`.
3. **Solver candidates** remain non-authoritative for merge/completion unless
   policy independently verifies them at the required level.
4. **LLM proposals and plan branches** are never an evidence tier above
   candidate/solver mapping; fluent text does not climb the ladder.
5. **Deterministic tests** are observations (or validation gates), not kernel
   proofs.
6. **ZK / attestation:** only real attestation over an eligible
   kernel-verified receipt reaches `attested`. Simulated ZK, hash-commitment
   placeholders, and ProveKit-ineligible predicates stay non-attested.
7. **Completion receipts** are their own class; landing code or merging a
   branch is not task acceptance and not a kernel proof.
8. **Optional datasets providers cannot self-certify success.** They may
   supply analysis nominations and capability receipts only.

### 5.4 Distinct classes checklist (acceptance map)

| Class | Examples | Trust ceiling if alone |
| --- | --- | --- |
| Lexical / AST / GraphRAG findings | Index hits, AST facts, GraphRAG candidates | Unverified / observation |
| LLM proposals | `PlanBranch`, edit drafts, Leanstral drafts | Proposal / candidate |
| Deterministic tests | Pytest, validation commands | Observation / validation gate |
| Solver candidates | ATP/SMT results, model-check candidates | `solver_checked` |
| Kernel proofs | `kernel_verification` receipts | `kernel_verified` |
| ZK attestations | ProveKit / attestation backends over kernel receipts | `attested` |
| Completion receipts | Authoritative completion, external operational receipts | Completion class only—not interchangeable with kernel |

Keeping these rows **distinct** is a correctness property, not a style
preference.

---

## 6. Provider and capability matrix

| Provider / surface | Package | What success means | What it must not claim |
| --- | --- | --- | --- |
| Local AST / lexical | `analysis/` | Structured observations for a bound forest | Kernel or completion |
| Bounded GraphRAG (local fallback) | `analysis_retrieval`, integrations local adapters | Context candidates | Proof authority |
| Exact datasets GraphRAG / Cypher | `integrations/ipfs_datasets_analysis_provider` | Capability probe + optional retrieval when modules load | Exact use without probe; completion; kernel |
| Program analysis datasets bridge | `ipfs_datasets_program_analysis_provider` | Optional enrichment | Self-certified success; non-diagnostic authority from simulated surfaces |
| Formal verification providers | `formal_verification_provider`, multi-prover router | Stage-appropriate evidence | Claimed assurance text without re-derivation |
| Leanstral / model draft | `leanstral_proof_provider`, goal development | Unverified draft artifacts | Kernel/attested without independent check |
| ZK / ProveKit | `provekit_setup`, `proof_attestation`, datasets ZK helpers | Attestation only when eligible | Simulated attested |
| Validation runtime | `validation/` | Deterministic command outcomes | Substitute for required proof tier |

**Discovery vs capability vs proof** (same doctrine as philosophy):

| Signal | Means | Does not mean |
| --- | --- | --- |
| Import / discovery | Module vocabulary exists | Backend works |
| Capability probe | Configured path can attempt work | Proof or completion |
| Proof / validation receipt | A specific claim met a stated assurance level | Future claims are free |

---

## 7. Cache trust rules

There is **one** formal proof-cache trust boundary:
`proof.formal_verification_cache.TrustAwareProofCache`.

| Rule | Detail |
| --- | --- |
| Cache is not a trust root | Hits reconstruct typed `ProofReceipt` and **re-derive** assurance from immutable evidence |
| Provider status ignored for admission | Claimed assurance strings never admit an entry |
| Semantic bindings checked | Obligation, tree, toolchain, policy, and key material must match |
| Miss ≠ refutation | Claim lifecycle maps miss → `open` (never `refuted`) |
| Stale ≠ refutation | Invalidation selectors mark `stale`; re-prove required |
| Rejected / poisoned | Malformed or mismatched entries fail closed |
| Draft / single-flight channels | May share expensive computation; non-authoritative outcomes are not proof-cache hits |
| Analysis caches | Separate from formal proof cache; analysis hits stay observational |

Claim records may memoize lookup outcomes (`hit` / `miss` / `stale` /
`rejected`) for metrics; they must not store a parallel cache root or
alternate assurance lattice.

Invalidation selectors include repository tree, AST scope, premise/assumption
sets, toolchain, policy, catalog, property, and obligation identities
(`InvalidationSelectorKind` in `code_claim_contracts`).

---

## 8. State and identity

| Identity | Binds | Used by |
| --- | --- | --- |
| Forest / repository id | Git root, commit, tree, gitlinks, dirty overlay, policies, authority | Analysis scope, proof binding |
| `objective_id` / revision | Durable intent | Capsules, plans, claims |
| `policy_id` / revision | Admission and assurance policy | Capsules, cache keys |
| Obligation id | Content identity of obligation payload | Query, cache, claims |
| `claim_id` | Content identity of claim payload | Lifecycle queries |
| Receipt / evidence ids | Typed evidence bodies | Assurance re-derivation |
| Cache key | Obligation + semantic/execution bindings | Trust-aware lookup |

Caller-supplied identity strings that do not match re-derived digests are
rejected. Large source and witness bodies live behind content-addressed
references rather than inline duplication in IR records.

---

## 9. Trust, authorization, and failure semantics

### 9.1 Who may act

| Actor | May | May not |
| --- | --- | --- |
| Model / planner | Propose branches, drafts, candidate proofs | Admit merges, mint kernel assurance, complete tasks by prose |
| Analysis / datasets provider | Nominate observations when capable | Self-certify success or completion |
| Solver backend | Emit solver-tier evidence | Skip kernel when policy requires kernel |
| Kernel verifier | Emit kernel-tier receipts | Attest without cryptographic backend |
| Attestation backend | Attest eligible kernel receipts | Elevate simulated proofs |
| Validation runtime | Pass/fail deterministic commands | Replace required proof tier |
| Control / daemon (out of scope here) | Admit mutations under policy | Treat this guide as a second policy engine |

### 9.2 Fail-closed defaults

- Unknown operational state does not satisfy a true constraint.
- Optional extras absent → capability miss, not free proof.
- Conflicting expectations → conflict report, not silent code-wins.
- Stale forest or selectors → re-bind and re-prove.
- Natural-language claims without reviewed property/obligation → reject.

### 9.3 Degradation

When a preferred prover or GraphRAG backend is unavailable, the system may
degrade to diagnostics, local lexical fallbacks, or explicit
`unsupported` / `not_measured` claim states. Degradation evidence remains
non-authoritative for completion.

---

## 10. Rationale

1. **Objective-driven work without evidence collapse.** Agents need a pipeline
   from goals to patches, but fluency must not equal verification.
2. **Multi-root reality.** Supervisor work often spans accelerator, kit,
   datasets, and other checkouts; forest identity prevents co-location from
   faking shared Git authority.
3. **Expected vs observed separation.** Contract assurance fails if the system
   under test can rewrite its own oracle.
4. **One assurance lattice.** Claim records normalize lifecycle without
   inventing a second trust model beside `AssuranceLevel`.
5. **Cache as accelerator, not authority.** Parallel lanes need shared memo
   without poisoned or provider-claimed assurance.

---

## 11. Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Single “confidence score” for all evidence | Collapses tiers; un-auditable upgrades |
| Treat GraphRAG consensus as proof | Context projections are not kernel-checked |
| Cache-hit short-circuit without re-derivation | Poisoned entries and stale trees would mint trust |
| Let datasets providers mark tasks complete | Optional integrations would self-certify |
| Implementation observations as expectations | Self-oracling contracts |
| Merge commit as task acceptance | Landing code ≠ acceptance (separate ladder rung) |
| Simulated ZK as production attestation | Breaks `attested` meaning |

A simpler design that folds these distinctions would make false admissions
cheaper than correct refusals—the opposite of the supervisor’s fail-closed
posture.

---

## 12. Consequences

**Positive**

- Operators and agents share one vocabulary for evidence class and assurance.
- Parallel documentation and implementation lanes can extend analysis or
  provers without rewriting trust rules.
- Cache and GraphRAG can improve latency and recall without becoming trust
  roots.
- Optional providers stay useful without owning completion.

**Negative / costs**

- More types and stages than a chat-only agent loop.
- Capability probes and forest binding add setup surface.
- Re-proof after invalidation costs wall time (accepted for soundness).
- Writers must not cite plans or boards as live assurance APIs.

---

## 13. Extension and compatibility

| When extending… | Do |
| --- | --- |
| New analysis channel | Land under `analysis/` or `integrations/`; cap at observation/GraphRAG tiers |
| New prover | Register through formal verification / multi-prover paths; map evidence kinds to tiers |
| New claim family | Extend closed catalog with review; fail closed for freeform NL |
| New cache | Do not fork a second formal proof-cache trust root |
| New provider | Expose capability probes; forbid completion/proof authority fields on success text |
| Docs | Keep this guide Current; leave sealed plans Historical/Plan |

Compatibility facades and root-level re-exports must not reintroduce a flat
module warehouse that hides package ownership
([PACKAGE_MAP.md](PACKAGE_MAP.md)).

---

## 14. Operational signals

| Signal | Where | Use |
| --- | --- | --- |
| Cache lookup status | `TrustAwareProofCache` / claim `cache_lookup` | hit, miss, stale, rejected metrics |
| Claim lifecycle counts | `code_proof_query`, claim records | open/satisfied/refuted/stale dashboards |
| Capability receipts | datasets/GraphRAG probes, formal verification capabilities | Distinguish unconfigured vs failed |
| Plan validation findings | `formal_plan_validator` | Countermodels and precondition failures |
| Forest bind errors | `repository_forest` | Scope misconfiguration |
| Validation command results | `validation/` | Pre-merge deterministic gates |
| Attestation eligibility | ProveKit / attestation policy | Simulated vs real attested |

Logs and metrics should name **tier** and **assurance** explicitly so operators
do not infer kernel success from “analysis green” or “provider returned OK”.

---

## 15. Verification

Deterministic checks for this guide’s claims:

```bash
# Guide contract
test -f docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md
rg -q 'Source anchors' docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md
rg -qi 'evidence tier|trust tier' docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md
git diff --check

# Live symbols still present
test -d ipfs_accelerate_py/agent_supervisor/planning
test -d ipfs_accelerate_py/agent_supervisor/context
test -d ipfs_accelerate_py/agent_supervisor/analysis
test -d ipfs_accelerate_py/agent_supervisor/proof
test -f ipfs_accelerate_py/agent_supervisor/repository_forest.py
test -f ipfs_accelerate_py/agent_supervisor/program_contracts.py
test -f ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py
rg -q 'class EvidenceTier' ipfs_accelerate_py/agent_supervisor/proof/code_claim_contracts.py
rg -q 'class AssuranceLevel' ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py
rg -q 'class PlanBranch' ipfs_accelerate_py/agent_supervisor/planning/plan_evaluator.py
rg -q 'class ContextCapsule' ipfs_accelerate_py/agent_supervisor/context/context_contracts.py
rg -q 'class TrustAwareProofCache' ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py

# Focused contract tests (when environment provides deps)
python -m pytest test/api/test_agent_supervisor_code_claim_evidence_contract.py -q
```

Reviewer checklist:

- [ ] Lexical/AST/GraphRAG, LLM proposals, tests, solver candidates, kernel
      proofs, ZK attestations, and completion receipts are described as
      **distinct** classes.
- [ ] Optional datasets providers are documented as **unable to self-certify**
      success.
- [ ] Evidence tier table matches `EvidenceTier` and ceilings.
- [ ] Cache hits re-derive assurance; miss ≠ refutation.
- [ ] Forest authority rules reject co-location as shared Git authority.
- [ ] No protected refresh plan/todo/objectives files were edited for this task.

---

## 16. Related ADRs / guides

| Document | Role |
| --- | --- |
| [AGENT_SUPERVISOR_PHILOSOPHY.md](../AGENT_SUPERVISOR_PHILOSOPHY.md) | Pillars and authority ladder |
| [agent_supervisor_code_claim_evidence_contract.md](../agent_supervisor_code_claim_evidence_contract.md) | `CodeClaimRecord@1` detail |
| [AGENT_SUPERVISOR_EXTERNAL_COMPLETION_RECEIPTS.md](../AGENT_SUPERVISOR_EXTERNAL_COMPLETION_RECEIPTS.md) | External operational completion class |
| [PACKAGE_MAP.md](PACKAGE_MAP.md) | Domain package DAG |
| [packages/planning.md](packages/planning.md), [packages/context.md](packages/context.md), [packages/analysis.md](packages/analysis.md), [packages/proof.md](packages/proof.md), [packages/validation.md](packages/validation.md) | Per-package semantic maps |
| [GUIDE_CONVENTIONS.md](../GUIDE_CONVENTIONS.md) | Architecture guide contract |
| [CONTROL_PLANE.md](CONTROL_PLANE.md) (DOC-011) | Intent, operations, authorization |
| [EXECUTION_AND_RECOVERY.md](EXECUTION_AND_RECOVERY.md) (DOC-013) | Daemons, merge, rescue, authoritative completion |
| Codebase-proof / formal planning **plans** under `docs/architecture/` | Intent/history only—not Current runtime contracts |

---

## Appendix A — Authority classes and possible strengthening

```text
1. Intent         objective / obligation identity
2. Proposal       PlanBranch, model draft, edit packet body
3. Context bind   ContextCapsule + forest scope
4. Observation    lexical / AST / GraphRAG / tests
5. Solver tier    solver_candidate evidence
6. Kernel tier    kernel_proof → kernel_verified
7. Attestation    cryptographic_attestation → attested
8. Validation     deterministic commands / scope policy
9. Completion     typed completion receipts (distinct class)
```

This is a vocabulary of distinct classes, not a mandatory sequential pipeline.
Policy chooses the required assurance level: a path may have no GraphRAG
artifact, a solver candidate may be checked directly, and kernel or attestation
work is required only when its gate demands that level. What is forbidden is a
silent promotion—skipping a *required* gate, renaming weaker evidence, or using
a cache hit to claim a stronger class than re-derivation permits.

---

## Appendix B — Program evidence (optional)

Historical boards and objective heaps (codebase-proof, analysis SCA, formal
planning) may cite modules on this pipeline. Product code and this guide do
**not** require board prefixes to understand planning or assurance. Board
completion markers are not authoritative completion of those tasks on a later
tree.
