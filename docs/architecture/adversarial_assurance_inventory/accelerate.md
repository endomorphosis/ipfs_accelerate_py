# AAE-001 Accelerate inventory

**Evidence:** `aae/accelerate-inventory@1`  
**Interface:** `AAEAccelerateInventory@1`  
**Machine-readable twin:** `docs/architecture/adversarial_assurance_inventory/accelerate.json`  
**Task:** AAE-001 — Inventory accelerate execution, verification, policy, state-machine, and ZK surfaces  
**Inspected tree:** `3256d8bb813173349388a931e77d8ecf4bbcb8ef`  
**Tree OID:** `a175187da75ff4028b0359abec879841f707b99b`  
**Planning authority pin:** `7c9f3fa3d2ac14c7b5bfa5036e2fe6fb59f0afda` (ancestor of inspected tree)  
**IVP public-API freeze:** `8c7800cedc5e1b848367db9952f912428466f8cc`

This is a read-only inventory of public current-tree surfaces under
`ipfs_accelerate_py.agent_supervisor` that adversarial assurance will reuse.
It records exact exports, signatures, status vocabularies, manifests,
receipts, isolation and resource seams, ZK nonclaims, tests, and known blind
spots. RED, simulated, unavailable, and missing-capability evidence are
recorded honestly. No production code is changed.

Nested gitlinks observed at inventory time:

| Nested repository | Commit |
| --- | --- |
| `ipfs_datasets_py` | `fbd1ba9f70803de157622bb20e22595ef09d606f` |
| `ipfs_kit_py` | `c7e5feeb24582ab68c1f5ca626366b665a82ad61` |
| `ipfs_accelerate_py/mcplusplus` | `dc3164653a48d059ae9812078359daeafb451c07` |

The planned package `ipfs_accelerate_py.agent_supervisor.adversarial_assurance`
is **not present**. Missing campaign APIs and proof-sealer surfaces are
`typed_unavailable`, not inventable substitutes.

---

## 1. Claimed interfaces (path- and revision-bound)

| Interface | Interface ID | Source path | Source revision | Primary test path | Test revision |
| --- | --- | --- | --- | --- | --- |
| `IncrementalVerificationPlanner` | `IncrementalVerificationPlanner@1` | `ipfs_accelerate_py/agent_supervisor/verification/planner.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_incremental_verification_planner.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| `VerificationReceiptCache` | `VerificationReceiptCache@1` | `ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_verification_receipt_cache.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| `VerificationExecutor` | `VerificationExecutor@1` | `ipfs_accelerate_py/agent_supervisor/verification/executor.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_verification_executor.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| `VerificationProcessRunner` | `VerificationProcessRunner@1` | `ipfs_accelerate_py/agent_supervisor/verification/process_runner.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_verification_process_runner.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| `ModelRoutePlanner` | `ModelRoutePlanner@1` | `ipfs_accelerate_py/agent_supervisor/verification/model_route.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_verification_model_route.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| `ContextPacker` | `ContextPack@1` | `ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py` | `69b8893087b491088a49bb53b8f256ec6fadcc3b` | `test/api/semantic_state/test_context_pack.py` | `69b8893087b491088a49bb53b8f256ec6fadcc3b` |
| `IsolatedWorktree` | `IsolatedPatchWorktree@1` | `ipfs_accelerate_py/agent_supervisor/semantic_state/worktree.py` | `f1b5d1572be8537a138fa5ab1d69f120774664e2` | `test/api/semantic_state/test_worktree.py` | `f1b5d1572be8537a138fa5ab1d69f120774664e2` |
| `ResourceScheduler` | `ResourceScheduler@1` | `ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py` | `6d2c27b91e1d726d66af9d06f70bf5646dd5559c` | `test/api/test_agent_supervisor_adaptive_resources.py` | (see tests) |
| `MutationLedger` | `MutationLedger@1` | `ipfs_accelerate_py/agent_supervisor/analysis/mutation_ledger.py` | `8c0aed40213b0d92c7eb58ef6299522daa2ec92f` | `test/api/test_agent_supervisor_mutation_ledger.py` | `8c0aed40213b0d92c7eb58ef6299522daa2ec92f` |
| `FormalVerificationPolicy` | `FormalVerificationPolicy@1` | `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_policy.py` | `22f7f54063ef1a599940f1a6754f6503361e5b1e` | `test/api/test_agent_supervisor_formal_verification_policy.py` | `22f7f54063ef1a599940f1a6754f6503361e5b1e` |
| `ProgramAnalysisZKP` | `ProgramAnalysisZKP@1` | `ipfs_accelerate_py/agent_supervisor/proof/program_analysis_zkp.py` | `ab984496fb1109700ec37ed91bda5b29d7214ecb` | `test/api/test_agent_supervisor_program_analysis_zkp.py` | `a3714348549b0ecc50f40d30ca3da091706686da` |
| `DatasetsZkAttestation` | `IpfsDatasetsZkAttestation@1` | `ipfs_accelerate_py/agent_supervisor/proof/ipfs_datasets_zk_attestation.py` | `16a819175a29e694a81da752020f46f062e2b8e0` | `test/api/test_agent_supervisor_ipfs_datasets_zk_attestation.py` | `16a819175a29e694a81da752020f46f062e2b8e0` |

`source_revision` / `test_revision` are the last commits that modified those
files. The inventory itself is bound to tree revision
`3256d8bb813173349388a931e77d8ecf4bbcb8ef`.

---

## 2. Execution surfaces

### VerificationExecutor (`VerificationExecutor@1`)

- Package export hub: `ipfs_accelerate_py.agent_supervisor.verification`
  (lazy, side-effect free; freeze evidence `ivp/public-api@1`)
- Schema: `ipfs_accelerate_py/agent-supervisor/verification-executor@1`
- Evidence: `ivp/execution-bundle@1`
- Result schema: `ipfs_accelerate_py/agent-supervisor/verification-execution-result@1`
- Entrypoints: `execute_verification_plan`, `create_verification_executor`,
  `compute_production_acceptance`, class `VerificationExecutor`

Signature (module entry):

```text
execute_verification_plan(
  verification_plan, *,
  observed_identities=None, post_observed_identities=None,
  cancellation=None, step_bindings=None, advisory_key_cids=(),
  human_review_required=None, model_route_decision=None,
  routing_hints=None, cache=None, process_runner=None,
  resource_scheduler=None, host_snapshot=None,
  check_runner=None, check_runners=None, require_resource_lease=...
) -> VerificationExecutionResult
```

Result carries plan, bundle, summary, commitment, model route, production
acceptance, reused vs executed receipts, counterexamples, resource rejections,
cancellation/timeout fencing, identity revalidation, and reason codes.

Production acceptance rules:

- `human_review_required` blocks acceptance.
- `mandatory_fallback_pending` blocks acceptance.
- Every non-advisory required receipt key must be present and successful
  (`passed` / `proved`).
- Advisory keys may remain unresolved; they are never upgraded by
  `compute_production_acceptance`.

Resource rejection kinds: `plan_resource_lease_denied`,
`resource_lease_denied`, `capacity_exhausted`, `lease_revoked`,
`bounded_parallelism_cap`.

Focused tests: `test/api/test_agent_supervisor_verification_executor.py`
(rev `52e8fe17fdc7ac63f905872b194d930ae36ab1db`).

### VerificationProcessRunner (`VerificationProcessRunner@1`)

- Source: `verification/process_runner.py` (rev `52e8fe17...`)
- Schema: `ipfs_accelerate_py/agent-supervisor/verification-process-runner@1`
- Evidence: `ivp/process-runner@1` (+ process-tree cancellation
  `ivp/process-tree-cancellation@1`)
- Exports: `VerificationProcessRunner`, `VerificationCommand`,
  `VerificationRunResult`, `VerificationCancellation`,
  `build_closed_sandbox`, `build_hermetic_environment`, `fence_process_tree`,
  `NETWORK_POLICY_DENY_ALL`

Run dispositions: `completed`, `timeout`, `cancelled`, `unavailable`,
`failed`.

Isolation properties:

- Explicit argv only; no shell interpolation of untrusted paths.
- Network policy defaults to deny-all.
- Hermetic environment strips production credentials and ambient secrets.
- ResourceScheduler lease required; capacity exhaustion is typed rejection.
- Process-tree fencing cancels without accepting late receipts as success.
- Bounded stdout/stderr capture.

Related schemas: `hermetic-sandbox@1`, `hermetic-sandbox-policy@1`,
`verification-filesystem-policy@1`, `verification-sandbox-identity@1`.

Tests: `test/api/test_agent_supervisor_verification_process_runner.py`,
`test/api/test_agent_supervisor_process_tree_fencing.py`.

### IsolatedWorktree (`IsolatedPatchWorktree@1`)

- Source: `semantic_state/worktree.py` (rev `f1b5d157...`)
- Exports: `IsolatedWorktree`, `create_isolated_worktree`,
  `recover_isolated_worktree`, `validate_patch`, `apply_patch`,
  `PatchValidator`, `WorktreePhase`

Phases: `preparing`, `ready`, `validating`, `applying`, `applied`,
`rejected`, `cleaning`, `terminal`.

Properties: detached worktree at pinned base; preimage validation and scope
enforcement; fence/phase journal; cleanup only of owned disposable worktrees.

Test: `test/api/semantic_state/test_worktree.py` (rev `f1b5d157...`).

### ResourceScheduler (`ResourceScheduler@1`)

- Source: `runtime/resource_scheduler.py` (rev `6d2c27b9...`)
- Methods: `evaluate`, `acquire`, `release`, `cancel`, `schedule`,
  `metrics_snapshot`
- Also exports proof-work scheduling types and adaptive admission helpers used
  by formal-verification and verification runners.

AAE reuse: admit mutation workers and campaign parallelism; capacity exhaustion
is typed rejection, not silent oversubscription.

---

## 3. Verification surfaces

### IncrementalVerificationPlanner (`IncrementalVerificationPlanner@1`)

- Schema: `ipfs_accelerate_py/agent-supervisor/incremental-verification-planner@1`
- Evidence: `ivp/verification-plan@1`
- Public freeze: `8c7800ced...` / `IncrementalVerificationPublicApi@1`
- Exports: `IncrementalVerificationPlanner`, `create_verification_plan`,
  `create_incremental_verification_planner`

```text
create_verification_plan(
  repository_state, invalidation_plan, context_pack, patch_delta, policy,
  *, cache=None, adapter=None
) -> VerificationPlan
```

Properties: deterministic, side-effect free, exact-key cache lookup only,
broadens under uncertainty, forces human review for unbound sandbox / policy
conflict / scope crossing, import performs no I/O.

Related policy schema: `verification-planner-policy@1`.

Tests: `test/api/test_agent_supervisor_incremental_verification_planner.py`
(and report/conformance suites).

### VerificationReceiptCache (`VerificationReceiptCache@1`)

- Source: `receipt_cache.py` (rev `52e8fe17...`); store collaborator
  `receipt_store.py`
- Exports: `VerificationReceiptCache`, `production_eligible`

Methods: `lookup`, `lookup_many`, `admit`, `mark_stale`, `tombstone`,
`replay`, GC metadata, historical index.

Cache reuse dispositions: `reused`, `stale`, `missing`, `corrupt`,
`mismatched`, `simulated`, `non_authoritative`, `policy_rejected`,
`terminal_status_rejected`.

Production admission: exact key only; re-derive identity/CID/kind/status;
success terminals only `passed`/`proved`; stale/simulated/timeout/unavailable/
invalid/corrupt never production; cache presence is never authority.

### ModelRoutePlanner (`ModelRoutePlanner@1`)

- Source: `model_route.py` (rev `52e8fe17...`)
- Closed routes: `deterministic_only`, `small_local_model`, `medium_model`,
  `frontier_model`, `human_review_required`
- Provider-neutral inventory only; vendor identity fields rejected; no
  downgrade when tier unavailable → `human_review_required`.

Distinct from harness routing at `semantic_state/routing.py`
(`route_model`, rev `644ad9df...`).

### Contracts and terminal statuses

Source: `verification/contracts.py` (rev `0cdf81bd...`).

| Vocabulary | Values |
| --- | --- |
| `TerminalStatus` | `passed`, `failed`, `proved`, `disproved`, `unknown`, `timeout`, `unavailable`, `not_modeled`, `stale`, `invalid`, `cancelled`, `simulated` |
| `VerificationReceiptKind` | `static_analysis`, `type_check`, `test`, `proof` |
| `CacheReuseDisposition` | see cache section |
| `ModelRoute` | see route section |

Canonical types include `VerificationPlan`, `VerificationBundle`,
`VerificationSummary`, `VerificationCommitment`, kinded receipts,
`CounterexampleReceipt`, `ModelRouteDecision`, `CacheReuseDecision`,
`VerificationReceiptKey`, `DirectExecutionObservation`.

### Selection and full-suite evaluation

- Selection: `verification/selection.py` — `select_affected_verification`,
  dispositions `exact` / `broader` / `full_suite`
- Evaluation: `verification/evaluation.py` —
  `compare_selected_with_full_suite`, corpus loaders; measures incremental vs
  full-suite cost, not paired semantic shadowing
- Minimizer: `verification/counterexamples.py` —
  `minimize_counterexample` / `CounterexampleMinimizer@1`
- Bundle: `build_verification_bundle`, `build_verification_summary`,
  `build_verification_commitment`
- Source snapshot: `build_source_snapshot` / `source_snapshot_id`
  (`source_snapshot.py`, rev `bca15baf...`)

Adapters: `adapters/pytest_adapter.py`, `mypy_adapter.py`,
`prover_adapters.py`.

---

## 4. Policy surfaces

### Planner / selection / model-route policies

| Policy | Schema / module | Role |
| --- | --- | --- |
| `PlannerPolicy` | `verification-planner-policy@1` | Incremental plan bounds, human-review triggers |
| `SelectionPolicy` | selection module `SELECTION_POLICY_SCHEMA` | Exact/broader/full_suite expansion |
| `ModelRoutePolicy` | model-route module | Capability-class routing without provider identity |
| Process sandbox policies | `hermetic-sandbox-policy@1`, `verification-filesystem-policy@1` | Closed sandbox and FS policy |

### FormalVerificationPolicy (`FormalVerificationPolicy@1`)

- Source: `proof/formal_verification_policy.py` (rev `22f7f540...`)
- Contracts: `proof/formal_verification_contracts.py` (same revision)
- No prover integration; selects assurance and fallback requirements only

| Vocabulary | Values |
| --- | --- |
| `ProofResultStatus` | `proved`, `disproved`, `unsupported`, `unavailable`, `timed_out`, `inconclusive`, `cancelled`, `error`, `missing` |
| `RolloutMode` | `disabled`, `shadow`, `canary`, `enforcement` (blocks on canary/enforcement) |
| `AssuranceLevel` | `unverified`, `candidate`, `solver_checked`, `kernel_verified`, `attested` |

Fail-closed: `unsupported`, `unavailable`, `timed_out`, `inconclusive`, and
`missing` never satisfy an assurance requirement. Fallback only when the rule
explicitly allows it and every named validation has an explicit pass.

Receipts: `MergeProofGateReceipt`, `RolloutTransitionReceipt`,
`OverrideReceipt`, `PolicyGateDecision`.

**Disambiguation:** `RolloutMode.SHADOW` is a proof-policy rollout gate, not
paired semantic shadowing and not AAE campaign completion.

Tests: `test/api/test_agent_supervisor_formal_verification_policy.py`
(rev `22f7f540...`).

### Semantic-state receipt admission

Source: `semantic_state/receipts.py` (rev `be82f85e...`).

| Vocabulary | Values |
| --- | --- |
| Freshness | `fresh`, `stale`, `unknown` |
| Admission | `admitted`, `incomplete`, `rejected`, `simulated`, `stale`, `unavailable` |
| Provider modes | `production`, `development`, `simulated` |

Gates: `admit_receipt`, `receipt_may_promote_root`, `receipt_may_verify`.
Simulated and stale receipts never promote production roots.

### Provider gate

`semantic_state/providers.py` (rev `644ad9df...`): `invoke_model`,
`ProductionProviderGate`, modes `production` / `development` / `simulated`.
Simulated paths cannot report production verification;
`human_review_required` blocks dispatch.

---

## 5. State-machine surfaces

### Worktree phases

`WorktreePhase`: preparing → ready → validating → applying → applied |
rejected → cleaning → terminal.

### MutationLedger lifecycle (lineage only)

`MutationLedger@1` schema
`ipfs_accelerate_py/agent-supervisor/mutation-ledger@1`
(source rev `8c0aed40...`).

| Vocabulary | Values |
| --- | --- |
| `MutationStatus` | `accepted`, `rejected`, `quarantined`, `no_op`, `partial`, `rolled_back` |
| `MutationDisposition` | `accepted`, `rejected`, `quarantined`, `no_op`, `formatting_only`, `parse_failed`, `stale_fence`, `snapshot_mismatch`, `missing_lineage`, `partial_write` |
| `FileChangeKind` | `added`, `modified`, `deleted`, `renamed`, `no_op`, `formatting_only`, `parse_failed` |
| `FenceStatus` | `active`, `superseded`, `released` |
| `RollbackStatus` | `verified`, `failed`, `pending` |

**Authority limit:** reusable file-lineage evidence only. Not an assurance
campaign engine. Does not generate semantic mutants, expected-detection sets,
gap taxonomies, or promotion flows. Must not be rebranded as
`AdversarialAssuranceEngine`.

### Lifecycle orchestrator (process fencing)

`control/lifecycle_orchestrator.py` (rev `a9c25754...`) exposes fenced process
lifecycle profiles, transition intents, sagas, and receipts
(`lifecycle-profile@1`, `lifecycle-transition-intent@1`,
`lifecycle-transition-saga@1`, `lifecycle-transition-receipt@1`). Reusable
fencing patterns for workers; **not** the AAE campaign state machine.
(Operator-protected path; inventory is read-only.)

### Planned AAE campaign pipeline (not implemented)

From the program plan, not present as code on this tree:

1. unmutated green baseline  
2. risk-select targets  
3. generate bounded candidates  
4. admit valid mutants  
5. run predicted incremental checks  
6. broaden survivors  
7. classify  
8. minimize important survivors  
9. diagnose gaps  
10. propose remediations  
11. evaluate held-out  
12. seal campaign  

---

## 6. Manifests and receipts

| Artifact | Schema / interface | Builder / owner |
| --- | --- | --- |
| Verification plan | `verification-plan@1` / `VerificationPlan@1` | `create_verification_plan` |
| Verification bundle | `VerificationBundle@1` | `build_verification_bundle` |
| Verification commitment | `verification-commitment@1` / `VerificationCommitment@1` | `build_verification_commitment` (**not ZK**) |
| Execution result | `verification-execution-result@1` | `execute_verification_plan` |
| Kinded receipts | `StaticAnalysis` / `TypeCheck` / `Test` / `Proof` / `Counterexample` `@1` | adapters + executor |
| Cache reuse decision | `CacheReuseDecision@1` | receipt cache |
| Model route decision | `ModelRouteDecision@1` | model route planner |
| Mutation ledger / set / fence / rollback | `mutation-ledger@1`, `mutation-set@1`, fence/rollback schemas | `MutationLedger` |
| Formal policy receipts | gate / rollout / override receipts | formal verification policy |
| Semantic state root manifest | `SemanticStateRootManifest` | semantic_state package |
| Source snapshot | `build_source_snapshot` | source_snapshot module |
| AAE campaign manifests | AssuranceManifest, MutationCampaignPlan, MutationExecutionReceipt, SurvivingMutantReport, AssuranceGap | **typed_unavailable** |

---

## 7. ZK surfaces and nonclaims

### VerificationCommitment is not ZK

`VerificationCommitment` (`contracts.py`, rev `0cdf81bd...`):

- `IS_ZERO_KNOWLEDGE_PROOF = False`
- Hash: `sha2-256`; domains `IVP-LEAF@1` / `IVP-NODE@1` / `IVP-EMPTY@1`
- Structural Merkle commitment over admitted receipts
- Signed receipts are not proof of execution without a trusted issuer
- Structural validation is not cryptographic validation of underlying execution
- **Cannot** substitute for released full-checkpoint or delta proof sealer APIs

### ProgramAnalysisZKP (scoped, non-authoritative)

Source: `proof/program_analysis_zkp.py` (rev `ab984496...`).

Scope statement: trace validity proves only commitment openings and supported
deterministic trace transitions terminating in the committed supported result.

`TRACE_VALIDITY_DOES_NOT_PROVE`:

- `inventory_completeness`
- `translator_soundness`
- `arbitrary_runtime_semantics`
- `theorem_beyond_committed_supported_result`

Public artifacts default `semantic_proof=false`, `authoritative=false`,
`trust=non_authoritative`.

### Datasets ZK attestation adapter

Source: `proof/ipfs_datasets_zk_attestation.py` (rev `16a81917...`).

`DatasetsZkStatus`: `attested`, `generated`, `simulated`, `unavailable`,
`degraded`, `not_applicable`, `pending_review`, `rejected`, `error`.

Only **`attested`** is authoritative.
`simulated_attestation_cannot_satisfy_attested` is enforced.
Receipt-bound attestation is **not** an `IncrementalProofSealer` campaign seal.

### IncrementalProofSealer (missing)

Not found on this tree:

- `IncrementalProofSealer`, `FullCheckpointSeal`, `DeltaSeal`
- `create_full_checkpoint`, `publish_full_checkpoint`
- `build_delta_seal`, `publish_delta_seal`

Disposition: **`typed_unavailable`**. An AAE-local proof system or IVP
commitment masquerading as a seal is forbidden. When released, seals still
cannot establish repository correctness, mutation-set completeness,
specification completeness, or direct execution unless the underlying proof
does. Campaign receipt signatures and seals must not overclaim each other.

---

## 8. Isolation and resource seams (summary)

| Seam | Path | Revision | Summary |
| --- | --- | --- | --- |
| Worktree | `semantic_state/worktree.py` | `f1b5d157...` | Detached fenced patch worktrees |
| Process runner | `verification/process_runner.py` | `52e8fe17...` | Deny-all network, hermetic env, tree fencing |
| Resource admission | `runtime/resource_scheduler.py` | `6d2c27b9...` | Host/provider leases and rejection kinds |
| Provider gate | `semantic_state/providers.py` | `644ad9df...` | production/development/simulated |
| Lifecycle fencing | `control/lifecycle_orchestrator.py` | `a9c25754...` | Supervisor process transition fencing |
| Root CAS | DurableSemanticStatePort | (semantic_state durable) | Candidates without advancing production root |

Plan-aligned AAE defaults: mutants network-disabled without production
credentials; controlled fakes/sandboxes; no mutation of trusted keys, policy,
verifier, benchmark oracle, or promotion authority except explicit verifier
fixtures; no public network service required.

---

## 9. Missing AAE public surfaces

Package `ipfs_accelerate_py.agent_supervisor.adversarial_assurance`: **absent**.

Planned Python APIs (all `typed_unavailable`):

```text
create_assurance_manifest
generate_mutation_candidates
predict_detection_set
execute_mutation
classify_mutation_outcome
diagnose_surviving_mutant
analyze_vacuity
propose_gap_remediation
evaluate_remediation
promote_assurance_policy
plan_mutation_campaign
execute_mutation_campaign
```

Planned CLI family: `assurance mutate plan|run|target|explain`,
`assurance gaps`, `assurance vacuity`, `assurance remediate`,
`assurance evaluate-remediation`, `assurance promote`, `assurance report`,
`assurance benchmark`.

---

## 10. Benchmarks and honest statuses

### Incremental verification (RED)

- Source: `benchmarks/agent_supervisor/incremental_verification.py`
  (rev `eb8e2a2f...`)
- Report: `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md`
  (rev `8c7800ced...`)
- Status: **red, non-authoritative**
- 20 cases; 1 seeded false negative; 7 false positives; real provers
  unavailable; 40% frontier/human escalation (`8/20`);
  zero controlled false negatives: **red**; cross-tree unaffected reuse: unmet

### Semantic compression harness (offline only)

- 40 tasks; median context reduction 58.9%; precision 36.22%; recall 100%;
  34 pass / 4 reject / 2 escalate; **zero** production-eligible rows
- Must not be relabeled live assurance

### Other honest flags

- Simulated statuses never production
- Commitment not ZK
- Adversarial assurance package absent
- Incremental proof sealer absent

---

## 11. Known blind spots

**Semantic analysis:** dynamic dispatch/imports, reflection,
descriptors/decorators, `eval`/`exec`, metaclasses, monkey patching, pytest
plugin collection, generated/native bindings, runtime generation,
configuration/fixture discovery, renames, uncontrolled I/O.

**Verification selection:** seeded IVP false negative; false positives under
broader/full suite; opaque/dynamic critical edges force expansion or human
review; real provers unavailable on RED snapshot.

**Vacuity:** only a narrow heuristic `is_vacuous_statement` /
`_check_non_vacuity` path for proof candidates; no integrated assurance
vacuity campaign.

**Campaign engine:** no integrated semantic mutation campaign, general
equivalent-mutant analyzer, expected-detection engine, assurance-gap taxonomy,
or authorized AAE promotion flow.

**Sealing:** IncrementalProofSealer public Python APIs absent.

**Authority matrix pin drift:** live datasets gitlink
`fbd1ba9f70803de157622bb20e22595ef09d606f` may disagree with frozen SCG
authority-matrix expectations; disposition is a known baseline blind spot for
`AAE-005`, not AAE success and not permission to weaken checks.

---

## 12. Collaborator pattern (how AAE should reuse accelerate)

For each admitted mutant, reuse existing accelerate seams rather than
reimplementing them:

1. Materialize mutant in an `IsolatedWorktree` under resource admission.
2. Derive invalidation / selection → `create_verification_plan`.
3. `execute_verification_plan` with `VerificationProcessRunner` + receipt cache.
4. Compare incremental vs full-suite costs via evaluation helpers when policy
   requires measurement.
5. Minimize survivors with `CounterexampleMinimizer`; pack diagnosis context
   with `pack_context` / `ContextPacker`.
6. Route any model-assisted remediation draft with `ModelRoutePlanner` /
   provider gate (never auto-promote).
7. Optionally record lineage in `MutationLedger` without treating it as the
   campaign engine.
8. Seal only through released sealer APIs when present; until then report
   `typed_unavailable`.

Do **not** introduce another semantic index, dependency graph, capsule
compiler, context packer, CID implementation, proof cache, receipt envelope,
scheduler, proof/ZK system, provider, or MCP++ profile.

---

## 13. Docs authority

| Path | Role |
| --- | --- |
| `docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md` | AAE program plan (operator-protected) |
| `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` | IVP release report with RED binding |
| `docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md` | Harness release narrative |
| `docs/architecture/adversarial_assurance_engine.todo.md` | Board task definitions including AAE-001 |

---

## 14. Acceptance checklist

- [x] Exact exports and signatures bound to source paths and revisions
- [x] Status vocabularies recorded (verification, cache, route, mutation, policy, ZK)
- [x] Manifests and receipts catalogued; missing AAE artifacts typed unavailable
- [x] Tests bound with last-modifying revisions for primary paths
- [x] Isolation and resource seams path- and revision-bound
- [x] ZK nonclaims explicit (commitment, program ZKP, datasets attestation, missing sealer)
- [x] Known blind spots and RED benchmarks recorded honestly
- [x] Inventory bound to inspected commit `3256d8bb813173349388a931e77d8ecf4bbcb8ef`
