# Legal autoformalization convergence plan

Create a source-faithful, versioned legal corpus that connects complete legal text to typed statements, scoped entities and definitions, and reviewed doctrine derivations in Lean 4. Extend the existing datasets semantics, trained decoders, entity cache and Lean backends. Use the accelerator agent supervisor for dependency scheduling, isolated implementation, bounded numerical work, independent verification and recovery.

The implementation board contains **10 goals, 30 subgoals and 60 tasks**, plus the root objective. All tasks are initially `todo`. The task graph is a plan for implementation; no supervisor, training run, task import or publication is started by creating these files.

## Artifacts and execution boundaries

| Artifact | Purpose |
| --- | --- |
| [goals.objectives.md](goals.objectives.md) | Root, goal and subgoal hierarchy in the existing objective-heap grammar |
| [tasks.todo.md](tasks.todo.md) | Native implementation-supervisor task projection with dependencies, ownership and acceptance criteria |
| [tasks.json](tasks.json) | Authoring contract for the same tasks, policies and resource profiles; this is not claimed to be the supervisor's native JSON ingest schema |
| [campaign.json](campaign.json) | Parallel execution template and initial ready tasks; resource and result adapters still require qualification |
| [evidence_baseline.json](evidence_baseline.json) | Observed repository commits, evidence hashes, cache pins and checkpoint anchors |
| [validate_plan.py](validate_plan.py) | Offline hierarchy, DAG, projection and optional native-parser validation |

`ipfs_datasets_py` owns legal meaning, labels, typed IR, model tasks, graph semantics, lowering and legal-source acceptance. `ipfs_accelerate_py` owns operational execution, provider capabilities, leases, resources, task evidence and merge coordination. The supervisor cannot confer legal meaning or proof authority through successful process execution.

Existing August PGIR campaigns and other concurrent boards remain separate. Reuse their qualified implementation where appropriate, but do not copy their old source authorities, training restrictions, provider choices or runtime state into this campaign. LAC-001 reconciles the current accepted trees and active work before establishing a new isolated integration target.

## Current training and implementation baseline

The October 13 observed datasets tree is `9eb06b73658d8203628a697db6199f14ecf479e4`, including the temporal-adapter comparison merged after the split-feature and permission-retention experiments. The external datasets checkout still has an October 1 HEAD; HEAD alone is also insufficient for the modified accelerator checkout. These are observations, not permanent launch pins. Recheck effective source bytes and newer work at dispatch.

Preserve the June consensus teacher, restart12, current sparse students, AF-029 proof-head models, source-conditioned formula models, joint latent heads, 384D/768D formula sidecars, native 4096D experiments, grouped source decoders and the source-pointer continuation lineage. Different widths, inputs, codecs, parameter groups and training populations are not interchangeable.

| Latest source-span model work | Observed result | Consequence |
| --- | --- | --- |
| Original source-only parent | 400 historical updates; older regression 96/96 | Retention control, not a universal semantic teacher |
| Structured continuation | Genuine model/Adam continuation already exists | Reuse `structured_span_continuation`; do not implement resume from scratch |
| Eight-cell actor/scope/modality curriculum | Best joint authored evaluation 52/72; older retention 94/96 in the all-factor cell | Preserve candidates and permission regressions; no promoted default |
| Four-cell ranking/presence comparison | Candidate coverage improved without reliable final selection; permission regressions remain | Diagnose assignment rather than repeat a broad auxiliary-weight sweep |
| Frozen span-head comparison | Older retention 96/96, limited newer improvement | Frozen modality is a required control |
| Split features | Older retention 94/96; newer authored joint 32/72 | Two remaining older failures concern temporal presence |
| Temporal CE and margin adapters | Two completed 32-update fits; older retention 95/96; newer joint still 32/72 | Both remain unpromoted; CE-versus-margin is completed work |

The actual next temporal proposal is **token-attention pooling versus the existing mean-pooling CE control on frozen features**. It is implemented and evaluated as LAC-030/031. Its initialization must reproduce the parent before training; additional parameter count and compute must be reported.

All 384 scored diagnostic references in the latest comparisons are exposed and share authored template ancestry. Eight statutory source observations remain unreviewed and unscored. The latest span fits have latent conditioning disabled and zero external encoder calls. Retain these cohorts as regressions, not fresh statutory evidence. The matched native-vector study in LAC-034 measures actual autoencoder contribution separately.

The observed checkpoint hashes and immutable Hub revisions are in `evidence_baseline.json`. Rejected checkpoints may be replayed or used as explicitly declared research donors, but their inclusion in an inventory does not select or promote them.

## Meaning of correctness and global consistency

Every source unit receives an explicit current disposition. An accepted record must separately satisfy the applicable checks below:

1. **Source identity** — exact release, document, edition, physical occurrence, original bytes and offsets.
2. **Context closure** — all meaning-bearing antecedents, definitions, scope provisions, exceptions, authority and temporal origins are available or represented as open obligations.
3. **Source fidelity** — independent authenticated review of the exact source/context/target, or an independently qualified proof-carrying checker for a restricted grammar applied to that exact record. A template match alone cannot admit meaning. Model agreement, structural validity and round-trip equality cannot substitute for this check. Checker-derived evidence is never mislabeled as per-record human review.
4. **Typed representation** — all supported roles, relations, negation, modality, quantities, binders, operators and attachments are valid and complete. Unsupported meaning remains a gap.
5. **Translation preservation** — supported lowering preserves the declared semantics; unsupported constructs cannot become opaque atoms merely to compile.
6. **Checked formal result** — exact declarations or theorem propositions, imports, premises and toolchain are checked by the appropriate backend. Compilation, theorem checking and source fidelity remain separate outcomes.
7. **Scoped consistency** — the complete accepted dependency theory is checked for its explicit jurisdiction, legal time, assumptions and semantic profile.

Legal sources can contain real conflicts and competing interpretations. The unified corpus stores these faithfully in separate named theories or unresolved conflict records. It must not force every historical rule and interpretation into one unconditional classical conjunction or delete a rule to manufacture consistency.

A duty can be violated without making the factual world logically inconsistent. Permissions have different strengths; a permission/prohibition collision is profile-dependent. Different conditions can be compatible. An exception or priority field resolves nothing without evidence that it applies to the particular collision. Unknown facts are not false. Inconsistent premises, impossible applicability and empty categories must not make vacuous proofs appear to establish source meaning.

Consistency results are `checked_consistent_within_scope`, `conflict_witness`, `unknown`, or `unsupported`, with exact closure, assumptions and bounds. Solver timeout, an exhausted frontier or missing referenced section never becomes a consistency pass. Vacuity is reported separately: do not invent existence assumptions or reject a faithful conditional norm merely because its applicability can be empty. No finite test panel certifies every arbitrary interpretation of natural language.

## Goals subgoals and tasks

| Goal | Subgoals | Tasks | Completion evidence |
| --- | --- | --- | --- |
| G010 Preserve current evidence and identities | Source/archive reconciliation; training lineage/replay; execution pins | 001–006 | Reconciled source denominator, immutable run DAG, exact effective-tree and capability manifests |
| G020 Establish semantic supervision | Complete context/classification; independent review/admission; exposure-safe partitions | 010–015 | Authenticated source/context reviews, representability labels, split closure and fresh semantic references |
| G030 Represent complete meaning | Typed semantics/evidence; scope/multiple statements; coverage/oracles | 020–024 | Lossless typed AST/codec, explicit attachments and adversarial source-preservation checks |
| G040 Improve decoding | Temporal retention; role/qualifier assignment; multi-rule/latent/rejection | 030–036 | Bounded matched experiments, casewise retention, fresh statutory metrics and explicit selection/no-go |
| G050 Aggregate semantic graphs | Mention/concept identity; definition closure; composition/invalidation | 040–046 | Source-bound semantic assertions, complete required dependencies and AST/graph parity |
| G060 Check scoped global consistency | Applicability/authority semantics; witnesses/nonvacuity; global composition | 050–055 | Checked collisions and resolutions, countermodels, closure/SCC certificates and honest unknown outcomes |
| G070 Compile unified Lean corpus | Semantic lowering/modules; kernel/axioms; preservation/rebuilds | 060–064 | Exact semantic builds, proof/assumption audits and clean/incremental dependency equivalence |
| G080 Compose doctrines | Source/authority; derivations/alternatives; evaluation/change impact | 070–075 | Reviewed doctrinal propositions, explicit premises/defeaters, supported checked consequences and contrary authority |
| G090 Execute parallel campaigns | Native transport/completion; resource/isolation; priorities/recovery | 080–085 | Actual parser/adapter qualification, shared admission, fenced results and interruption tests |
| G100 Converge through releases | Connected pilots; corpus coverage/freshness; reproducibility/gaps | 090–095 | Full disposition accounting, connected statutory acceptance, immutable build/release roots and remaining goals |

IDs in this table abbreviate `LAC-G` goals and `LAC-` tasks. Every task in the companion board contains its objective, dependencies, existing owners, write scope, output path, acceptance criteria, resource class and independent validation requirements.

### Source and supervision work

LAC-001/002/003 are immediately ready for parallel inventory work. They freeze current effective source, retain all training ancestry, and reconcile the pinned span/entity inventories without a model run.

The historical all-span scan reports 3,840,988 regenerated occurrences, 64,969 exact archive joins and 378,935 unresolved archived IDs. These are distinct populations from 443,904 archived IDs and from paired observations/formula occurrences. LAC-003 revalidates them and continues the existing identity recovery. Same text or citation does not imply the same occurrence. Accepted aliases must be separately evidenced; original IDs remain preserved.

LAC-010/011 extend existing context packets and coherent spans. Preserve original UTF8, exact offsets, headings, list lead-ins and whole source sections. Normalized sentence hashes are not byte-roundtrip originals. A residual/source-retrieval route may preserve original text, but must be scored separately from learned semantic reconstruction.

LAC-012 extends the existing review recorder. It adds authentication, independence, competence scope and an explicit admission decision; it does not invent a replacement recorder. Independent first submissions remain immutable before adjudication. Missing review, reviewed absence, ambiguous reading and competing interpretations remain distinct.

The initial review cohort is small enough to complete: approximately 16 independent section neighborhoods and up to 32 spans, expanded by actual dependencies and construction coverage. It is a bootstrap corpus, not a sample large enough for an all-law accuracy claim. Review capacity and fresh grouped confirmation are prerequisites for semantic claims, not barriers to implementing codecs or constructed fixtures in parallel.

Split documents, amendments, overlapping spans and interpretation derivatives before target fitting or retrieval indexing. Definition-connected and cross-reference-connected components can be large; use documented boundary and context-only policies instead of claiming impossible partition independence. Every held-out context exposure must be disclosed.

### Typed statements and model improvement

Reuse canonical statement scope, symbol bindings, ExtendedLegalIR, logic slices and the current formula codecs. Add supported representations in stages:

- Complete single norms with exact roles, polarity and optional qualifiers.
- Ordered coordinated norms with shared arguments and scoped qualifiers.
- Quantified and nested Boolean statements, temporal ownership and relational terms.
- Scoped definitions, inclusions/exclusions and incorporated references.
- Constitutive rules, powers and procedures under explicitly supported semantic profiles.

An AST must represent the complete supervised target before fitting. Do not truncate targets, choose a convenient clause, hash semantic identifiers into irrecoverable buckets or relabel checkpoint metadata to broaden support.

The training lanes run independently after shared target/split contracts seal. Temporal pooling tests LAC-030/031; complete role assignment LAC-032; qualifier presence/boundary/ownership LAC-033; authentic latent contribution LAC-034; multi-rule and definition generation LAC-035. Keep losses, raw scores, selected assignments, attempted endpoints, chosen checkpoints, optimizer history and invalid/abstained cases separately attributed.

Screen one finite cell first. Replicate a meaningful finalist across predeclared seeds before opening confirmation. Keep all exposed panels as retention controls, including the 96 older exact cases and 32 older permissions. Require 96/96 legacy raw/joint retention for the temporal research gate; do not patch a failing diagnostic ID. Fresh statutory promotion criteria additionally require independent review and supported complete semantics.

The latent experiment must train matched decoders on authentic representation profiles. Raw384, core384 and trained384 are different inputs. Source-only versus zero/disabled/permuted vectors remain controls, and encoder cost is included. A vector-reconstruction gain is not automatically a legal-semantic gain. If a latent offers no measured advantage, preserve it as an auxiliary representation rather than asserting it causes formalization improvement.

### Span to entity graph composition

Extend `EntityCache`, scoped symbols, meta ontology and definition closure. Maintain separate source, candidate and accepted semantic views. A mention is an exact occurrence; a concept is a scoped denotation; a definition is a source-bound semantic assertion; a norm is a structured statement. A current officeholder and a statutory office are not interchangeable entities.

Use structured rule nodes or hyperedges that retain argument order, quantifiers, Boolean operators and local/shared attachments. Recommended edge roles include `mentions`, `denotes`, `defined_by`, `applies_in`, `has_argument`, `condition_of`, `exception_to`, `temporal_constraint_of`, `incorporates_definition`, `derived_from`, `lowered_to` and `checked_by`. These names are proposed ontology extensions, not claims of already-implemented APIs.

The current definition bridges support explicit bounded unary declarations. Extend them profile by profile to relational definitions, inclusions/exclusions and external definitions. “Means,” “includes” and deeming language cannot silently share one biconditional interpretation.

Persist dependency frontiers and strongly connected components. A finite job may pause at its resource limit and later resume; it may not declare the closure complete because a hop cap was reached. Accepted graphs require all mandatory references resolved or an explicit conditional theory with open assumptions.

### Entity graph to global theory and doctrine

Repair the current conflict heuristic before using it as admission evidence. Candidate matching must include typed actor/action/object or relation arguments, then test joint applicability and legal-time/jurisdiction overlap. Require actual exception, priority or authority witnesses. Keep minimal conflicting premise sets and countermodels for independent review.

Check graph shards locally, then check their complete dependency closure and interface assumptions. Local consistency alone does not establish global consistency. Use incremental closure certificates and full bounded replay on the pilot to verify composition. A change to a shared definition reopens every dependent rule, theory, theorem and doctrine.

Doctrines are a new explicit composition layer. Start with a bounded statutory doctrine grounded in reviewed connected provisions. Every claim records premises, substitutions, inference steps, applicability, exceptions, authority, alternatives and proof obligations. Clustering and retrieval suggest topics but cannot assert doctrine.

Case-derived doctrine needs separately admitted full judgments and reviewed holding/dictum, procedural posture, court, jurisdiction, date and subsequent-treatment bindings. Do not infer precedent strength from embedding similarity or present a statutory derivation as a court holding. Preserve contrary authority and competing theories. Checked consequences remain conditional on reviewed source interpretations and stated premises.

### Unified Lean corpus

Extend existing family routes, source-family/definition/norm lowerings, native emitters and kernel checks. Use stable namespaces for jurisdiction, instrument, edition, interpretation profile and theory; derive module dependencies from the semantic graph.

Separate definitions/norm declarations and source-interpretation assumptions from derived theorems. `lake build legal` or the exact declared target must execute on current generated artifacts. Audit expected modules and transitive assumptions with the existing kernel backend and `#print axioms`. Reject `sorryAx`, admissions and unapproved custom axioms; explicitly declare permitted standard logical axioms for each profile.

Fingerprint-based statute/term units remain useful identity fixtures but receive no clause-proof credit. Opaque string transport and compilation do not establish semantic preservation. Every accepted semantic constructor needs a supported translation and its preservation obligation.

Compare clean and incremental builds after source/graph changes. Cache results only when source, AST, premises, imports, semantics, code, toolchain and policy identities match. A content hash proves identity, not legal truth.

## Parallel execution and convergence order

The board's lanes are logical workstreams, not promises of hardware concurrency. Start with **three implementation workers and one independent validation role**, subject to the current shared capacity ledger. Numerical jobs default to one; parallel model/prover cells require measured capacity and separate leases. Reviewer capacity is external and cannot be inferred from available CPU.

| Wave | Work that can overlap | Exit condition |
| --- | --- | --- |
| 0 | Current-source authority, training lineage, cache denominator | LAC-001/002/003 evidence sealed; no historical replacement |
| 1 | Context/review contracts; typed IR; supervisor transport/resources; baseline replay | Exact source/runtime and shared semantic interfaces are usable |
| 2 | Temporal implementation; role/scope training preparation; graph schema/definition work; consistency semantics; Lean contracts | Independent owners can exchange immutable typed fixtures |
| 3 | Bounded training cells; entity closure; solver checks; Lean modules; doctrine source review | Complete per-layer evidence and an early constructed end-to-end trace |
| 4 | Connected statutory pilot and independent replay | Accepted source-to-graph-to-Lean/doctrine trace or explicit partial/no-go |
| 5 | Section/title dependency shards; amendment invalidation; release preparation | Full disposition accounting and unchanged correctness gates at scale |

LAC-090 wires an early constructed neighborhood before broad numerical work or corpus rollout. It exposes interface failures quickly without calling fixture success statutory qualification. LAC-091 then uses a reviewed real-law neighborhood and the best supported route; a deterministic route can establish a valid pilot even if no learned model wins.

Workers receive isolated worktrees and immutable input manifests. One shared resource-admission owner coordinates independent supervisors; in-process leases alone are insufficient. GPU and checkpoint writers are exclusive when used. Shared semantic schemas, corpus roots, graph admission, model promotion and release pointers have single writers with compare-and-swap. Independent checker/reviewer results are never accepted by the producing worker alone.

Prioritize missing high-fanout definitions and context before isolated low-impact spans. Then prioritize critical semantic failures, unsupported frequent constructions, uncertain accepted candidates and high-cost repeated failures. Stable tie breaks and priority features are recorded. Limit attempts to three per task/input and no-progress refills to two; exhausted work becomes a named capability or review gap, not an infinite retry loop.

Resource profiles in `campaign.json` are proposed admission ceilings based on existing bounded runs. They are not reservations. Actual telemetry, retained storage, task preparation cost and existing foreign reservations govern admission. A requested profile that does not fit waits or is revised as a separately reviewed experiment; it does not silently raise the shared cap.

## Supervisor handoff

Validate the planning artifacts without importing tasks or starting services:

```bash
python3 -B implementation_plan/legal_autoformalization_convergence_20261013/validate_plan.py

python3 -B implementation_plan/legal_autoformalization_convergence_20261013/validate_plan.py \
  --accelerate-root /home/barberb/lift_coding/external/ipfs_accelerate \
  --require-native-parser
```

The optional check invokes the actual task and objective parsers in a child that rejects file writes, network access and subprocess effects. Structural validation does not qualify future numerical or semantic outputs.

The initial ready tasks are **LAC-001, LAC-002 and LAC-003**. Bootstrap implementation uses the existing generic supervisor in isolated worktrees. Its finite inventory/code-only allowlist is LAC-001/002/003/005/006/080/081/082, with only dependency-ready tasks dispatched. LAC-080 verifies actual transport and metadata preservation; LAC-081/082 install and qualify thin result/resource adapters for the later campaign jobs. LAC-081 first binds generic result envelopes; legal-specific checkers are registered after LAC-020. These bootstrap tasks must not depend on their own unimplemented adapters. Numerical replay LAC-004 explicitly waits for source/tool pins and resource admission.

Do not feed `tasks.json` directly into a generic ingest endpoint. The current generic JSON normalization can drop output fields, and some validate/import paths mirror a meta index. The authoritative planning projection is `tasks.todo.md`; LAC-080 binds the exact supported transport and CLI of the selected runtime before import. Repository metadata alone does not select a worker checkout. Materialize repository-specific boards within their pinned roots, or a tested umbrella projection with qualified managed-submodule paths. Native parsing alone does not enforce custom lease, resource or semantic-review metadata. File-level lease keys are derived from every write path; remaining shared-code bootstrap work is serialized until those claims are executable.

Task validation fields currently describe behavioral acceptance. Before a task starts, its worker freezes exact executable commands, owner tests, artifact paths, expected coverage and independent acceptance routing into its job packet. Do not execute descriptive validation strings as shell commands or use plan validation as evidence that implementation succeeded. If a command or tool is unavailable, the task receives a capability blocker rather than a fabricated pass. A producer cannot change the protected board to redefine its own acceptance.

Once LAC-080 materializes a native scheduler configuration in a clean tracked integration repository, the supported configured-board route is:

```bash
python -m ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler \
  --repo-root <integration-repository> --config <native-scheduler.json> preflight

python -m ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler \
  --repo-root <integration-repository> --config <native-scheduler.json> \
  launch --implement --dry-run --duration-seconds <finite-duration>
```

The angle-bracket values are required materialized inputs, not runnable example paths. `campaign.json` is not that native scheduler configuration. The validator accepts `--check-all` for preflight compatibility. Preflight executes the declared validator; only the qualified read-only validator should be supplied. Actual bounded launch is a subsequent execution action after source, commands, provider, claims and completion adapters pass.

Numerical children receive only approved TRAIN targets and declared development references. Confirmation target files and target-bearing context bundles remain with an independent scorer until candidates and policy freeze. Access to an upstream result identity does not grant blanket access to its payload; role-specific packets enforce this separation.

October 13 repository-planning freshness and provider-resource admission additions should be inspected during LAC-001/080/082, including commits `24bafa1179` and `76b31227dd` in their retained worktrees. Their presence in history does not establish that this controller has installed their consumer adapters.

## Release gates and progress measures

| Level | Required acceptance | Reported gaps |
| --- | --- | --- |
| Source unit | Exact identity, complete context and semantic disposition | Missing source/context, editorial fragment, unsupported construction |
| Statement | Reviewed interpretation, complete typed meaning and grounded attachments | Omission, role/modality/scope error, alternative reading, abstention |
| Entity graph | Scoped identity, source-bound assertions and complete mandatory closure | Ambiguous entity, definition conflict, unresolved dependency, stale root |
| Legal theory | Checked applicability/authority and scoped consistency with nonvacuity | Conflict witness, unknown solver result, unsupported semantics |
| Lean artifact | Supported lowering, current build, theorem/premise/axiom evidence where claimed | Translation gap, failed build/proof, unapproved assumption |
| Doctrine | Reviewed authority/meaning and checked supported derivation | Missing premise, contrary authority, unsupported inference, unresolved dispute |

For each frozen generation, reconcile exclusive source dispositions: accepted, review pending, missing context, ambiguous, unsupported, conflicting, technical failure or superseded. Track candidate coverage and admission evidence independently, so an item may have a generated proposal without being accepted. Count occurrences, observations, assertions, theorems and doctrines separately.

Promotion and rollout require zero missing mandatory evidence and zero unresolved critical regressions for the declared accepted scope. All accepted source units must satisfy record-level gates; fresh held-out evaluation estimates model generalization and informs routing, not automatic correctness of every future output. Confidence/calibration thresholds and statistical acceptance margins are sealed from development before confirmation rather than invented after results.

Track validated coverage gained per CPU/GPU/prover/reviewer hour, dependency fanout closed, contradiction resolution rate, abstention precision, semantic error severity, stale dependent count and successful restart/rebuild rates. Report actual costs including encoding, context preparation, verification and retained storage. Do not report repeated row presentations as unique spans formalized per second.

Prepare append-only releases of the accepted graph, Lean modules, source/interpretation bindings, evaluation evidence, model checkpoints where applicable and complete known-gap inventory. Preserve older releases and original failed candidates. No release is described as all-law complete while mandatory source units, required dependencies, authority disputes or unsupported semantics remain unresolved.

The root objective closes only for the exact declared corpus and supported theory profiles when all required evidence is current. Otherwise the final result is a measured partial corpus and a new bounded residual goal graph. This is the convergence mechanism: every iteration increases qualified coverage or identifies a specific unresolved dependency without weakening correctness.
