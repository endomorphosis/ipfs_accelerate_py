# Token economy while retaining LLM planning

The user wants the supervisor to **keep the LLM planning session** and use
DuckDB/Quack/DuckLake metadata, formal logic, semantic capsules and minification
to reduce **total planning plus coding tokens**. Removing planning was the
wrong primary experiment. Its result remains a historical ablation; it is not
the requested optimization. The normal direct planning route remains available
and was never globally disabled. Future context experiments must retain both
the planning and coding routes through `llm_router`.

The desired architecture is **catalog-backed, progressively disclosed context**:
retain complete native records, source bindings and certificates locally;
select the facts needed for the current decision; send a compact checked view
to each LLM; expand specific handles when the agent needs details. Formal
systems should constrain and inform the LLM's work, rather than replace its
planning session. The implementation sequence and measurement gates below
make that goal concrete without claiming an unrun token-saving experiment.

## Evidence from the existing path

The [planning audit](planning-audit.json) and [coding audit](coding-audit.json)
independently bind current source and retained metadata. The historical
successful direct supervisor used **291,432 tokens** across one planning and
one coding router/CLI session. Its **285,558 input tokens** already include
**244,992 cached tokens**; output was **5,874 tokens**. Only **40,566 input
tokens** were uncached. Cached prefixes still contribute to the user's total
input-token objective. Session counters include provider harness context and
internal turns, not just the initial supplied prompt.

| Existing component | Historical direct run | Historical symbolic ablation |
| --- | ---: | ---: |
| Planning model input | 23,292 bytes | No planning LLM; excluded as primary comparison |
| Native coding input | 62,519 bytes | 60,092 bytes |
| Existing reversible semantic transport | 55,076 bytes | 52,649 bytes |
| Doctor advisory appended afterward | 2,109 bytes | 2,109 bytes |
| Public instruction and additional context appended | 1,226 bytes | 20,076 bytes |
| Workspace advisory appended | 890 bytes | 890 bytes |
| Complete coding model input | 59,301 bytes | 75,724 bytes |

The semantic transport reduces representation size by **7,443 bytes** in each
run. That combines removal of nested JSON escaping/chunk overhead with typed
identifier aliasing; it is not entirely an alias-dictionary saving. The complete
direct input is only **3,218 bytes smaller** than its native input. The symbolic
ablation's complete input is **15,632 bytes larger** despite its smaller
translated core. This makes measuring the **complete assembled model input** a
first priority. Bytes are not measured provider tokens.

The original public instruction, `eigen.py` and public `eval.py` total only
**3,604 bytes**. The retained coding capsule carries a semantic context body of
31,939 bytes, retrieval context of 4,832 bytes and world context of 6,425 bytes.
For this tiny task, repeated administrative data is a larger opportunity than
whitespace-only source minification. The task-specific IntentIR append repeats
source units, atoms, task bindings, scope, validations and authority fields.
Its 20 KB size partly arises from the experimental requirement contract, so it
must not be described as the default direct route's overhead.

## Existing components to reuse

| Component | Current role | Missing link for token economy |
| --- | --- | --- |
| DuckDB AST, semantic, BM25/vector, proof and meta-index catalogs | Store and join source facts, identities, locators and task context | A compact, task-scoped projection of those records for both LLM phases |
| Quack owner/query boundary | Qualified native supervisor state and retrieval access | Bounded read-only lookup of a selected handle through the existing authenticated boundary |
| DuckLake projections | Retained historical ranges, cursors and provenance | Changed-record/context deltas; history must not be treated as current scheduling or proof authority |
| Producer semantic capsules and selection | Query-selected signatures, effects, dependency references and source bindings | Smaller decision views and proof/status cards instead of repeating complete metadata |
| Semantic router translation | Exact representation reconstruction, repeated typed ID aliases, literal source preservation | Non-expansion/token guard, broader typed slots and a versioned controller-held alias table |
| ContextCompiler decision, delta and retry capsules | Completeness/closure evidence, parent bindings and content-addressed expansion | Integration with the actual terminal planner/coder input path |
| DeltaTaskPacket | Unresolved frontiers, effects and validation budgets | Active terminal integration and accounting for the complete provider envelope |
| Formal provers and dependency analysis | Check supported obligations and reuse scoped proof results | Compact applicable constraints, counterexamples and unresolved work in the model view |

These are different responsibilities. A catalog locator alone is not a working
model tool. A stored certificate does not prove a new source revision. An
existing delta compiler does not demonstrate that the coding CLI's internal
conversation consumes deltas. The independent audits distinguish implemented
code from missing terminal integration.

## Implementation backlog, in order

### 1. Measure complete inputs while keeping the two sessions

Extend the existing `router_implementation_runner` receipts and
`terminal_context_audit` to account for each final-input component: request
core, response grammar, selected evidence, semantic/world/retrieval views,
alias dictionary, literal public instruction, IntentIR appendix, Doctor
residuals and workspace contract. Include complete assembled input hash/bytes
and a declared tokenizer estimate when available. A local tokenizer estimate
must name its encoding/version and remain distinct from observed provider
usage; `bytes / 4` is not an exact count for CIDs, code or multilingual text.

Collect per-turn input/cache/output counters when the provider's native records
permit it. Preserve unknown totals and observed failed-call subtotals. Record
tool-result sizes, expansions, repeated-prefix growth and retries without
publishing source/solution bodies or private reasoning. Count tokens spent on
retrieval and extra turns as part of the task total. This instrumentation is
required before predicting end-to-end savings from a shorter first prompt.

The pilot must keep one LLM planning session and one coding session, with the
same router, model, CLI, reasoning, task profile, public inputs, resource limits,
time allowance and retry policy. Do not select the version-2 symbolic planning
contract as the primary treatment. If source-bound IntentIR coverage is used,
the version-1 provider coverage route still invokes LLM planning. The new
treatment should be the context representation, not the session count.

### 2. Build model views after native validation, before router dispatch

Use the existing meta-index and native semantic view to select source facts by
task/output scope, entrypoint, dependencies and public validation requirements.
The controller should retain complete manifests, exact CIDs, signatures,
certificate bodies and admission receipts. The model view should retain:

- The original public instruction once, plus uncertainties or ambiguities.
- Permitted edits and validation commands, including benchmark limitations.
- Relevant entrypoint signatures, interface/effect facts and needed source.
- Applicable constraints and unresolved correctness/performance obligations.
- Compact local handles with explicit expansion availability and omissions.

The planner hook is after source and initial-context verification in
`terminal_indexed_preparation`, before `build_prompt_goal_provider_request`
dispatch. The coding hook is after semantic/world/retrieval loaders verify
their native records and before `router_implementation_runner` assembles the
complete model input. Original parser, critic, formal compiler, effect checks
and native admission continue to use the complete records.

For planning, replace duplicate acceptance/declaration prose with one exact
task declaration and shared acceptance/validation references. Represent the
candidate output grammar concisely, while retaining the full native JSON
Schema and strict validator controller-side. Do not omit required response
fields merely to make the request shorter. References in model replies must
expand before the current strict parser and normal plan checks.

For coding, reuse ContextCompiler decision-context and completeness/closure
machinery. Selection should include dependencies and reasons for omission,
not only a vector top-k. A source slice needed for an edit remains literal or
must be explicitly fetched before editing. Raw source need not be repeated
both as capsule content and as escaped evidence summaries.

### 3. Deduplicate structured data and extend existing minification

First separate exact representation changes from semantic selection. Reversible
compaction can share identical source units, native atoms, outputs, validations,
root references and false-authority records in tables. The full native object
must reconstruct exactly. Keep paths, Python names, string literals, comments,
instruction wording and actual policy decisions unchanged. Renaming program
identifiers or stripping source comments is a separate source-transformation
problem and is unnecessary for the first context pilot.

A proposed version-2 extension to `semantic_router_translation` can keep the
full alias-to-CID map in the controller-held immutable translation table and
give the model short handles plus task/root/scope bindings. Opaque identity
strings generally do not help the model reason about eigenvalues. This is a
new transport, not a claim that current code already omits the map. Verify
exact reconstruction, reserved-marker collisions, unknown handles, wrong roots,
stale source and ambiguous mappings. A new envelope must be chosen only when
its complete token estimate improves on the native alternative; a shorter
core can be defeated by mapping instructions and appended metadata.

For the IntentIR appendix, use one source-unit table, one task-scope/validation
table, one explicit authority block and atom references. The same public clause
and validation should not be restated in every requirement row after the
verbatim instruction. For Doctor advice, share evidence/status entries and
report the actual remaining work, while preserving its scope and unknown or
unsupported analysis. Keep one workspace contract. Restore and check native
objects at the controller boundary rather than asking the LLM to maintain
their administrative equality.

### 4. Use formal logic to shrink the reasoning frontier

Run the supported deterministic analysis and provers over current source and
candidate obligations. Send a compact status card rather than a full proof
transcript: claim, `proved`/`disproved`/`unknown`/`unsupported`, assumptions,
source/dependency revision, useful counterexample and local certificate handle.
Proof bodies and solver logs stay in the proof catalog. Do not repeatedly send
already discharged obligations; reinclude them when a relevant dependency or
assumption changes. Unknown and unsupported obligations remain visible.

For `largest-eigenval`, the model needs the general nonsymmetric input domain,
possible complex eigenpair, dominant modulus, nonzero vector, residual criterion
and performance requirement. Supported exact reasoning can reject an
unconditional symmetric-only solver or establish shape/index/precondition
facts. It does not certify the complete float/complex implementation or its
speed. A short unsupported-numerics status is more useful than repeated
administrative proof receipts or syntax-operator misses. All correctness and
timing obligations stay with planning and coding.

The LLM still chooses strategy, weighs tradeoffs and handles unresolved
mathematical/engineering work. Symbolic analysis reduces what it must rediscover
and prevents repeated invalid approaches; it is not used to bypass planning.

### 5. Add checked progressive disclosure and deltas

Provide a bounded local read-only lookup path for declared source, capsule,
proof and history handles. Resolve through the existing authenticated owner
boundary with task, repository, source head, semantic root, query and schema
bindings. Return selected fields or source spans, not database dumps, credentials
or arbitrary SQL access. If a handle is unavailable, expose that fact; a CID in
the prompt is not readable context by itself.

Use existing content-addressed context expansion and retry/delta APIs to send
changed source slices, new counterexamples, validation failures and changed
requirements. Use DuckLake history cursors to identify changed catalog records
while checking present source state separately. Stable cached prefixes can
help reuse, but cached tokens remain in total-input accounting.

Qualify the actual provider-session boundary before claiming later-turn
savings. The current CLI coding session owns its internal history and tool
outputs, and its initial input compressor does not trim those turns. If the
CLI does not support controlled history compaction, start with compact initial
views and a bounded local lookup/tool-result adapter. Do not pretend a fresh
session remembers an omitted parent capsule, or add frequent summaries that
cost more tokens than they save.

## Qualification and comparison protocol

| Gate | Required evidence |
| --- | --- |
| Source and scope | Same task/source/root bindings; currentness checks before selection and expansion |
| Lossless compaction | Exact reconstruction, native field equality, unchanged literals and instructions |
| Semantic selection | Closure/completeness witness, visible unknowns and omission reasons, expansion available |
| Proof reuse | Exact source/dependency/assumption binding; unsupported float/complex claims retained |
| Budget | Complete final prompt measured; map, wrappers, instructions, tool outputs and lookups included |
| Provider behavior | One planning and one coding session retained; per-turn accounting where observed |
| Task outcome | Original public/official checks, reward, native completion and lifecycle cleanup |
| Evaluation integrity | No hidden-source training, candidate answer reuse or benchmark-input modification |

Start with offline full-versus-projected rendering and negative controls for
stale/foreign roots, missing dependencies, corrupted references and invalid
proof reuse. Do not launch a model merely to measure a JSON size. First qualify
reconstruction and the actual bound-input collector. Then run the
planning-preserved context ablation, reporting complete input/output usage,
cached versus uncached input, context expansions, retries, wall time and
official reward. Avoid stacking several new projections in one experiment:
measure reversible transport first, then selected decision views, then lookup
and delta integration. Repeat matched arms only when needed to establish the
reliability/efficiency conclusion; a historical comparison alone is not causal.

No percentage reduction is claimed from this audit. The previous symbolic
trial's 12.17% total-token difference does not measure the requested
planning-preserved context optimization. Planner schema/CID savings and coding
metadata savings must be measured with both sessions retained, and reductions
must not be offset by more tool reads, more coding turns or failed validations.

The existing 8D spaCy, 384D and 768D representations can rank facts, candidate
operators and relevant source/certificate records. A possible Leanstral
representation can support retrieval of formal patterns. These are nominations,
not proof or omission authority; source/proof dependency checks determine what
can be reused. New embedding training is not a prerequisite for the first
deduplication and catalog-backed context experiment.

## Delivered scope

This change records the corrected goal, read-only source audits and
[bounded prompt/usage accounting](prompt-accounting.json). It makes no new
provider calls, official trials, training updates or production runtime changes.
Full prompt/source bodies, source-bearing contracts and model weights are not
published. Current code locations and hashes are retained in both audit JSON
files and the [artifact manifest](artifact-manifest.json); documentation checks
are recorded in [documentation gates](documentation-gates.json). The original
shared workspace's source and index remain preserved.
