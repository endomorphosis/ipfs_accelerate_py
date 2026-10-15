# Signed scope closes after working-set preflight, 2026-10-14

The router's instruction `@5` replay now follows working-set `request_page`,
world-packet checks and descriptor-context entry. It reloads the pinned public
artifact and replays the signed profile, complete signed originals in both
checkouts, baseline commits and compiled semantic selection. The full block and
receipt must match their initial values. Actual CWD must still be the allocated
worktree before provider-attempt flags change. Legacy instruction paths retain
their existing behavior.

## Reproduction and qualification

The unchanged published parent `b184d30229b3b8f23a390ca24580962d44620536`
reproduces the gap in [baseline-01](baseline-01/commands.json): one authored
post-page mutation to an omitted canonical signed file reaches the capturing
provider. Its retained invocation reports `router_calls=1` and
`terminal_semantic_scope_dispatch_rechecked=true`. The expected test failure
is preserved in [JUnit](baseline-01/junit.xml), [stdout](baseline-01/tests.stdout.log)
and [stderr](baseline-01/tests.stderr.log), with original source snapshots.
This is a controlled host regression, not a demonstrated deployed exploit.

The [first patched qualification](qualification-01/commands.json) passed 145 cases
across [eight complete suites](targets.json), including 18 new cases, with zero
failures, errors or skips. All source bindings stayed unchanged. Elapsed time
was 538.257 seconds; this is authored-test runtime, not production latency. The positive combined path expands the exact native capsules and
restores the original semantic transport, retaining required raw sources,
constraints and signed frontier counts. Fifteen post-page cases cover selected
bytes and executable modes, support, profile, omitted originals and baseline
HEADs in both checkouts, plus artifact digest/permissions and CWD. Two further
cases mutate state during descriptor-context entry. Every refusal must occur
before the capturing provider is called, with the dispatch flag false and
`router_calls=0`.

The merged qualification has 268 [listed bindings](source-inventory.json) are explicit: 217 accelerator
files and 51 datasets files. Actual datasets execution remains
`a0f60d42f836571a0023515e92dfb0e5724f0d79`; kit execution remains
`40c98c71b95be750509ffdb21e7e5494d91b4e79`. This inventory does not attest a
complete import, binary or toolchain closure. The [qualification runner](qualification_runner.py)
retains exact arguments, interpreter/library versions, bounded private runtime
paths, source hashes and raw outputs. Historical results and the deliberately
failed baseline control are not added to the patched run's count.

The [incoming world-replay review](incoming-world-replay-review.json) preserves
seven parallel paths and expands the source inventory to include three world
modules. [Qualification-02](qualification-02/commands.json) passed the same 145 cases in
eight complete suites at the merged source in 587.803 seconds, with zero failures,
errors or skips and all 268 bindings unchanged. The earlier pass and its
[original manifest](first-qualified-manifest.json) remain separate. Incoming
world-replay suites outside this roster are not counted.

## Real mechanisms and authored seams

Git, Ed25519 signatures, 73-source signed inventories, native owners,
compiled context capsules, linked allocation, source views and exact native
capsule paging are exercised. The new combined cases use an explicitly authored
same-process endpoint and container UID/descriptor route; their descriptor bytes
and page envelopes pass the existing closed protocol validators. Provider
replies and usage figures are authored. The existing working-set dispatch suite
also uses a real registered inert Python socket client under an authored UID
admission. This does not qualify a deployed coding worker or container boundary.

No external provider, embedding encoder, decoder trainer, solver, native proof
kernel, Docker run, target program or official task verifier is invoked. These
checks establish source/transport currentness at the checked boundary; they
create no proof, execution, completion, omission or settlement authority.
Sequential source rechecks create no filesystem writer lease. A concurrent
writer after the closing check remains a separate handoff concern. Native
claim/allocation and permission joins still need deployed-consumer qualification.

## Parallel work and next gates

The [parallel survey](parallel-review.json) found no equivalent post-working-set
replay in the seven inspected peer worktrees. Their separate candidate/world
routes may overlap textually and must be preserved during main integration.
No foreign worktree, model checkpoint, embedding inventory or runtime database
was modified. Each IR family keeps its schema/version, decoder task and
8D/384D/768D lineage. This increment changes no decoder weights and supplies no
learned legal-text or semantic-IR reconstruction result.

The [updated proof-context guide](../../repository_proof_context.md) and
[improvement plan](../../reversible_codebase_views_plan.md) retain those separate
teacher-readiness and ModelManager registration gates. Reuse existing assets;
qualify actual encoder conditioning and held-out semantic/text reconstruction
before distillation. Measure complete planning/coding/retrieval/expansion/edit/
verification/retry session input and output tokens at matched official success;
fixture byte reduction and authored usage figures do not establish token saving.

The [local documentation receipt](documentation-checks-02.json) and
[workflow log](documentation-workflow-02.log) bind the final source qualification,
baseline control and documents while executing all four exact workflow blocks.
Hosted CI is recorded separately by exact publication SHA; local success does
not imply hosted success. The final `manifest.json` is created after these
checks and excludes itself to avoid a hash cycle.
