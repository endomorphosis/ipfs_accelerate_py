# Signed Terminal source selection qualification, 2026-10-13

The [profile and consumer guide](../../signed_terminal_source_scope.md) describes
opt-in `terminal-public-task-profile@4`. It signs a bounded context/program
selection inside the existing profile support while retaining the complete
signed task inventory and original output effects.

## Execution record

The final [commands](qualification-02/commands.json),
[raw stdout](qualification-02/tests.stdout.log),
[stderr](qualification-02/tests.stderr.log) and
[JUnit identities](qualification-02/junit.xml) retain one full compatibility
invocation across 13 suites, without separate collection. The two new suites
contain 76 cases. All **502 distinct cases passed**, with zero failures, errors or
skips, in 550.705 seconds. Raw collection, executed JUnit identities and the final
pytest summary agree. All 214 listed sources stayed unchanged before/after the
full execution, and datasets sources matched their pinned immutable Git objects.

The [initial focused run](qualification-01/commands.json) collected and executed
76 distinct cases: 75 passed and one failed during fixture cleanup. All assertions
in the warm native-owner/compiled-worker/historical-audit case preceded the
nonexistent `daemon.close()` call. The [correction](fixture-correction.json)
changes that one test line to the actual `close_event_runtime()` method. Runtime
code and the runner were unchanged between attempts. All original raw outputs,
JUnit and 11 source snapshots remain retained. There are two qualification
invocations in total: focused two-suite execution and full 13-suite execution.
The [runner](qualification_runner.py) binds listed sources, arguments, private
runtime directories, environment, timings and raw outputs independently.

The execution source base is accelerator
`7147fb5b2afcde9f6a2bb8abb35794d03755d862`, plus nine frozen production files and
two new test files. The actual immutable datasets execution revision is
`c9d0f016a9fcfa327346e056e38e2ffa3b25ffa0`; the kit revision is
`8ac00bb1974b1378e12091d546d8425af34049e0`. All 214 listed source bindings
comprise 163 accelerator and 51 datasets files, with datasets Git-object checks.
This is a listed dependency inventory, not a complete import closure or toolchain
attestation. Snapshots retain the 11 changed production/test sources for each
attempt. Earlier phase execution results are not substituted for this run.

## Tested boundary

The new integration case prepares 70 ordinary inert Python originals. All 73
inputs remain signed, including the three generated supports; five sources enter
the semantic manifest and two enter the vector index. Required raw support
includes the full canonical profile and original modified code. Bounded planning
coverage carries selected/full/frontier identities and counts without the full
source map. Both deterministic and ranked capsule modes cold-load and reuse
identical selected index/semantic bytes after native task admission, with zero
new embedding calls on reuse. Direct admitted context derives the same subset.

Controls cover sorted closed declarations, generated-path exclusion, role and
modify-output coverage, 61 ordinary plus three support files, exact 1,000,000
original bytes including support versus one-byte overflow, and legacy canonical
bytes. Current selected, omitted and support changes, unsigned population/profile
deletion, rehashed descriptor tampering and mutation during final empty-owner
capture refuse. An independent owner cannot sign a malformed scope. Full-population
decoder/training combinations refuse before selected context outputs. Complete
Doctor repair explicitly abstains, retaining selected diagnostics as planning
nominations. A selected vector population never claims complete-repository
emptiness.

The warm case uses a real native Quack owner, materialized task and compiled
Portal/Todo implementation prompt. It verifies selected context and no new
embedding/generation calls during rebind. Persisted capsule audit still works
after an omitted current source changes, because it checks historical artifacts;
a fresh live rebind refuses that drift. Historical profile downgrade also
refuses. No provider coding child is launched by this test.

Git, owner signatures, DuckDB/Intent, vector/semantic artifact producers and
native warm capture are real. Planner graphs and target originals are authored
fixtures. No training, model/provider call, solver, Docker container, target
program execution or official benchmark verifier is dispatched by this
qualification. Existing family/schema/version/decoder-task/8D/384D/768D weights,
embeddings, model-manager registrations and inventory databases are preserved.
Runtime test databases remain private and are not published.

## Historical and native limits

The pure scope observation checks exact retained profile/SHA/raw-CID/role
consistency. Signed source rows contain hashes and executable mode, without
lengths. Separately retained selected lengths therefore remain captured metadata;
available raw bodies have their hashes and UTF-8 lengths checked, while historical
audit reports `all_selected_lengths_independently_replayed=False`. Observation
flags remain historical, with no live signature, currentness, source-semantics,
proof, execution, completion or omission authority. Live producer/loader/rebind
checks independently replay the current complete signed population. Checks are
sequential and do not establish an atomic filesystem writer lease.

The new profile is a manual signed declaration. It is not automatically joined
to the historical CAS inventory/nomination, native source capability, allocation
handle or actual coding-child birth. The independent native source broker is
still limited to 32 paths; semantic scope allows up to 64 total files. New scopes,
edits and expansion need separately admitted lifecycles. Data and multitask
selection, per-decoder population qualification and complete Doctor repair remain
open. Rebuild and pin a fresh deployment archive before live activation; older
frozen archives remain historical.

Complete source replay still reads omitted originals, and the full profile is
required inline support. Source-count reduction does not establish displaced
provider tokens: the existing baseline already represents some nonrequired
programs by references. No latency, accepted proof, learned reconstruction gain,
benchmark reward, complete-session token savings or billing reduction is
established by this increment. The
[comprehensive plan](../../reversible_codebase_views_plan.md#signed-terminal-context-selection-2026-10-13)
retains native adoption, fixed-trace accounting and repeated matched experiments.

## Documentation and review custody

The [documentation runner](documentation_runner.py) checks retained qualification
and source custody, extracts and executes the four exact pinned offline
workflow blocks, and checks relative links/anchors in four changed guides and
this README. The [receipt](documentation-checks.json) and
[raw log](documentation-workflow.log) record outcomes. The documentation runner
has its own binding outside the 214 listed qualification sources and does not
rerun pytest. Prior phase evidence remains unchanged.

Root-authored [review observations](review-observations.json) record collaborator
feedback, source hashes, prequalification corrections and review limits. No
reviewer-authored execution receipt is claimed. The existing repository
sessions/compiler/planning stack and its policy/schema collision remain separate
reconciliation work. Foreign worktrees, source codecs and mapped edit semantics
are unchanged. Later publication/merge/hosted-CI receipts belong to the parent
publication record; they do not relabel the executed source tree.
