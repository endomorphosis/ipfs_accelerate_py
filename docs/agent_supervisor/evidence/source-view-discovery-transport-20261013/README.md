# Source discovery and shared-policy transport — 2026-10-13

This increment starts from accelerator
`49b01d2515f338af73f8c14a96bbf235e339e4bd`, preserving the parallel bounded
context-selection observation and worker capability changes. It subsequently
reconciles accelerator `8aafb960fd634d218478376aa18bc83f8b121dce`, including
parallel output-observation fixes and Intent384 readiness. Parent integration
base `b69fe600ee350d4c016cbc3dd889265963e6e903` retains parallel LegalIR,
metadata projection and readiness publications. Existing model assets, separate
IR/checkpoint identities and prior evidence bundles remain unchanged.

## Delivered workflow

The actual standard-library worker client offers `--discover` alongside
`--path`. Discovery pages list only admitted baseline paths and their
source/view/manifest hashes, byte sizes and effective profiles. They contain no
source bodies, maps or database credentials. The owner checks current source,
complete retained original population and deterministic cached projections before
exporting pins. A discovered row can select a compact read or pinned exact
expansion; working reads retain explicit paths and generations after edits.

Discovery uses closed request version 3 under existing legacy descriptors or
new shared-policy descriptors. An owner-supplied `SourceViewPolicy` produces
descriptor version 2 with exact policy and baseline-scope hashes. The client
chooses request version 3 automatically, preserving shared baseline/working
page versions 3/4 and discovery version 2. Legacy read requests 1/2 retain
original shapes and page versions; downgrade or mixed-policy requests refuse.
Shared candidate-context reads still require the existing current generation.
The router's shared dispatch observation records both descriptor pins.

The owner constructor and sealed-prompt factory accept the explicit policy;
factory profile and policy profile must match before reading stdin. Standard
benchmark preparation still exposes its existing profile selector and legacy
policy defaults. This increment transports an admitted policy, without adding
a benchmark policy-config selection field or allowing a worker to choose one.
Admission remains at most 64 files and 1 MiB total, baseline discovery refuses
source edits, and continuations remain single-use and query-bound. There is no
repository-wide admission, working catalog, full compact filesystem or new
model mutation tool in this increment. The existing worker capability guard
already pins all six changed product modules; current preparation requires an
archive rebuilt against these hashes. No archive or container is built here.

## Accounting and qualification

Discovery metadata counts as rows, separately from source slices and candidate
artifact bytes. Every complete response remains charged to the existing bounded
session byte ledger. Shared audits use version 3 and shared dispatch observations
version 2. The collector validates policy/scope/mode joins and typed discovery
counts; historical audits without these fields retain an unknown discovery
count rather than asserting zero. This accounting does not estimate model tokens.

The final merged pytest invocation passed **1,603 distinct cases across 39
selected targets**, including all **79 new discovery/transport cases**, with
zero failures, errors or skips. All **166 listed source bindings** were unchanged
before/after. The [execution log](integration-01/tests.stdout.log),
[JUnit identities](integration-01/junit.xml) and original
[command receipt](integration-01/commands.json) retain the execution.

The original runner receipt keeps its `fail` status because its collection regex
missed pytest's `collecting ... collected 1603 items` prefix. This was a metadata
parser failure after pytest exited successfully. The separate
[result reconciliation](integration-01/result-reconciliation.json), produced by
the [standard-library verifier](reconcile_integration_result.py), joins that
single collection count to 1,603 unique successful JUnit cases and rechecks the
166 source, datasets and kit bindings. It does not rewrite the original receipt
or rerun tests. The [reconciliation review](result-reconciliation-review.json)
independently verifies the raw counts and verifier.

The [first attempt](qualification-01/commands.json) stopped after 48 passing
cases and one fixture failure: its cross-operation cursor case tried to index a
source page that had refused a 2,048-byte budget. The
[fixture correction](fixture-fix.json) uses a sufficient identical read/discovery
budget and adds a direct refusal check for the authentic source cursor. Its
[prior test bytes](qualification-01/test_worker_source_discovery_transport.py.snapshot.txt)
were reconstructed afterward and match the first attempt's recorded source hash.
No product or runner file changed for this correction; the initial attempt and
[initial review](independent-review.json) are retained unchanged.

The [second attempt](qualification-02/commands.json) passed 687 cases, including
all 79 new transport cases, then stopped on a legacy router fixture whose mocked
loader omitted the schema supplied by the production loader. The
[second fixture correction](fixture-fix-02.json) supplies that legacy schema;
its [prior bytes](qualification-02/test_source_view_worker_wiring.py.snapshot.txt)
match the recorded source hash. Both attempts remain failures, with their logs,
JUnit and original runner preserved. The final integrated run includes the
newly merged output/readiness changes and executes the newer datasets pin.

The [runner](qualification_runner.py) retains exact commands/environment,
collection IDs, JUnit, raw logs and source hashes before/after in the
historical qualification attempts. The [integrated runner](integration_runner.py)
uses one full pytest invocation against the merged source, verifies its collected
count and distinct JUnit identities, and binds source/kit identities before and
after. Runtime databases, AST seal state and caches remain outside public evidence.
The first two attempts execute immutable datasets
`f47161b757190380e9898aff155067c1ad609217`, with
`5723a541c5d575b88f01532e4d241dc95a1f3e99` as reference only. The final merged
run uses a separate locked checkout at
`afadfbcf8d33a67568c133bb761c43d42a9a133c`, matching the parent integration pin
and providing the new Intent384 checkpoint owner absent from the older checkout.
Its 51 listed datasets dependencies bind to immutable Git objects. No existing
checkout, weights or cache was replaced to make that dependency available.

The new CLI/socket tests use the real semantic producer, Git worktree, framed
socket, client owner-peer checks and registered child ancestry. Their authored
single-UID descriptor-loader/worker-UID seams do not qualify the container's
cross-UID filesystem boundary. No model/provider calls, checkpoint inference,
training, Docker, solver or official task verifier are run by this qualification.
No complete-session token savings or task-success advantage is established.

The [final integration review](integration-review.json) records incoming source
compatibility, the corrected legacy descriptor fixture and the newly pinned
Intent dependencies. Earlier [policy/transport review](independent-review.json)
and [first fixture review](independent-review-02.json) remain unchanged. Client
authorship is stated separately from consumer review. Static review establishes
neither a test pass nor a task-success or token benefit.

The [worker guide](../../worker_source_views.md#discover-admitted-sources),
[policy API](../../reversible_source_views.md#share-an-explicit-policy-across-owner-apis)
and [comprehensive plan](../../reversible_codebase_views_plan.md) describe the
remaining benchmark-selection, inventory-to-admission, read-observation and
matched live-model experiments. The historical scripted 133-byte difference is
preserved, without relabeling it as a result of this change. The parallel
[metadata projection pair](../../context_projection_pair.md), including its own
native token/task observations and limitations, is also retained without
re-execution by this qualification.

The [documentation checks](documentation-checks.json) record the four exact
local workflow blocks and additional changed-document link/anchor checks. The
[publication manifest](publication-manifest.json) binds changed sources, guides,
failed/final qualification attempts and review evidence, excluding its own
hash. Hosted CI is observed separately in the parent publication note.
