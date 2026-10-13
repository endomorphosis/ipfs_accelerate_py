# Shared source-view policy qualification — 2026-10-13

This increment reconciles accelerator
`85b21ded006fa9850ab49ee9f33bd320457a79a5`, including the parallel matched
source-view read/edit experiment. It implements an explicit common policy for
historical inventory, owner source reads and host mapped edits. Existing IR
weights, embeddings, checkpoint inventories, database owners and earlier
qualification bundles are preserved.

## Delivered owner contract

The new immutable `SourceViewPolicy` selects regular `.py`/`.pyi` views and
caller-protected physical lines. Its binding records canonical configuration,
resolver implementation bytes and the unchanged codec. Consumers independently
admit the source population and bind that policy to their source identities.
The resolver does not scan a repository, authenticate a source or grant a read.

Historical inventory roots explicitly opt into version 2. The owner reader
uses version 2 admission, version 3 baseline pages, version 4 working pages and
version 2 discovery. Working generations and continuations bind the policy via
their scope. The edit adapter obtains a checked candidate projection from that
same reader and emits version 2 application receipts with a policy digest.
Original source stays exact; the existing fenced candidate remains pending
validation and acceptance after application.

Default inventory roots and worker policy/wire formats retain their original
versions. The codec file and pure mapper are byte-identical to the baseline.
Shared host pages intentionally refuse through the old closed client; actual
model policy selection and discovery transport remain later increments.
Runtime archive capability now pins the new resolver dependency. Missing,
altered or copied-feature hashes refuse; existing archives need rebuilding
against this source before current preparation accepts them.
Source reader admission verifies regular filesystem files, while historical
inventory retains Git modes. This does not claim a new executable-mode check
for working source. Physical protected lines are not semantic anchors across
arbitrary external edits; the mapper retains its newline-edit refusal.

## Qualification and custody

The frozen qualification collected and executed **798 distinct tests**, including
**52 new policy cases**, with zero failures, errors or skips. Its
[command/source custody](qualification-01/commands.json),
[node population](qualification-01/nodeids.json), [JUnit](qualification-01/junit.xml)
and [execution log](qualification-01/tests.stdout.log) retain the result.
All 103 listed source bindings were unchanged before/after. Existing codec,
pure mapper, client and broker compatibility checks remain qualified alongside
real mapped edits, cache/policy tampering controls and runtime archive checks.

The [source review](independent-review.json) approved the consumer/packaging
integration after three fixes: complete cached-population validation, strict
legacy profile argument types and the new runtime dependency pin. It states
that its reviewer authored the pure helper and separately reviewed consumers.
The review also compares 29 scoped datasets dependencies at the executed and
parent revisions; equality of those files is not execution of the parent pin.

The [runner](qualification_runner.py) records exact interpreter/environment,
collection population, raw stdout/stderr, JUnit and source hashes before/after.
It uses separate temporary state/seal/cache outside public evidence. Tests use
the existing immutable datasets checkout
`f47161b757190380e9898aff155067c1ad609217`; parent datasets pin
`b95938c2a816075466f8dc84f339cf622414030d` is recorded as reference only,
without claiming execution of that newer revision. Existing kit identity is
recorded separately. No providers, checkpoint inference, training, Docker or
official verifier are invoked by this qualification.

The parallel [matched read/edit pair](../../source_view_read_edit_pair.md)
retains 79,362 versus 79,229 displayed bytes, a 133-byte difference in an authored
scripted workload. That historical zero-model experiment is not re-executed
here and establishes no provider-token or official success advantage. This
policy increment likewise makes no token-saving claim.

The [API guide](../../reversible_source_views.md#share-an-explicit-policy-across-owner-apis),
[worker boundaries](../../worker_source_views.md#explicit-shared-policy-in-host-apis)
and [improvement plan](../../reversible_codebase_views_plan.md) distinguish
implemented owner convergence from model-facing discovery, read observation,
full filesystem views and matched live comparisons.

The [documentation checks](documentation-checks.json) record the four exact
local workflow blocks and additional changed-document links/anchors. The
[publication manifest](publication-manifest.json) binds the changed sources,
guides and public evidence, excluding its own hash. Hosted CI is observed
separately in the parent integration note.
