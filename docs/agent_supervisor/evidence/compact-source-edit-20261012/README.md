# Compact source edit qualification — 2026-10-12

A compact view can now supply one exact retained byte-range edit. The mapper
preserves all original bytes outside that range and requires exact reprojection
of the candidate. A host adapter applies the resulting original-source patch
through an existing fenced `IsolatedWorktree`, checks staged and admitted source
postimages, and refreshes the working generation. It reuses existing source
custody, patch scope and lifecycle ownership.

- [API, preservation rules and limitations](../../reversible_source_views.md#map-an-explicit-compact-edit-to-original-bytes)
- [Architecture audit and remaining stages](../../reversible_codebase_views_plan.md)
- [Final regression receipt](final-01/qualification.json), [JUnit](final-01/junit.xml), [log](final-01/pytest.log), [exact command and source pins](final-01/command.json)
- [Public fixture receipt](public-fixture-01/qualification.json), [command](public-fixture-command.json), [execution/source pins](public-fixture-execution.json), [authored runner](public_fixture_probe.py)
- [Independent implementation review](independent-review.json)
- [Local documentation checks](documentation-checks.json)
- [Artifact checksums](manifest.json)

## Results

The final regression passed **339 tests**, with no failures, errors or skips.
All recorded source pins were unchanged during execution. Cases cover mapper
fidelity and refusals, native Git application, lease/fence/scope checks, staged
bytes, CRLF and missing final newlines, aggregate source limits, intervening
changes, candidate refresh and stale formal-context cursors. The formal refresh
integration uses the native datasets capsule/formula builder.

The separate host probe copied only public `largest-eigenval` environment inputs
(`eigen.py` and `eval.py`) into an authored isolated fixture. A prescribed local
variable rename through the compact view produced an exact staged source patch,
preserved the hidden comment and untouched sibling, and invalidated the old
generation. All **11 authored numerical comparisons** passed. The canonical
fixture source remained unchanged and the allocated worktree was removed.
Its 642-byte source had a 607-byte compact view; the rename changed an 86-byte
span to 80 bytes. These are source bytes, not model-token measurements.

This was a host qualification with authored task/semantic admission, not full
supervisor START, a Docker worker/model-tool test, an official benchmark attempt
or an official verifier result. No provider, checkpoint inference or prover was
invoked; no source obligation was discharged. Provider tokens and official reward
are unmeasured. No total-token savings or executable equivalence is claimed.

## Remaining boundaries

The adapter supports one modified text file in a fresh clean candidate. Dirty
overlay iterations, special path quoting, binary patches and file create/delete/
move operations are unsupported. Its checks are sequential; external writers are
not excluded and a failed post-write check may leave an isolated candidate for
cleanup. Receipts remain pending validation. The worker source socket stays
read-only; model-facing edit submission and accepted proof/index publication
require further integration.

The regression ran accelerator code based on `06a437cdba28e148912d998bf2cb85a805a3d4d9`
plus the recorded edits, datasets `f47161b757190380e9898aff155067c1ad609217`, and
kit `8ac00bb1974b1378e12091d546d8425af34049e0`. The parent retains the newer datasets
pin `e39fe996bd9e786a84f739fda9dde9a6c70546da`; this run does not qualify unrelated
changes in that revision. The command receipt records the actual environment;
the runner overrides the template's orchestration/seal paths per run. Retained
scripts contain qualification-machine paths and require explicit adaptation for
replay elsewhere. Private Git allocations, lifecycle stores and copied benchmark
source are excluded from this bundle. The independent review preceded completion
of the final regression; the final receipt and JUnit record its completed result.
