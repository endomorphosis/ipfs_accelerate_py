# Compare capsule selection with reversible encoding

`semantic_selection_comparison.compare_semantic_selection` is an experimental,
provider-free measurement seam. It compares four complete rendered user
messages for one current task and source inventory:

| Arm | Context |
| --- | --- |
| `full_native` | Complete native worker context. |
| `full_v2_aliases` | The same context with reversible V2 identifier aliases. |
| `query_selected_native` | Native lexical capsule selection with all source retained. |
| `query_selected_v2_aliases` | The selected context with reversible V2 aliases. |

The canonical repository remains in its ordinary state. This control isolates
capsule selection and encoding; it neither rewrites executable code nor enables
a new supervisor selection policy.

## Prepare matching inputs

Use the existing `prepare_semantic_context` twice with the same repository,
complete permitted `paths`, task, objective and preparation bounds. Set
`required_raw_paths` to **every scoped path** for this first control. Use
`worker_query=""` for the full artifact and an explicit task query for the
selected artifact. Keep outputs separate so both original artifacts and full
producer block indexes survive.

Compile both artifacts with the existing context compiler. Keep the immutable
task, authority, scope, acceptance, unrelated evidence, workspace and literal
suffix identical. Build semantic references with `chunk_bytes=6144` on both
arms; this API verifies that policy rather than assuming that different chunk
sizes are comparable.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_comparison import (
    compare_semantic_selection,
)

report = compare_semantic_selection(
    repository=canonical_repository,
    full_prompt=compiled_full_prompt,
    selected_prompt=compiled_selected_prompt,
    workspace=allocated_workspace,
    purpose="coding",
    provider="codex_cli",
    model=pinned_model,
    token_counter=None,
)
```

Both artifacts must pass the current codec's native producer/source verifier.
The helper checks the full capsule population against the retained root-backed
index, exact selected capsule/admission subsets, equal source manifests and
roots, identical complete raw source, and matching compiler/reference policy.
Each encoded arm restores to its own native prompt. Parent and source checks
repeat after counter work.

The report retains hashes, byte sizes and optional local content-token counts,
rather than prompt or source bodies. Without a compatible supplied counter,
model-token counts remain unknown. Failed or inconsistent counter observations
clear every local count; a strict token budget refuses such measurements.
Provider instructions and subsequent tool turns are outside these local counts.

## Examine what selection changes

The native selector ranks lexical matches in symbol names and paths. It can
fill remaining slots with zero-match symbols and rebuilds the context pack from
selected admissions. Selection can change assumptions, exclusions, risk, routes
and obligation references. The report records these differences by field and
digest; it does not claim the complete original decision context survived.
The comparison verifies the supplied subset and native root bindings; it does
not independently replay or qualify the lexical ranking algorithm.

`semantic_selection_dependency_audit.audit_selected_capsule_dependencies`
revalidates native capsule schemas and identities, then traverses only their
declared `dependency_stable_ids`. It reports omitted known dependencies,
unresolved targets, reachable cycles and bounded graph identities.
Native dependency identifiers can include unresolved textual names such as
`global:helper_operation`. The audit retains such a name as unresolved when it
is absent from the native inventory; matching a source spelling does not create
a resolved edge.

`declared_graph_closure_selected=True` means only that the supplied declared
graph closes within the selected set. It does not prove every program dependency
was represented. Source freshness belongs to the enclosing current-source
verifier; semantic completeness, source substitution, proof and deployment
authority remain outside this diagnostic. The comparison always records
`dependency_closure_verified=False` and `deployment_ready=False`.

## Use metadata as discovery

The existing `SupervisorMetaIndex.compose_for_subject` can locate a capsule
catalog/index for a bound subject. Its rows are nominations, not source-current
or dependency-coverage evidence. The fixture uses only the private metadata
catalogs created by its native preparations, then verifies the referenced native
root and index independently.

Do not use `compose_for_subject` as a read-only connection to another owner's
database: its existing connector opens read/write and performs schema setup.
Current composed rows lack repository/tree bindings and a complete pagination
frontier. Capsule metadata mirroring also bounds recorded dependencies; it
cannot supply exhaustive closure. The default Quack metadata operations expose
owner identity/generation, not this semantic-catalog query.

A future live nomination path needs an existing owner-supplied bounded query,
current root/source bindings and an explicit dependency/coverage contract.
DuckLake history can retain observations; historical metadata alone cannot
authorize omission or reuse.

## Evaluate total task usage separately

Four smaller or larger messages are representation observations. A Terminal
Bench token result requires the same assigned tasks, source, provider/model,
acceptance checks and execution settings, including every planning, coding,
expansion, failure and retry attempt. Use the existing session reconciler and
paired usage comparison; local measurements are not additional provider charges.

The first real-producer fixture retains target, helper, unrelated operations,
tests and instructions as exact raw source. No model/provider runs or acceptance
execution are part of the probe. Its findings and scoped validation are retained
in `artifacts/supervisor-selection-comparison-20261012-01` in the enclosing
workspace. Token savings, ranking quality and semantic completeness remain
unmeasured. See [whole-session accounting](TOKEN_USAGE_RECONCILIATION.md),
[projection comparison](SEMANTIC_CONTEXT_PROJECTION.md), and the
[implementation plan](../architecture/terminal_bench_token_reduction.plan.md).

The retained fixture contains eleven full capsules and selects one target
capsule, with all three raw source/test/instruction files still embedded:

| Arm | Complete rendered bytes |
| --- | ---: |
| Full native | 62,167 |
| Full V2 aliases | 48,979 |
| Query-selected native | 16,800 |
| Query-selected V2 aliases | 13,968 |

The native declared graph identifies one known omitted dependency capsule and
the unresolved name `global:helper_operation`. Both remain visible in the
audit; the smaller packet is not dependency-qualified for deployment. The
comparison also records changed pack decisions. These are bytes for one
authored task with real native preparation and verification, not model-token
counts, successful-task outcomes or a Terminal Bench campaign.
