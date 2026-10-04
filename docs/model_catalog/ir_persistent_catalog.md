# Discovering retained IR checkpoint metadata

`IRPersistentCatalogSource` connects to one explicit absolute ModelManager store
and rereads it on every `load()` or `refresh()`. It supports `.duckdb`/`.ddb`
through the existing read-only DuckDB reader and explicit JSON/JSONL metadata
files. It never constructs ModelManager, loads checkpoints, probes an encoder,
creates router bindings or changes running jobs.

```python
from ipfs_accelerate_py.model_catalog.catalog import AIServiceCatalog
from ipfs_accelerate_py.model_catalog.sources.ir_persistent import IRPersistentCatalogSource

source = IRPersistentCatalogSource(
    path="/home/barberb/lift_coding/external/ipfs_accelerate/model_manager.duckdb",
    source="ir-models.persisted",
)
catalog = AIServiceCatalog({source.source: source})
result = source.load()
bindings = result.ir_bindings
refreshed = catalog.refresh([source.source], raise_on_error=True)
```

An integrator that already owns a manager instance can add the same source to
its canonical catalog. Register it as a coordinated configuration change under
an unused source name; the generic registration API replaces an existing source
with the same name, so refuse a conflict instead of registering over it.

```python
# `manager` is an existing instance supplied by the application.
if source.source in {state.name for state in manager.catalog.source_states()}:
    raise ValueError("IR source name is already registered; inspect it before refresh")
manager.catalog.register_source(
    source.source, source, side_effecting=False, load=True, strict=True,
)
manager.refresh((source.source,), raise_on_error=True)
```

This updates only the manager's canonical catalog source. It does not reload or
replace `manager.models`, inference caches or model weights. The registration
and source-name check are separate public calls; an application must coordinate
concurrent source configuration. The example does not establish that any running
supervisor or MCP service has performed this registration.

Each result has immutable catalog records and a detached `ir_bindings` sidecar.
Known family, dimension, dimension role, schema, task, profile, format, checkpoint
SHA, record ID and checkpoint role are also exposed as catalog labels. Null
fields stay null in the sidecar and have no label. Original `ir_checkpoint`
declarations, including donor, initialization, training and qualification flags,
are retained as declarations. A declaration digest label makes these changes
visible in the catalog revision even when the selectors stay the same. A
component with a null external dimension keeps
its `source_tokens` or `unbound_component` role and its declared internal widths;
these widths do not establish an 8D/384D/768D embedding lane.

`result.resolve_ir_binding(request)` selects only within that captured result.
`source.resolve_ir_binding(request)` validates the request first, reads the
selected store again, then selects. Requests contain exactly `record_id`,
`ir_family_id`, `dimension`, `dimension_role`, `schema_version`, `task_id`,
`profile_id`, `format_id`, `checkpoint_sha256` and `role`. Nullable fields must be
provided explicitly. All ten values must match one record; there is no default,
latest checkpoint, donor fallback or decoder dispatch. Catalog descriptor IDs
and persistent record IDs are separate; each sidecar entry supplies their join.

All projected providers/models have `DECLARED` lifecycle, unknown operational
facts and no callable capabilities. No deployment or router binding is created.
Stored readiness/teacher/proof declarations do not become runtime observations.
Checkpoint bytes, tensors, cached embeddings, remote Hub state, decoder source
compatibility and reconstruction quality are not verified by this adapter.

Use a distinct source name when adding this source to an existing catalog. Its
refresh preserves unrelated sources. A missing/wrong store, malformed IR peer,
duplicate ID, inconsistent checkpoint revision or failed projection refuses the
whole load; the catalog retains the preceding successful generation and marks
the source unhealthy. A failed read must not trigger manager cache clearing,
service restart, database creation or fallback to a cwd-relative store.

The reader caps record count, stored config bytes, detached sidecar bytes and
store size. JSON files use regular nofollow/nonblocking descriptors, a capped
read and descriptor/path witnesses. DuckDB still materializes the selected bounded rows before config
validation, so deployments should bound process memory and time. Sequential file
witnesses can detect endpoint changes but are not an atomic filesystem snapshot.
An active writer or incompatible DuckDB connection may prevent a read; this is a
reported refresh failure. Configuring this explicit metadata source does not
establish that a running supervisor or MCP process already uses it. Legacy
ModelManager defaults, cached singleton refresh behavior and inference APIs are
unchanged.
