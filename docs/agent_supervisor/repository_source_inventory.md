# Retained committed source inventories

[`repository_source_inventory.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_source_inventory.py)
implements the first storage stage of the
[compact source-view plan](reversible_codebase_views_plan.md). It retains the
complete committed Git population in an existing datasets `ImmutableCAS`, then
reconstructs exact historical bytes without the original checkout or Git objects.
It returns data records, not a restored filesystem or accepted semantic state.

The separate [current-owner source lookup](current_source_lookup.md) reuses this
inventory and existing CAS, adds native task custody and full selected working-file
checks, and exposes a bounded host-issued subprocess read. It does not make the
historical inventory a current-source or coding-model authority on its own.

The producer reuses `preflight_committed_repository`,
`snapshot_chunked_repository` and `read_chunked_blob_frame`. The consumer reuses
`parse_chunked_snapshot_manifest` for closed canonical DAG, population, order,
mode, count, decoder profile and contiguous chunk validation. It additionally
reads every retained frame, checks its length/raw CID, recomputes the complete
source CID and verifies Git's blob-object hash. Neither a manifest hash nor
the metadata parser alone establishes source byte integrity.

## Caller-supplied source and storage

Supply the existing exact native `ImmutableCAS` object, such as an already
admitted index store's `cas`. This API does not construct another CAS or database
owner, discover a locator, enumerate proof caches or allocate a model. Capture
requires the CAS directories to be outside the source repository. It pins the
same store object, paths, directory identities and object-size configuration
around callbacks. Caller-supplied request and limits are privately copied and
their original bodies are checked for callback-time drift.

The caller independently admits task/repository identity, full commit/tree IDs,
complete population CID and source/store access. Capture checks a clean exact
Git repository and the complete native population before acquiring bytes.
Metadata preflight bounds per-file/per-path bytes and predicts the frame reads
and repeated verification work before native blob acquisition.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_inventory import (
    SourceInventoryLimits, SourceInventoryRequest,
    retain_source_inventory, load_source_inventory,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import (
    preflight_committed_repository,
)

# source_root, commit, tree, repository_id, task_id and existing_cas
# come from the caller's admitted source/store context.
limits = SourceInventoryLimits()
plan = preflight_committed_repository(
    source_root, expected_commit=commit, expected_tree=tree,
    repository_id=repository_id, max_entries=limits.max_entries,
    max_file_bytes=limits.max_file_bytes, max_total_bytes=limits.max_total_bytes,
    max_metadata_bytes=limits.max_metadata_bytes,
)
request = SourceInventoryRequest(
    task_id, repository_id, commit, tree, plan.population_cid,
)
ref = retain_source_inventory(
    source_root, cas=existing_cas, request=request, limits=limits,
)

# Retain ref.inventory_cid and request independently through the current owner.
# Later, supply the same admitted content store; source_root is not needed.
restored = load_source_inventory(
    cas=existing_cas, expected_inventory_cid=ref.inventory_cid,
    request=request, limits=limits,
)
page = restored.metadata_page(offset=0, limit=64)
```

Capture publishes source frames and the original native manifest DAG immutably.
It first replays all stored bytes, rechecks current source, then publishes the
closed inventory root last. It replays the actually stored root/content again
and rechecks source/store/input bindings before returning a reference. Failure
can leave immutable partial artifacts or a root, but returns no successful
capture. These observations are sequential, without an atomic source/store
lease, rollback or permanent availability guarantee. A subsequent load always
revalidates the retained content.

Storage and native producer errors retain their original exception types;
`SourceInventoryError` identifies this adapter's own refusals. Callers must not
assume every missing/corrupt native object has that one exception type.

## Representation and coverage

`RestoredSourceInventory.entries` contains immutable `RestoredSourceEntry`
records with the native committed entry, source CID and exact bytes. Paths use
`raw_path_hex` as identity. Preserve those records without interpreting synthetic
display paths as permission to write files.

| Committed entry | Restored data | Boundary |
| --- | --- | --- |
| Regular file, `100644` | Exact committed blob bytes | Binary, non-UTF-8, CRLF, empty and unsupported-language source remain bytes |
| Executable file, `100755` | Exact committed blob bytes and mode | Git executable classification, without arbitrary POSIX permissions |
| Symlink, `120000` | Exact link-target blob bytes and mode | Never follows the target or creates a symlink |
| Gitlink, `160000` | Exact child commit reference; data/source CID are null | Explicit separate-repository frontier, without child/forest content |

Empty blobs have no frames; replay still verifies their empty raw CID and Git
blob identity. Duplicate blob contents share stored frames while all committed
paths remain present. Per-path materialization totals are bounded independently
of unique source streaming totals. Normally filtered directories remain in this
complete committed population.

The representation is Git committed bytes. Checkout smudge filters, newline
conversion, untracked files, dirty working overlays, Git history, empty
directories, uid/gid, timestamps, xattrs and hard-link relationships are outside
this inventory. Exact committed bytes are not a claim that a current working
filesystem is identical. The caller needs separately admitted current-source
checks for operational use.

`observation()` identifies the repository/commit/tree/population, entry/blob
counts, unique/per-path bytes and Gitlink frontier. It marks historical-only,
current-source-unverified, filesystem-not-restored and all authority flags false.
`metadata_page()` returns at most 256 entry descriptions with an offset/next
offset, source identities and explicit Gitlink status. It includes no source
bodies and does not perform current-source admission. These pages are host data,
not a registered Quack operation or qualified coding-model expansion tool.

## Bounds and export cost

Defaults are 20,000 entries, 4 MiB per file, 128 MiB per-path/unique-source
budget, 32 MiB native manifest metadata, 1 MiB frames, 65,536 native chunk
references, 4,096 export frame reads and 512 MiB predicted Git verification work.
Typed overrides remain inside fixed caps, including 128 MiB materialized content
and 1 MiB frames. The existing supplied CAS must have its object-size limit at
most 16 MiB; every admitted manifest/frame is checked against its smaller bound.
Inadequate limits refuse the complete request rather than dropping entries.

The current native frame reader hashes the complete blob on every read. A blob
with `n` frames therefore costs one initial full hash plus `n` full Git streams
during export. The work estimate includes these streams. Cold replay reads each
declared chunk occurrence once, including shared CIDs appearing in different
blobs, and verifies the declared acquisition profile against its
caller limits. There is no indexed random-access or export speed claim. A future
single-pass native retained-frame producer can improve this cost without changing
the exact-byte consumer or introducing a new database owner.

The [qualification evidence](evidence/repository-source-inventory-20261012/README.md)
retains 28 new integration cases and 26 existing reconstruction regressions,
including fresh-process reload after the authored source checkout is deleted.
Normal AST sealing is enabled; model/provider calls and token measurements are
absent. Rebuild and qualify a current runtime archive before deploying this API
through a Terminal worker; existing frozen archives remain historical evidence.

## Next integration

The first stage qualifies committed source custody and historical byte replay.
The [reversible source views](reversible_source_views.md) now supply compact
read-only Python candidates and complete unchanged-source restoration maps.
Working-overlay candidate refresh, bounded current-owner `source.get`/`capsule.get` through the actual
coding interface, mapped edits and filesystem materialization remain separate
stages. Required source stays available in the current prompt until qualified
expansion exists. Keep existing source execution and acceptance checks.

Record source inventory references alongside existing metadata identities while
preserving the admitted catalog/owner contracts. DuckLake history stays
observational; it cannot promote this inventory to proof or current source.
Existing LegalIR, SecurityIR, CodebaseIR and IntentIR inventories/checkpoints
retain their family/schema/version/decoder-task and 8D/384D/768D separation.
This adapter performs no model inference, training or embedding regeneration.
Token measurements require the later complete-input and complete-session
experiments; source byte fidelity establishes no token savings.
