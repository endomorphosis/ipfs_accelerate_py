# Read-only source views in the coding worker

The opt-in source reader connects the existing semantic producer and reversible
source codec to the isolated coding session through `llm_router`. It admits the
public path/hash population already verified in the task's semantic manifest.
It does not require a numerical task, Source384 inference, another database, or
another database owner.

The owner retains exact original bytes and codec maps. The worker can request a
compact view or expand the exact original. Editing continues in the allocated
original-source workspace. An edit makes baseline reads stale. The optional
working reader captures edited bytes as a separate candidate snapshot; it does
not refresh capsules or certify the changed program. Required source and
constraints remain in the initial prompt.

## Select and freeze the treatment

Add `--source-view-profile python-whitespace@1` to the existing
`benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark prepare`
command with `--arm full` and a freshly built runtime archive. Keep all other
task/provider/resource controls fixed. Preparation records the profile; execute
and collect refuse profile changes or removal from the prepared configuration.
The current worker capability pins the reader, codec, router, launcher,
observation collector, policy and command builder.

| Profile | Behavior |
| --- | --- |
| `disabled` | Default; no source endpoint or extra model guidance |
| `identity@1` | Exact source navigation through the same retrieval channel |
| `python-whitespace@1` | Conservative compact Python views preserving comments and names |
| `python-comments@1` | Explicit additional removal of ordinary comments, with codec-protected comments retained |

Unsupported files use exact identity views. Scope is the admitted semantic
manifest, at most 64 files and 1 MiB total, rather than an implicit claim to index
every file in a checkout. Non-model implementation routes and the no-index arm
refuse an enabled profile.

The worker sees a short instruction to invoke the pinned client:

```sh
/opt/ipfs-supervisor/venv/bin/python -I \
  /opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py \
  --path mod.py --view compact
```

Request `--view original` before editing, passing `--expected-source-sha256` and
`--expected-view-sha256` from the compact result to bind that expansion. Supply
the result's `next_cursor` JSON through `--cursor` for another page. A cursor is
issued, single-use, and bound to its exact source, representation and page
budget. Replaying it or changing its query refuses.

Normal client output contains the source slice, representation identities and
paging information after validating the full owner response. `--raw-response`
is an explicit diagnostic option for the complete protocol envelope. Restoration
maps and unrequested original bytes remain owner-side. Repeated owner-validation
fields are omitted from normal model output.

## Read working files after edits

Add `--working` to an unpinned first-page request after editing an admitted file:

```sh
/opt/ipfs-supervisor/venv/bin/python -I \
  /opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py \
  --path mod.py --view compact --working
```

The owner captures the whole admitted working-source population, regenerates
compact views, and rechecks the source before serving the candidate. An unchanged
snapshot reuses its generation. A changed snapshot receives a new generation
bound to the source hashes, immutable admission, codec, session and sequence.
Edit/revert cycles observed by working-source checks cannot revive an earlier
generation. Files outside
the admitted manifest are neither scanned nor served; this is not a claim to
capture every file or dependency in the worktree.

Candidate output includes `source_state: "working-candidate"`,
`workspace_generation`, `baseline_source_sha256`, and
`baseline_evidence_applicable: false`, including in compact default CLI output.
The page's `source_sha256` identifies the edited workspace bytes. Its semantic
binding and baseline digest continue to identify the original admission.

To expand or continue a candidate, keep `--working` and pass
`--expected-workspace-generation` from that page, along with both source/view
pins. For pagination, also pass `next_cursor` as `--cursor`. Pinned requests
never refresh the snapshot. Any admitted-file change invalidates the old
generation's continuations, including continuations for an unchanged file.
Request a fresh unpinned first page to observe the new working state.

This cache is session-local and holds one candidate snapshot. It does not update
DuckLake metadata, formal obligations, capsules, task state or accepted source.
Numerical and native source readers retain their own freshness checks. Candidate
text availability grants no reuse of baseline proofs. Deleted, renamed, unsafe
or oversized admitted files refuse; newly created unadmitted files remain outside
scope. Source writes and mapped compact edits are not performed by this reader.

The separate [candidate context reader](worker_candidate_context.md) can rebuild
advisory capsules and pending formulas for an explicitly pinned working
generation. Its artifact bytes are accounted separately from source slices;
accepted indexes and proof receipts retain their own publication lifecycle.

## Currentness and worker lifecycle

The existing owner launcher seals the compiled prompt, independently replays
the semantic producer, checks the allocated worktree, and creates the endpoint.
The service shares the numerical reader's process registration, socket framing,
expiry and cleanup code. Kernel peer identity and ancestry must match the actual
launched child. The model receives no database credential or unrestricted query
interface. The descriptor is public routing information, not a bearer grant.

Before and after a response, the owner checks its semantic artifact, source
manifest and boundary. The source service walks regular files without following
symlinks. Baseline reads verify the admitted hashes in canonical and allocated
workspaces. Working reads verify the unchanged canonical baseline and the exact
candidate generation in the allocated workspace. The owner checks that returned
generation again before delivery without refreshing it. It retains codec
identities and exact restoration checks. These are sequential observations, not
an atomic filesystem lease; unobserved changes between checks are not detected.

Responses are bounded to 8,192 JSON bytes including their protocol envelope;
sessions allow at most 64 connections and 524,288 response bytes. The lifetime
uses the existing selected coding timeout, up to 1,800 seconds. Large inputs
can exhaust those bounds; the reader makes no promise to expand the entire
repository within one session. A stopped or stale reader does not remove the
worker's ordinary access to its allocated source files.

This adapter is separate from the typed Quack owner's
[`source.get`](current_source_lookup.md). It reuses verified semantic source
custody and does not copy a host PID's grant into a model descendant or claim
native task-revision custody from a semantic identifier.

## Evidence and remaining work

The [qualification evidence](evidence/source-view-worker-20261012/README.md)
records exact source reconstruction through actual owner/worker UIDs in Docker,
stale/foreign-request refusals, cleanup, and smaller default CLI JSON output.
It uses an authored provider fixture and establishes no live-model token score.
The subsequent [working-source qualification](evidence/working-source-refresh-20261012/README.md)
adds actual worker edits, exact expansion of edited bytes, three observed
generations and refusal of stale continuations. Its provider hook is also
authored; it does not measure live-model savings.

The router records the source guidance in its complete rendered-input hashes
and bytes. The context audit reconstructs that guidance. After worker cleanup,
`terminal_source_view_observation` joins private session accounting to the
selected profile, semantic/router receipts and task alias. It exports counts
and hashes; missing, incomplete or mismatched evidence remains unavailable.

Response bytes count owner protocol envelopes, excluding framing and client
newlines. Returned source bytes include repeated reads and both representations.
They are not provider tokens or unique source coverage. Native planning and
coding usage remains the measure for a subsequent token comparison, including
tool results, expansions and repairs.

The [matched scripted read/edit pair](source_view_read_edit_pair.md) now compares
`identity@1` with `python-comments@1`, counting complete prepared input and actual
default CLI text. It saves 133 bytes out of 79,362 (0.17%) on that trace; it has
zero model calls and supplies no token or official reward score.

General capsule expansion, rebuilding formal indexes after working edits,
model-facing submission to the [host compact-edit mapper](reversible_source_views.md#map-an-explicit-compact-edit-to-original-bytes),
and measured live-model savings remain separate work. Complete-session
comparisons need fixed harness controls, all-attempt native usage and repeated
official task verification.
