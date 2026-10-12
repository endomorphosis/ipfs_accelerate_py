# Read-only source views in the coding worker

The opt-in source reader connects the existing semantic producer and reversible
source codec to the isolated coding session through `llm_router`. It admits the
public path/hash population already verified in the task's semantic manifest.
It does not require a numerical task, Source384 inference, another database, or
another database owner.

The owner retains exact original bytes and codec maps. The worker can request a
compact view or expand the exact original. Editing continues in the allocated
original-source workspace. An edit makes the old source session stale; it does
not silently refresh capsules or certify the changed program. Required source
and constraints remain in the initial prompt in this first integration.

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

## Currentness and worker lifecycle

The existing owner launcher seals the compiled prompt, independently replays
the semantic producer, checks the allocated worktree, and creates the endpoint.
The service shares the numerical reader's process registration, socket framing,
expiry and cleanup code. Kernel peer identity and ancestry must match the actual
launched child. The model receives no database credential or unrestricted query
interface. The descriptor is public routing information, not a bearer grant.

Before and after a response, the owner checks its semantic artifact, source
manifest and boundary. The source service walks regular files without following
symlinks and verifies the entire admitted population in both canonical and
allocated workspaces. It retains codec identities and exact restoration checks.
These are sequential observations, not an atomic filesystem lease.

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

General capsule expansion, automatic working-overlay refresh, translating
compact edits into original patches, and measured live-model savings remain
separate work. A useful first source-view comparison is `identity@1` versus
`python-whitespace@1` through this same channel, with fixed harness controls and
repeated official task verification.
