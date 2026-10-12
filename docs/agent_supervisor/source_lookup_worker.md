# Exact source lookup from an admitted coding worker

The [source worker broker](../../ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_broker.py)
provides a bounded shell route to the [existing native source reader](current_source_lookup.md).
A trusted launcher explicitly selects the original ready Quack owner, its already
bound allocated-worktree source scope and the actual worker process. The broker
keeps one source-only native grant private while model-spawned shell helpers use
a separate public Unix socket. It creates no database, state owner or CAS.

This is an opt-in integration. Ordinary Terminal Bench tasks do not automatically
receive a source binding or session. The route does not remove required source
or facts from prompts, refresh a working overlay, admit edits or certify logic.
LegalIR, SecurityIR, CodebaseIR and IntentIR assets, embeddings and separate
schema/version/decoder-task/8D/384D/768D checkpoints are reused unchanged. No
model, embedding generation, provider call or proof execution occurs in a read.

## Trusted launch and the original owner

The host first binds `source.get` on its existing `QuackStateServer` gateway as
described in [native host admission](current_source_lookup.md#host-admission-and-native-custody).
For this channel, that binding must name the allocated checkout, the retained
source inventory and an actual current task CID, revision and body. The
canonical repository and allocation must be distinct clean worktrees in the
same Git repository. An additional canonical reader replays the same inventory
and CAS; it does not create another owner or storage scope.

The optional [host-only native task guard](current_source_task_admission.md)
captures and refences an actual one-attempt claim through the existing
coordinator. A reader bound with that handle requires the same handle as the
broker's `task_admission`. This supplies task/claim custody without activating
the ordinary Terminal remote allocation or coding-child handoff.

The optional [native allocation guard](current_source_worktree_allocation.md)
also checks the sealed Portal projection and managed-worktree lifecycle.
Allocation-marked readers require the exact `worktree_admission` handle and its
retained `task_admission`; automatic coding-context/process handoff remains open.

```python
import subprocess

from ipfs_accelerate_py.agent_supervisor.runtime.source_lookup_worker_broker import (
    start_source_lookup_worker_broker,
)

# existing_server is the original READY owner in this process. Its existing
# gateway has already been bound to the exact admitted allocated source scope.
session = start_source_lookup_worker_broker(
    server=existing_server,
    canonical_repository=canonical_repository,
    workspace=allocated_worktree,
    boundary=deployed_boundary,
    prompt=original_prompt_bytes,
    timeout_seconds=90,
)

# Within the supervisor's existing child-custody and credential-scrub lifecycle:
child = subprocess.Popen(
    [owner_entry, *ordinary_coding_arguments, *session.extra_argv],
    cwd=allocated_worktree,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
session.note_spawned(child)
# The existing launcher supplies the exact original prompt, supervises/drains
# the child and cleans up its workers. Its finally path calls session.close().
```

The launcher registers one actual direct child of the original owner after
`Popen`; a caller-supplied PID or descriptor hash is not authentication. The
broker checks kernel socket peer UID and live process birth, then verifies the
bounded descendant chain to that exact child. An unrelated process with the
worker UID, an owner-UID caller, a dead child or a reused PID cannot read merely
by knowing the public descriptor. The native proxy grant remains bound to the
original owner process and has only `source.get` plus exact task/repository/tree
scopes. Its token and native socket/store/database locators stay private.

The factory requires the original server, connection, gateway, lock, owner
identity, source reader and CAS objects. Native task revision/body, store
generation/fence/UUID, grant expiry/revocation and canonical/allocated source
are checked again around each native read. A changed gateway, source binding,
public descriptor or protected endpoint refuses reuse. This version supports
one existing source binding per gateway lifetime; it does not discover another
owner or silently rebind a task.

## Public descriptor and shell usage

Deployment installs a root-owned `0555` helper at
`/opt/ipfs-supervisor/bin/source-lookup`. It uses isolated runtime Python and
imports only the protected deployed source/datasets roots. The router verifies
the helper's ownership, permissions, stable bytes and SHA before provider
dispatch. It rechecks the descriptor and helper after provider initialization.

The launcher supplies both flags to the ordinary coding route:

```text
--source-lookup-descriptor /opt/ipfs-supervisor/worker-observations/source-SESSION.json
--source-lookup-descriptor-sha256 ADMITTED_SHA256
```

Only the validated public descriptor path and digest are exposed during the
provider call through `IPFS_SUPERVISOR_SOURCE_LOOKUP_DESCRIPTOR` and
`IPFS_SUPERVISOR_SOURCE_LOOKUP_DESCRIPTOR_SHA256`; previous environment values
are restored afterwards. No token is passed in argv, environment or stdin.
`OWNER_ENTRY` forwards these public arguments without creating a source owner.
`WORKER_ENTRY` forwards them to the ordinary isolated coding router. The route
can be selected without semantic/numerical formalization; their independent
existing admission rules remain in force when also selected. Planning,
preflight, census-only and alternative finite/Doctor candidate branches refuse
the source flags. Without the flags, prompts and child environments retain
their existing defaults.

The closed `supervisor-source-worker-descriptor@1` object names task/revision,
repository, native context CID, source binding CID, original prompt SHA,
allocated workspace, boundary SHA, owner/worker UID and live owner process,
socket inode/device, session lifetime and byte/request limits. Its opaque native
grant ID supports cursor joins; it is not a credential. Context CID and
repository ID retain their native identities; a bundle digest or filesystem
path is not substituted for either.

`source_refs` contains every host-admitted path in this session. Each entry has
exactly `raw_path_hex`, `source_cid`, `size_bytes` and `git_mode`; entries are
sorted and unique. The first profile permits 1–32 regular files, modes `100644`
or `100755`, within the complete 8 KiB descriptor limit. Oversized or unsupported
populations refuse in full; they are not truncated. Read these public references
to select an actual admitted path/CID pair rather than guessing an identity.

The helper accepts one JSON object on stdin with the native seven-field shape:

```json
{
  "schema": "supervisor-current-source-request@1",
  "binding_cid": "<descriptor source_binding_cid>",
  "raw_path_hex": "<selected source_refs raw_path_hex>",
  "source_cid": "<selected source_refs source_cid>",
  "offset": 0,
  "max_bytes": 4096,
  "continuation": null
}
```

Invoke `/opt/ipfs-supervisor/bin/source-lookup` with that object on stdin. The
router's guidance uses the smaller of 4,096 and the actual admitted
`max_slice_bytes`; the native reader can further shorten a slice to fit its
complete response limit. The client rejects a foreign path/CID pair or slice
bound before connecting. Duplicate fields, nonfinite JSON, unsafe raw paths,
oversized input and unexpected fields refuse. The public descriptor must be a
single-link owner-owned `0440` file in a protected `0750` worker-observations
directory; its socket is `0660` with the pinned owner/inode/device. The separate
worker UID, exact cwd, root-owned container boundary, namespace/capability
checks, owner process birth, server peer and expiry are independently checked.

The CLI writes the entire closed `supervisor-source-worker-response@1` envelope
as canonical UTF-8 JSON plus one LF. Success returns exit 0; refused responses
and transport errors return a nonzero exit. Transport errors use a fixed stderr
code without exception, source or credential text. Successful public results
retain byte content/ranges/CIDs, the unresolved byte frontier, native source
binding CID and backend accounting. They do not serialize the full native
binding, owner identity or database/store locator.

Follow `result.end_offset` and `result.next_continuation`, keeping the same file
and `max_bytes`. Shell subprocesses share the session's original native grant,
so cursors remain bound to that grant across helper calls. Forged, foreign or
expired cursors refuse; restart does not recover the ephemeral cursor registry.
UTF-8 slices use text, other byte slices use canonical base64. The client checks
the actual slice CID and declared size, range/frontier and false authority flags.
A whole-file response starting at zero also recomputes the source CID. A slice
CID alone is not an independent proof that the slice belongs to the full source;
that join relies on the admitted current owner and native inventory replay.

## Limits, accounting and cleanup

The complete worker request envelope is capped at 8 KiB, including its session
and request identifiers. Replies have a 64 KiB framing hard limit and a 32 KiB
session profile. The session accepts at most 64 socket connections and accounts
at most 512 KiB of prepared worker response JSON plus LF; it conservatively
reserves one maximum response before another native read. Malformed, refused or
undelivered attempts still consume applicable connection/response budgets.
Lifetime is explicitly bounded to at most 1,800 seconds. Native slice, response,
inventory and source-file bounds still apply independently.

`response_bytes` counts the complete canonical worker JSON plus CLI LF;
`response_wire_bytes` replaces that LF with the four-byte socket length prefix
and therefore equals `response_bytes + 3`. Backend counters describe the full
validated native success envelope independently; `backend_response_sha256`
hashes its canonical JSON body without LF. They include validated native reads
that a subsequent currentness check refuses to publish. They do not measure
native handshake/error frames, partial transports or provider tokens. Unknown
costs are not assigned zero.

`close()` revokes the original native proxy grant, poisons its connection,
closes the listener, joins the reader for up to three seconds and removes only
the original socket/descriptor inodes. A protected owner-only audit retains
request/response hashes, byte totals, refusal/delivery observations, in-flight
state, revocation and cleanup status. `delivered` records successful socket
write, not client acceptance. An in-flight thread can outlive the bounded join;
the audit then marks accounting/cleanup incomplete. It does not claim an atomic
rollback, writer lease or complete native network accounting.

Both clean committed checkouts and full retained inventory are replayed; final
selected-file checks follow native custody callbacks. Edits, mode/filter/newline
differences, symlinks, stale task/context or owner changes refuse until the host
admits an appropriate refreshed source lifecycle. The cold full-population
replay and full-file reads for a small slice can impose substantial I/O, CPU and
owner-lock contention. Sequential checks do not prevent a writer from changing
source after the last observation.

## Qualification and remaining work

The [qualification bundle](evidence/source-lookup-worker-20261012/README.md)
records 537 distinct passing merged host regressions and an actual UID1000-owner to
UID1001-worker container exercise. An authored response drove the real router
and protected helper: three inert files reconstructed exactly across eight
primary pages and 11 distinct helper processes sharing one private native grant.
Wrong peers and canonical/allocated edits refused; original grant and endpoint
cleanup passed. No model, provider or solver was called.

The source-only container control used the existing legacy reply arm and a plain
prompt without semantic_repository. The modern ordinary-completion route needs
its independent sealed semantic context; this fixture does not qualify it.
An authored response establishes helper access, while live model tool choice,
formal proof, official task reward and total token savings remain unmeasured.
The router's closed observation records `requests_observed=False`,
`actual_tool_usage_unknown=True` and `total_token_savings_measured=False`.

The concurrent [semantic source-view channel](worker_source_views.md) provides
compact navigation and exact-original expansion through its own semantic
admission. That channel retains semantic source custody and an 8 KiB response profile; this bridge preserves the original Quack owner's source grant.
Both channels default to disabled until their respective host admission is selected.

Next integrate explicit task/source admission into the ordinary Terminal
launcher lifecycle, qualify restart and working-overlay refresh, and add capsule
reads as a separate operation. Reuse the already published
[numerical obligation worker channel](evidence/numerical-obligation-worker-20261012/README.md)
through its existing admission, without merging its authority or budgets into
this source reader. Keep required facts inline until actual model-access and
omission experiments qualify the change. Compare complete rendered inputs and
all native planning/coding/lookup/continuation/repair usage, including refusals
and unknown usage, against official rewards before claiming lower total tokens.
