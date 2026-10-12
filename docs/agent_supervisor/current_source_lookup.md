# Bounded current source through the existing owner

The supervisor's existing typed Quack owner can now serve a host-admitted
`source.get` operation backed by the [retained source inventory](repository_source_inventory.md).
The [reader](../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py)
checks historical byte integrity, native task custody and selected working-file
equality separately. It creates no database, content store, inference or provider
session. It returns exact original source, independently of the
[compact read-only views](reversible_source_views.md).

This first profile supports clean committed regular files only. Changed source
refuses; a refreshed working overlay needs a separately admitted current root.
Symlinks and Gitlinks require their own scope rather than being followed or
treated as ordinary editable source. No response grants proof, edits, omission
or completion authority. Required facts stay inline in current coding prompts.

## Host admission and native custody

```python
# server and existing_cas are the already admitted native owner and source store.
# All identities and the public file scope come from the trusted host admission.
binding = server.bind_current_source_lookup(
    repository=source_root, cas=existing_cas,
    expected_inventory_cid=source_ref.inventory_cid,
    request=source_request, limits=source_limits,
    task_cid=native_task_cid, task_revision=native_task_revision,
    expected_task_body_cid=native_task_body_cid,
    context_cid=admitted_context_cid,
    allowed_paths=tuple(sorted(public_raw_path_hex)),
    max_response_bytes=16384, max_slice_bytes=4096,
)
```

`expected_task_body_cid` is the existing control-plane `content_identity` of
the parsed native task body. The inventory's historical `task_id` does not
establish native task custody. The host independently joins that inventory and
context to an actual native task CID, revision and body. `context_cid` is a
host-admitted identity; this reader does not compile or semantically validate it.

The optional [host-only native task guard](current_source_task_admission.md)
captures the actual one-attempt claim, active daemon grant and full native body,
then can supply those task fields and refence the broker's reads. It retains the
existing coordinator and owner; the ordinary Terminal remote allocation,
compiled-context and actual coding-child handoff remain separate work.

Binding occurs once on the live gateway under its original owner lock. It retains
the exact owner identity, connection, reader and supplied CAS object, plus
physical/configuration source-store and workspace bindings. The public binding
records task/repository/tree/population/source/context identities and limits.
Public-path and source-limit populations are represented by count/content
identities rather than repeating their complete lists in every response.

This version has one task/source binding per gateway lifetime. Task revision,
source or context changes refuse reads rather than silently rebinding. Multiple
concurrent task bindings and an admitted refresh lifecycle belong in the next
session-channel qualification; this API does not start another owner to obtain
a different binding.

Only explicitly minted source grants can invoke the operation. They contain
exactly `source.get`, no command or database-task capabilities, and exactly the
native `task_cid`, `repository_id` and `tree_id` scopes. Existing status and worker
grants gain no source operation. The same kernel peer/process birth, grant
expiry/revocation and original gateway binding are rechecked before and after
each read under the owner lock. Current store generation/fence/database UUID and
native task revision/body are checked independently. No arbitrary SQL, caller
filesystem path, store locator or callback is accepted over the wire.

## Exact reads, bounds and continuation

```python
request = {
    "schema": "supervisor-current-source-request@1",
    "binding_cid": binding["binding_cid"],
    "raw_path_hex": public_path_hex,
    "source_cid": original_source_cid,
    "offset": 0,
    "max_bytes": 4096,
    "continuation": None,
}
response = admitted_source_connection.source_get(request)
result = response["result"]
```

Every request replays the original inventory from the same existing CAS and
verifies all retained frames, complete source CIDs and Git blob hashes. Complete
native Git preflight checks commit/tree/population, clean ordinary checkout and
visible index flags. Selected working bytes are read with `O_NOFOLLOW` directory
descriptors and compared in full to retained original bytes, including Git's
owner-execute bit. Every parent and leaf is rewalked and matched after reading.
This catches checkout newline/filter differences even when Git status is clean.
A final selected-file check follows native task/storage callbacks and another
preflight.

Raw path hex is the identity, including non-UTF-8 or control-byte POSIX names.
No synthetic display path is used for access. Absolute paths, NUL, empty path
components, `.`/`..` and `.git` components refuse. Source read errors retain
their native type internally; the RPC exports fixed error types/codes without
exception text or source bodies.

The closed response binds the host admission, caller grant, exact request and
path/source identities, byte offsets and actual slice CID. UTF-8 slices use
text; other slices use canonical base64. Whole-file responses starting at zero
also recheck the complete source CID on the client. Partial slices rely on the
authenticated admitted owner for their relationship to the verified whole blob;
their slice CID alone is not a proof of that relationship.

Default slice and complete response limits are 4 KiB and 16 KiB. Fixed hard caps
are 16 KiB and 64 KiB; public admission permits at most 256 paths. The bound covers
the complete canonical UTF-8 response envelope, including identity metadata,
escaped content, continuation, counters and the CLI's final LF. A separate exact
wire count substitutes the native four-byte length prefix for that LF. Native
source response frames also have a 64 KiB header-level receive limit, before
their body is retained. Oversized envelopes shorten the slice conservatively,
with an explicit remaining byte frontier. An envelope unable to fit with a
nonempty slice refuses. These are byte counts; no provider token or tokenizer
estimate is assigned.

Follow `end_offset` and `next_continuation` on the same grant, keeping path,
source CID and `max_bytes` fixed. The next cursor is content-bound to the admission,
grant, file and offset, and must have actually been issued by this reader.
Consumed, forged, foreign or expired cursors refuse. At most 4,096 unexpired
cursor records are retained in owner memory; they disappear on restart. This is
ephemeral read bookkeeping, without a new database or successful-read proof.
Every continuation repeats freshness checks. `source_complete` describes only
that file's returned range ending at EOF; `repository_complete` remains false.

Checks are sequential, without an atomic filesystem mutation lease. Another
writer can change source after the final observation. The first profile deliberately
cold-replays the complete inventory and reads the full selected file for a small
slice while holding the owner lock. Repeated expansion can have substantial
CPU/I/O and owner contention. This establishes no latency or token improvement;
a qualified selective frame reader with equivalent custody checks is later work.

## Host child CLI and session integration

The [CLI](../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup_cli.py)
performs one actual authenticated read through the existing owner's Unix socket:

```text
python -m ipfs_accelerate_py.agent_supervisor.runtime.current_source_lookup_cli --credential-fd N
```

The trusted parent starts the child, issues a source-only grant bound to that
actual PID/process birth, then writes a closed credential object into an inherited
same-UID read-only pipe. Credentials contain the existing socket/store identity,
client/birth and token; they never belong in argv, environment or public stdin.
Credential input is limited to 16 KiB with a ten-second deadline. Public stdin
contains the request and is limited to 8 KiB. Duplicate/nonfinite JSON refuses.
The child closes the credential descriptor, performs one handshake/read, closes
the connection and prints the complete validated canonical reply plus LF.
Failures emit a fixed error without credential or exception text. A one-call
child does not provide continuation across new grants.

This credential-pipe CLI remains a host-issued subprocess transport; a new
model-spawned child is a different native grant peer. The optional
[coding-worker session bridge](source_lookup_worker.md) provides a separate
protected public descriptor, exact descendant admission and pinned shell helper
while keeping one native proxy grant private in the original owner. Its host
controls and actual UID1001 Docker helper qualification now pass with an authored
response on the source-only legacy route. The
[session evidence](evidence/source-lookup-worker-20261012/README.md) records that
bounded result. Ordinary Terminal Bench tasks still require explicit native
task/source binding and trusted
launcher selection; the bridge does not discover or create those admissions.

Next qualify that task admission lifecycle, restart and working-overlay refresh.
Add capsule reads as a separate closed operation. The
[published numerical obligation worker channel](evidence/numerical-obligation-worker-20261012/README.md)
already qualifies isolated shell-child retrieval of all 45 pending public-source
obligations through its own admission. Source correctness, formal proof and live
model tool choice remain independent of that retrieval result. Qualify actual
model access before moving required source or facts out of the prompt. Compare
complete rendered inputs and all native planning/coding/lookup/continuation/repair
tokens against official rewards; count failed/refused lookups and unknown usage
too.

The [qualification evidence](evidence/current-source-lookup-20261012/README.md)
records actual local DuckDB/typed socket and subprocess controls. Its transport
fixture does not qualify the native Quack extension network or a live coding
model. Existing LegalIR, SecurityIR, CodebaseIR and IntentIR assets, embeddings
and schema/version/decoder-task/8D/384D/768D checkpoints remain reused unchanged.
