# agent_supervisor.runtime

**Layer:** Ops · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Multi-lane execution fabric: multi-supervisor runners, event log, CAS, and resource/batch schedulers.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, merge/rescue coordination surfaces |
| **Typical dependents** | todo_daemon, ops scripts |

## Modules

| Module | Path |
| --- | --- |
| `artifact_store` | `runtime/artifact_store.py` |
| `configured_board_scheduler` | `runtime/configured_board_scheduler.py` |
| `context_selection_observation` | `runtime/context_selection_observation.py` |
| `event_log` | `runtime/event_log.py` |
| `intent_requirement_context` | `runtime/intent_requirement_context.py` |
| `intent_requirement_observation` | `runtime/intent_requirement_observation.py` |
| `learning_checkpoint` | `runtime/learning_checkpoint.py` |
| `local_planning_admission` | `runtime/local_planning_admission.py` |
| `local_prompt_tokenizer` | `runtime/local_prompt_tokenizer.py` |
| `multi_supervisor_runner` | `runtime/multi_supervisor_runner.py` |
| `operational_campaign` | `runtime/operational_campaign.py` |
| `provider_batch_scheduler` | `runtime/provider_batch_scheduler.py` |
| `repository_source_cli` | `runtime/repository_source_cli.py` |
| `repository_source_capsules` | `runtime/repository_source_capsules.py` |
| `repository_source_index` | `runtime/repository_source_index.py` |
| `repository_source_mcp` | `runtime/repository_source_mcp.py` |
| `repository_source_policy` | `runtime/repository_source_policy.py` |
| `repository_source_probe` | `runtime/repository_source_probe.py` |
| `repository_source_revision` | `runtime/repository_source_revision.py` |
| `repository_source_session` | `runtime/repository_source_session.py` |
| `repository_source_symbol` | `runtime/repository_source_symbol.py` |
| `repository_source_token_budget` | `runtime/repository_source_token_budget.py` |
| `repository_source_view` | `runtime/repository_source_view.py` |
| `repository_intent_world` | `runtime/repository_intent_world.py` |
| `repository_world_packet` | `runtime/repository_world_packet.py` |
| `resource_scheduler` | `runtime/resource_scheduler.py` |
| `role_context_selection` | `runtime/role_context_selection.py` |
| `router_public_instruction` | `runtime/router_public_instruction.py` |
| `runtime_cas` | `runtime/runtime_cas.py` |
| `runtime_temporal_monitor` | `runtime/runtime_temporal_monitor.py` |
| `scheduler_metrics` | `runtime/scheduler_metrics.py` |
| `semantic_candidate_index` | `runtime/semantic_candidate_index.py` |
| `semantic_capsule_expansion` | `runtime/semantic_capsule_expansion.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.runtime.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

`intent_requirement_context` projects exact accepted source and task bindings for
worker delivery. `router_public_instruction` verifies their signed public
admission without owner keys and includes them after semantic encoding.
`intent_requirement_observation` reads actual native owner validation records,
checks event order and source/revision/attempt freshness, and reports public
checks and output presence separately. These projections do not prove source
meaning, expand task scope, change canonical state or grant completion authority.

`semantic_context_runtime.prepare_semantic_context` records the final capsule
selection after byte-budget repacking in `context-selection.json`. The sidecar
binds the unchanged worker payload, source handles, native capsule/admission
identities, declared dependencies and obligation references. Its reference is
stored in the existing world observation and metadata catalog, with the existing
DuckLake projection. It is host metadata and adds no text to the worker prompt.
The observation distinguishes structural Codebase IR evidence from modality
advisors that did not run at this preparation stage. It does not establish
complete dependencies, current source at a later dispatch, provider delivery,
proof, or token savings. `verify_context_selection_observation` replays the
observation against a caller-supplied trusted producer payload; callers still
use the native semantic loader for task and source freshness checks.
If the observation exceeds its own resource bounds, preparation records a fixed
unavailable status and publishes no partial sidecar. The native context and its
acceptance requirements remain unchanged. Oversized worker payloads skip
observation decoding.

`role_context_selection` prepares separate, immutable host artifacts for an
explicit planning, coding or review profile. A profile binds its policy revision,
query, capsule limit, byte limit and optional verified retrieval artifact. It
uses the existing lexical or independently replayed retrieval selection; it is
not a learned role ranker. Required raw sources and native admission checks stay
in effect. Changing only the owner-declared observation role leaves worker bytes
unchanged. The role appears in the existing selection observation and its
DuckDB/DuckLake metadata reference; `role-context.json` additionally binds the
profile and exact worker/observation artifacts outside the worker prompt.

Use separate fresh output directories, and retain the returned reference and
the independently chosen profile:

```python
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime.role_context_selection import (
    RoleContextProfile, prepare_role_semantic_context, load_role_semantic_context,
)

repository = Path("/path/to/repository").resolve()
profile = RoleContextProfile(
    role="planning", query="public contract", capsule_limit=2,
    policy_id="repair-context", policy_revision=1,
)
reference = prepare_role_semantic_context(
    profile=profile, repository=repository,
    paths=["code.py", "instruction.md"], program_paths=["code.py"],
    required_raw_paths=["instruction.md"],
    objective="Repair the program while preserving its public contract.",
    task_id="repair-001", output=repository / ".semantic" / "planning-v1",
)
worker_bytes = load_role_semantic_context(
    repository=repository, artifact=reference["artifact"],
    expected_sha256=reference["sha256"], task_id="repair-001", profile=profile,
)
```

The loader checks the expected profile, native producer reconstruction, current
sources, selection controls and available observation bytes. Stale profiles
require explicit preparation into another fresh directory; the existing native
refresh API does not carry this host profile. An unavailable observation remains
explicitly unavailable. Neither a role label nor a successful load establishes
provider delivery, semantic relevance, modality coverage, proof or token savings.
This API does not change the benchmark's current planning/coding artifact reuse
or add a review provider route. The peer repository context compiler and planning
consumer remain the intended integration point for mandatory domain evidence,
question-bound expansion and retry deltas.

`ContextCompiler` enforces the provider byte ceiling on both transmitted retry
deltas and reconstructed full inputs, including consumer reverification. These
checks supplement the existing token, invariant-core and required-coverage gates.

## Bounded worker context

`prepare_supervised_task_context()` uses the native task title as its default
worker query and selects at most eight capsules within a 32 KiB worker payload.
`semantic_worker_capsule_limit` and `semantic_worker_max_bytes` control these
limits. Passing `semantic_worker_query=""` explicitly retains the full-context
comparison path. The lower-level `prepare_semantic_context()` keeps its existing
legacy defaults; use `worker_selector="duckdb-query-symbol-name@1"`, a nonempty
query, and explicit `program_paths` to request indexed selection directly.

The DuckDB candidate projection binds every row to the current repository,
source scope, native root, symbol version, source CID and capsule CID. Ranking
currently preserves the existing lexical relevance score. It uses one thread,
a 64 MB DuckDB memory limit, no spill, and at most 1,024 candidate rows. These
are query limits, not a process RSS guarantee. Preparation persists the snapshot
in `metadata.duckdb`, registers it with the meta-index, and retains its immutable
block. Loading reconstructs the projection from current source; SQL history
and DuckLake catalog observations cannot override freshness or admission.

Required raw input remains literal. If it cannot fit, preparation fails with a
budget error. Omitted source paths retain exact fetch bindings. A terminal
worker can explicitly expand native capsule IDs from the retained candidate
snapshot with a bounded response:

```sh
python -m ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_expansion \
  --repository /path/to/repository --artifact .runtime/context/semantic/worker-context.json \
  --expected-sha256 WORKER_PAYLOAD_SHA256 --task-id TASK_ID \
  --symbol STABLE_SYMBOL_CID --max-bytes 32768
```

Expansion checks the live sources and native block identities, preserves exact
capsule bodies and admissions, and grants no completion authority. It is an
explicit Python/CLI interface; workers still choose when to request expansion.

The router accepts `--prompt-tokenizer-artifact`, `--prompt-tokenizer-sha256`,
and `--max-input-tokens` to measure complete rendered alternatives and enforce
a local input budget before dispatch. The descriptor binds a local tokenizer
JSON asset, its SHA256, library version, provider/model, and special-token
policy. No assets are downloaded. Counts include literal task/reply instructions
and the separate native output schema, but exclude provider framing, hidden
tools and later turns. Receipts distinguish those local counts from actual
provider usage; tokenizer compatibility with a remote model is owner-declared.
The existing metadata byte/proxy admission remains in place before token-based
selection. Without an explicit tokenizer, existing router behavior is retained.

The offline benchmark can compare native full context, indexed initial context,
and exact expansion of every omitted capsule using an existing tokenizer cache:

```sh
python -m benchmarks.agent_supervisor.container_coding.indexed_context_benchmark \
  --output /path/to/new-output --tokenizer-cache /path/to/existing-o200k-cache
```

Its synthetic fixtures report local representation tokens and diagnostic
preparation time. They do not establish Terminal Bench success or cumulative
provider-token savings. Live comparisons must count all planning, coding,
retry and expansion turns, include failed attempts, and preserve task success.

## Reversible repository views

`repository_source_view` retains a bounded complete filesystem snapshot and a
compact presentation of supported Python files in a local immutable package.
Every original byte remains in SHA256-addressed blobs. Files whose rendering is
unsupported or does not reduce bytes pass through unchanged. Hidden files,
binary files, empty directories, permission bits and safe relative symlinks are
represented; `.git` metadata is explicitly excluded at every depth. Ownership,
timestamps, ACLs and extended attributes are outside the restoration contract.
Special files, hard links, escaping symlinks and privileged mode bits are refused.

The package can restore original source after its input checkout is removed.
It can also materialize a compact view for inspection. Compact Python retains
identifiers, strings and docstrings, but omits ordinary comments and formatting.
The package retains the original source required to restore those details.
AST comparison checks rendering structure; it does not establish behavior under
reflection, source inspection or tools that interpret comments.

```sh
python -m ipfs_accelerate_py.agent_supervisor.runtime.repository_source_cli \
  create /path/to/repository /path/to/new-package
# Use manifest_sha256 from the creation receipt in subsequent commands.
python -m ipfs_accelerate_py.agent_supervisor.runtime.repository_source_cli \
  read /path/to/new-package src/example.py --expected-manifest-sha256 HASH
python -m ipfs_accelerate_py.agent_supervisor.runtime.repository_source_cli \
  restore /path/to/new-package /path/to/new-original-tree --expected-manifest-sha256 HASH
python -m ipfs_accelerate_py.agent_supervisor.runtime.repository_source_cli \
  restore /path/to/new-package /path/to/new-compact-tree --representation compact \
  --expected-manifest-sha256 HASH
```

Creation, validation and restoration enforce file, byte and topology bounds.
Version one allows at most 4,096 entries, 64 MiB of source and 1 MiB per file;
creation options can lower these limits. Exceeding a limit fails the operation
instead of producing a partial repository. Outputs must be new, disjoint paths.
Publication uses Linux `renameat2` after staging and verification; existing
destinations are never replaced. Receipts report whether the parent directory
was successfully synced after publication. Source rereads detect changes during capture; callers should
still use an isolated checkout when other processes may edit the source.

`repository_source_revision` constructs a new package from a bounded batch of
hash-checked operations: `replace_symbol`, `write_file`, `delete_file` and
`rename_file`. It restores the parent into private temporary storage, applies
the batch there and publishes a complete successor. Parent source and packages
remain unchanged. Overlapping operation paths and stale hashes are rejected.
Symbol changes reuse the existing supported Python definition edit protocol;
arbitrary edited compact trees are not automatically reconciled. Use an
explicit full-source `write_file` for whole-file or comment-only changes.

For example, an operations JSON array can add a file:

```json
[{"operation":"write_file","path":"settings.json","expected_sha256":null,"content":"{}\n"}]
```

```sh
python -m ipfs_accelerate_py.agent_supervisor.runtime.repository_source_cli \
  revise /path/to/package /path/to/new-successor --expected-manifest-sha256 HASH \
  --operations /path/to/operations.json
```

Existing-file operations require `expected_sha256` for the parent's original
file. A symbol replacement adds `symbol` and `replacement`; a rename adds
`destination`. A full-source write adds `content` and optionally `mode` (an
integer permission mask). Edits to symlinks or through symlink ancestors are
refused; unrelated safe links remain unchanged. Test the restored canonical
successor before admitting it as a task result.

Package hashes identify this presentation artifact. They do not replace
Datasets semantic CIDs, proof-cache keys, source freshness checks or required
raw-source obligations. This interface performs no model calls and does not
redirect native terminal tools. The opt-in worker integration below connects
package reads and source refresh; proof admission remains separate. Smaller package views
are representation measurements; session token savings require actual agent
use, complete usage accounting and maintained task success.

## Native repository tools for coding workers

The Codex coding runner accepts `--repository-source-policy
repository-source-compact@1`, or `repository-source-raw@1` for a matched original
source comparison. Add that option to an otherwise configured
`router_implementation_runner` invocation, or select it when preparing the
[full container benchmark](../../../benchmarks/agent_supervisor/container_coding/README.md).
The benchmark freezes the choice and carries it through deployment, supervisor
dispatch and the coding router; planning does not receive repository tools.
The default is `legacy` pending live
task-success and total-token qualification. It cannot be combined with the old
optional `--source-view-advisory-policy` shell helper. Other provider routes are not yet
integrated with these tools.

The runner adds a private, invocation-scoped stdio MCP server to the native
CLI configuration. It uses the documented [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)
and fixed command arguments; user MCP configuration files are not edited.
The CLI's shell tools remain available. Configuration, successful initialization,
actual tool calls, compact reads, canonical edits and failures are separate
receipt counters. Configuration alone is not adoption or evidence of savings.

| Tool | Behavior |
| --- | --- |
| `repository_status` | Current generation, source freshness, native head, coverage, resource bounds and retained-cache counts/digest |
| `repository_cache_inventory` | Explicit page of exact retained-cache metadata, bound to generation and inventory digest |
| `repository_context_query` | Bounded lexical selection of native structural capsule references in the current generation |
| `repository_capsule_expand` | Exact requested capsules and admissions bound to that generation and semantic root |
| `repository_list` | Bounded file inventory with original and compact sizes |
| `repository_symbols` | Source-current native datasets symbol metadata |
| `repository_read` | Original or compact content, generation and canonical source SHA256 |
| `repository_replace_symbol` | Hash-checked Python definition edit in the canonical checkout, followed by refresh |
| `repository_refresh` | Capture external edits and advance the package/index generation |

Start with status to obtain the generation. Reads, symbol queries and edits
require it, and reject older generations. Full-tree checks before and after
reads detect ordinary concurrent shell edits, including added/deleted files,
permission changes and symlink changes. Call refresh after those edits. Native
qualified symbol names are structural metadata, not automatically safe editor
handles; supported replacements use the unambiguous definition name from the
source read. Comment-only changes require a canonical source editor and refresh.

The session retains original files, creates an original materialization outside
every Git checkout, and joins every eligible retained file/digest to the datasets-owned
`RepositoryCodebaseIndex`. DuckDB persists that native projection; Accelerate
does not introduce another semantic writer. Unchanged AST shards are reused;
semantic reconstruction remains fresh. Symlinks and non-UTF8 files remain opaque
to the native index; binary inventory still requires exact original bytes and CID.
Actual directories named `__pycache__` or `.pytest_cache` and their descendants
are retained separately as opaque cache coverage. The package still preserves
their bytes, modes, empty directories and safe links. Status exposes them under
`native_index.retained_opaque_cache`, with counts and an inventory SHA256;
paths and per-entry metadata are retrieved explicitly. The native `coverage` field
remains unchanged and covers only the datasets snapshot. Classification uses
the directory name alone: even a Python source file placed inside such a tree
is retained without native indexing or proof authority.

Cache changes participate in the same full-tree freshness checks as other
files. Test-generated caches therefore require an explicit refresh; their
creation, deletion or changed contents can advance the session generation
without reparsing unchanged eligible source. They still consume package entry
and byte limits. Files or symlinks merely named `__pycache__` or `.pytest_cache`
are refused. Other native exclusions such as `.venv`, `build`, `dist` and
`vendor` remain explicit refusals, including when nested inside a cache tree.
Code is never imported or executed by the server.

Limits are 512 entries, 16 MiB of original source, 256 KiB per file, eight
successful captures and 512 protocol messages per server. Failed captures do not
advance the public generation. DuckDB uses one thread, 128 MB memory and 128 MB
spill; the shared native scheduler applies pressure-aware admission with a
derived reservation capped at 2 GiB. These are cooperative parser admission and
SQL bounds, not a hard process RSS guarantee. Private packages, materializations
and native CAS artifacts are removed after the owned CLI process group stops.
No network installer or provider request is part of this server.

Single-symbol writes preflight their result and byte limits against retained
originals. If a canonical edit commits but index refresh fails, the reply says
`refresh_required`; old reads remain fenced until refresh succeeds. An uncertain
post-commit error is reported separately. Multi-file revisions remain immutable
proposals through `repository_source_revision`; this tool does not promote them
to the canonical checkout.

Full identities are available through status. Routine read replies omit repeated
native CIDs and index receipts to reduce transport overhead while keeping the
generation and canonical hash needed for guarded edits. This does not change
native semantic/proof cache identities. Local stdio tests and public-fixture
benchmarks verify this transport; they do not establish live worker adoption,
official Terminal Bench reward or end-to-end provider-token savings.

The model-facing status reply summarizes retained caches without listing roots
or entries. Its metadata size no longer grows with the number of cache paths,
apart from count digits. Internal `RepositorySourceSession.status()` preserves
the full receipt; `status_summary()` supplies the MCP response. To inspect
details, call `repository_cache_inventory` with the current `generation` and
`expected_inventory_sha256`, plus optional `offset` and `limit` (1–100). Pages
contain sorted original metadata and a `cache_root` marker; collecting all pages
reconstructs the exact inventory and its digest. Stale generation/digest handles
and source changes before or during retrieval are refused. Full retention,
freshness and native coverage remain unchanged; explicit page traffic belongs
in token comparisons.

When the coding router also receives its existing `--prompt-tokenizer-artifact`
and `--prompt-tokenizer-sha256` options, it passes that exact local descriptor to
the repository server. For compact read requests, the server counts both
complete JSON-RPC reply frames, including wrappers and the final newline, and
chooses compact only for a strict token decrease. Ties return original source;
explicit original reads remain original. During reads, failed counts or replies
over the counter's 256,000-byte limit return original source without claiming
measured savings. An invalid configured descriptor is rejected during loading.
Without an explicit tokenizer binding, the existing byte-based compact policy
is retained.

The descriptor binds the provider/model declaration, local tokenizer JSON,
asset hash and runtime version. Loading and counting verify local assets; there
is no discovery, download or network fallback. Source freshness is checked
again after counting. The retained content-free selection summary separates
measured choices from original fallbacks; detailed per-read selection records
stay in private invocation state and are removed during cleanup. Selection
metadata is kept outside the counted model reply. These counts describe local
wire text, not the provider's framing, remote tokenizer or billed usage; they
cannot establish total-session savings.

Repository context queries and capsule expansion now share this session's
current source snapshot and native DuckDB owner. `repository_context_query`
accepts `generation`, `query`, optional `limit` (1–32) and `max_bytes`
(1,024–65,536; default 32,768). It returns small candidate references, source
hashes, native capsule IDs and explicit population/match/omission counts.
Selection uses exact lexical terms from symbol names and paths. A no-match
result does not establish that a requirement is satisfied or irrelevant.

Pass the returned `semantic_root_cid`, `generation` and selected
`stable_symbol_ids` to `repository_capsule_expand`. Expansion returns the exact
native capsules, freshness admissions and source-read bindings for all requested
symbols, or refuses the entire request. Both operations bound the complete
returned JSON payload including the generation field; JSON-RPC framing is
separate. The capsule's native confidence and limitations remain intact.
`raw_source_required` on a candidate describes native admission; edits always
require an exact current source read and its canonical hash. No proof or intent
satisfaction is inferred from a capsule.

The native semantic bundle is compiled lazily from the already joined repository
state and reused within that generation. There is no extra source scan, training
job or database owner for compilation; normal source freshness checks still scan
the bounded materialization. Refresh discards the bundle. Old generations,
foreign semantic roots and source/head changes refuse delivery. Optional query
limits and resource admission refusals preserve ordinary source access; native
integrity failures require refresh.

This first profile admits at most 512 symbols, 4,096 edges and artifacts each,
4 MiB of source/state bytes each, and a 16 MiB native bundle. Compilation and
selection use shared resource admission with a 512 MiB reservation and one
30-second cooperative deadline spanning both native freshness checks and capsule
work. Native construction checks cancellation before
and after the build; these are not hard process CPU/RSS limits. Responses retain
at most 32 candidates or capsules. Only one generation's bundle is cached.
The query path supplies structural evidence. Learned Codebase IR, Intent IR,
Security IR, UI/UX IR and checked proof evidence still require explicit joined
interfaces; this change does not claim a complete multi-IR planner.

`session.context_adapter(compiler)` connects that same source owner to an
existing `ContextCompiler`. Its `compile` method accepts the complete
`required_evidence` population plus explicit `required_reference_ids` and
`required_coverage_ids`, the normal objective/policy/core fields, and a
generation, query and unresolved question. Every required reference must carry
its full nonblank body. The adapter refuses missing IDs, missing coverage,
conflicting duplicates, or insufficient budget. It preserves caller-authored
goal, scope and acceptance; authority contains the unchanged caller authority
under `task` and current native provenance under `repository_context`.
Optional references retain the native capsule CID rather than receiving a new
semantic identity. This does not infer the required population from retrieval
or verify the truth of caller-authored domain claims.

Call `expand_for_question(parent, EvidenceExpansionRequest(question, ids),
generation=...)` to resolve selected native handles into a standard compiler
delta. Capsule, admission, source hash and confidence must join the original
handle. Both selected and compiler-omitted handles can be resolved, but omitted
handles are supervisor-side: `render_context_capsule` does not transmit them.
A model must receive and be charged for any discovery needed to name such a
handle. Expansion never supplies mutation or proof authority.

`retry(parent, generation=..., evidence=...)` uses the existing delta compiler,
retains every original mandatory body and sends only changed or explicitly
requested diagnostics. Unchanged unrequested evidence is not replayed. Both
the transmitted delta and reconstructed provider input obey token and byte
ceilings. Successful reconstructed capsules can become subsequent parents;
foreign parents, edited sources and even same-source generation refreshes
require a fresh base. Checks before and after compilation detect ordinary
concurrent edits. Source-currentness does not establish intent satisfaction.

The adapter admits at most 32 input references per call, 256 KiB of context and
eight invocation-local parents, without eviction. `close()` forgets its parents
and leaves the source session open. Compiler work uses a separate shared
256 MiB reservation and a 30-second cooperative stage deadline; native fences
retain their own limits and release their leases before compiler admission.
These bounds do not provide hard Python RSS or CPU containment. The adapter
suppresses incidental metadata mirroring and creates no second catalog. This
is a host-side planning API; existing MCP tools remain the source access boundary.

`prepare_repository_world_packet` joins an issued context compilation with the
exact task text, explicit unresolved obligations, validation commands, permitted
effects and write paths through the existing `DeltaTaskPacket` builder and
admission. At compilation, include `repository_world_task_binding(task_id,
task_text, allowed_effects, write_paths)` under the caller authority's
`world_task_packet` key. Preparation rejects a changed task or effect/path scope
and refuses older contexts without that binding. Write paths must be explicit
and nonempty; an empty path population cannot accidentally imply unrestricted
writes. Every selected context body and core constraint must survive packet
admission unchanged; overflow or normalization that removes required content
refuses preparation. The packet has a bounded self-contained representation,
including an explicit unavailable-world marker when no projection was supplied.
An optional `RepositoryWorldState` must match repository, tree, objective, task
and policy revision. It remains an owner-supplied planning projection; this
adapter does not independently refresh its world owner or establish its claims.

The opt-in in-process call `router_implementation_runner.run(prompt=task_text,
world_task_packet=prepared, provider="codex_cli", ...)` delivers that packet at
the existing coding worker boundary. Keep the source session and context
adapter alive until dispatch. The runner verifies exact task/workspace/source
bindings again immediately before the provider attempt. Edits or generation
refreshes require a new context and packet. Worker source handles must come from
the worker's own source session, even when the preparation context refers to
the same canonical files. Existing repository source tools, provider timeouts,
cleanup and single-attempt behavior remain in force.

An optional declared tokenizer measures the actual rendered prompt and explicit
native output schema. An input-token ceiling requires a usable counter and
cannot fall back to a prompt that omits the packet. Provider-internal system
text, tool schemas and history are outside this measurement; packet item token
estimates are separately labeled byte heuristics. Receipts record identities,
counts, freshness and dispatch disposition without task/evidence bodies. They
always declare `provider_retained_parent=False` and grant no completion authority.
The API has no serialized transport, CLI selection or qualified Terminal Bench
replay yet; the benchmark audit refuses its marker. It is disabled by default
pending that qualification and matched complete-run evidence.

`capture_repository_intent_world` supplies a required context reference from an
independently owned `IntentRepository`, the existing signed local requirement
observer and the native intent world snapshot reader. Include `.reference` in
the compiler's declared required population, then pass the issued observation
as `intent_world_observation` when preparing the packet. The selected task's
reviewed specification, current signed check outcomes, applicable requirements
and unavailable world components enter the self-contained worker input. Generic
validation rows cannot stand in for signed observed checks. The capture API is
read-only and runs no task validation commands; the existing observer may run
read-only Git subprocesses. Validation remains with the existing local owner.

Packet preparation and dispatch repeat both source and native intent checks.
A newer check, changed task revision, modified admission or changed source
requires a new observation and context. Source restoration alone cannot revive
old validation evidence. The local validation tree and source-session tree use
different identity contracts and remain separate. No accepted repository-tree,
policy or completion authority is manufactured to fill missing world components.
The local capture entry point uses an independently owned intent database.
Capture checks shared resource admission before reading the owners and reserves
256 MiB for the world projection. The reference defaults to 32 KiB and cannot
exceed 64 KiB; exceeding a bound refuses the whole reference. The 30-second
deadline is cooperative and cannot interrupt an active native query. The global
intent event watermark deliberately invalidates captures on unrelated owner
updates as well. Recapture overhead has not yet been benchmarked.

`capture_repository_owner_intent_world` accepts the same source session,
admission, task CID, generation and byte limit, with `server` replacing `intent`.
It requires the actual running `QuackStateServer` and native Quack transport,
and reads through the owner's existing exclusive connection. Use its issued
observation and required reference with the same compiler and packet APIs.
The original signed `intent_owner_id` remains separate from the live server's
`native_owner_binding`, since admission and current custody can have different
owners. Public evidence includes only the allowlisted custody identities; the
connection, listener endpoint, database path and credentials stay private.

Native capture checks shared admission before readiness, then lets the strict
signed observer own its transaction under a deadline-bounded lock acquisition.
A separate resource lease precedes the
owner lock for the world snapshot transaction. Exceptions roll back that
transaction without closing the owner's connection. Every dispatch rechecks
readiness, exact connection/transport custody, restart identity and observations.
A stopped or restarted owner requires a new capture. Deadlines remain
cooperative, including inside the reused observer. This is an in-process owner
API; remote packet transport and durable restart reconstruction remain unqualified.
Capture leaves native task data unchanged; the reused readiness check may
restore the owner's private attach-token file as part of normal owner maintenance.

Opt-in container deployment requires an archive matching the current runtime
and datasets owner sources. Preparation, execution and deployment verify the
compressed archive digest, inspect the actual member inventory without
extracting it, and hash the required packaged source bodies against their
current implementations. Canonical paths, unique regular members, sizes and
permitted modes must match; extended-header metadata is bounded before parsing.
Unrelated large model bodies are streamed past, not loaded or individually
hashed by this source compatibility check. Declared manifest hashes alone do
not qualify the packaged runtime.

Deployment runs `repository_source_probe` under the
installed supervisor account on disposable authored source. The real stdio
server must initialize, expose its tools, report exact retained cache coverage,
serve a compact read, commit a guarded edit, refresh the index, reject the old
generation, and query and expand a capsule bound to the edited source before
cleaning up its private state. The probe returns content-free
checks and usage counters; it executes no target code and makes no provider
calls. It qualifies the installed tool runtime, not live model adoption.

The benchmark separately joins each observed coding invocation's retained input
audit, native configuration receipt and server usage. Qualification requires
actual successful repository reads, complete private-state cleanup and no
uncertain edits. Without a tokenizer binding, the compact policy requires
actual compact reads. With a binding, it requires a closed selection summary
matching the successful-read counters, at least one complete measured
comparison and no unmeasured fallbacks. Measured choices may all return original
source and still establish policy use; they do not establish token savings.
Missing startup, zero reads, malformed observations and failed provider
attempts stay visible;
configuration cannot substitute for those observations. Matched live trials
with total provider usage and official task results remain required to establish
practical savings.

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/runtime.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)

The checkout helper uses `source_view_advisory_policy` and
`--source-view-advisory-policy`. The structured `SourceViewPolicy` recipe and
`--source-view-policy-json` continue to bind signed source codec custody.
These selections are independent; helper guidance does not grant source access.


## Maintenance lease validity

New exclusive maintenance acquisitions retain `MaintenanceLease@1` and persist
`maintenance-lease-validity@1` in the existing extension columns. The extension
records canonical TTL seconds; expiry is an actual UTC deadline. Backup entry
and each exclusive repository access bind the complete supplied record to the
native row, including owner session, process birth label, token, fence epoch,
revision, acquisition/expiry and extension body. Validity is inclusive at
acquisition and exclusive at expiry. Supplied records cannot strip the native
marker or select legacy behavior for a marked row.

Legacy empty-extension leases retain their historical state/token/scope checks.
Equal legacy acquisition/expiry watermarks do not acquire a new deadline. No
existing rows are migrated, expired, swept or taken over. An existing active
legacy row still blocks a new acquisition until explicitly released. Marked
release uses the exact owner/row/revision CAS while allowing expired-owner
cleanup; exact response-loss replay returns the already released record without
another revision. Missing authority databases refuse before creation, and the
existing missing-source restore guard remains in force.

The lease records bind stored identity. Free-form session/process-birth labels
do not establish operating-system liveness, independent owner authentication or
IR admission. Marked exclusive repositories refuse raw `client`, `transaction`
and custom transaction callbacks before returning a retainable native handle.
Migrate those marked-mode callers to closed repository operations. Ordinary and
legacy modes retain their exact native client/transaction API. Release timestamps
are also pinned: active records require no release time, released records require
the actual native UTC transition time, and replay cannot replace it.
Access checks do not make filesystem backup/restore race-proof or guarantee that
an operation started before expiry completes before expiry. Operators must stop
Quack and retain actual exclusive ownership for direct-file
maintenance. Keep genuine backup custody outside disposable source worktrees.


The exact native Quack command fabric's private owner connection reuses the
existing immutable typed-owner operation catalog for named lifecycle-history
operations in its original transaction. Its ingress and read projection remain
actual separately served Quack endpoints; the operational connection is private.
Named calls require an active owner transaction, a known non-bootstrap operation
and exact parameter count. This capability exists only during that transaction;
a retained method refuses after commit, rollback or close. Outside transaction
custody, existing generation/session/heartbeat/close lifecycle operations keep
their original trusted closed-template path. Client SQL, templates or operation
catalogs cannot enlarge this surface. Original signed authorization, live-lease/fence checks,
replay consumption, task CAS and durable receipt remain in the same transaction.
This owner-local binding introduces no remote grants or owner trust and does not
grant model, IR or Lean admission.
