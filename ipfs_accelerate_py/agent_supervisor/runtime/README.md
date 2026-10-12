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
| `repository_source_index` | `runtime/repository_source_index.py` |
| `repository_source_mcp` | `runtime/repository_source_mcp.py` |
| `repository_source_policy` | `runtime/repository_source_policy.py` |
| `repository_source_probe` | `runtime/repository_source_probe.py` |
| `repository_source_revision` | `runtime/repository_source_revision.py` |
| `repository_source_session` | `runtime/repository_source_session.py` |
| `repository_source_view` | `runtime/repository_source_view.py` |
| `resource_scheduler` | `runtime/resource_scheduler.py` |
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
optional `--source-view-policy` shell helper. Other provider routes are not yet
integrated with these tools.

The runner adds a private, invocation-scoped stdio MCP server to the native
CLI configuration. It uses the documented [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)
and fixed command arguments; user MCP configuration files are not edited.
The CLI's shell tools remain available. Configuration, successful initialization,
actual tool calls, compact reads, canonical edits and failures are separate
receipt counters. Configuration alone is not adoption or evidence of savings.

| Tool | Behavior |
| --- | --- |
| `repository_status` | Current generation, source freshness, native head, coverage and resource bounds |
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
`native_index.retained_opaque_cache`, with exact inventory paths, original file
hashes and sizes, counts, and an inventory SHA256. The native `coverage` field
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

Status currently returns the full retained cache inventory. A cache-heavy
checkout can therefore add substantial metadata to the worker's context even
when source reads are smaller. A pending optimization is to return bounded
counts and an inventory digest by default, with explicit paginated retrieval of
the detailed inventory. That change must preserve full package retention and
freshness checks; measure status and retrieval traffic in future comparisons.

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
generation and clean up its private state. The probe returns content-free
checks and usage counters; it executes no target code and makes no provider
calls. It qualifies the installed tool runtime, not live model adoption.

The benchmark separately joins each observed coding invocation's retained input
audit, native configuration receipt and server usage. Qualification requires
actual successful repository reads, including compact reads for the compact
policy, complete private-state cleanup and no uncertain edits. Missing startup,
zero reads, malformed observations and failed provider attempts stay visible;
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
