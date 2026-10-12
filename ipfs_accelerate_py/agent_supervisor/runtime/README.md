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
| `repository_source_revision` | `runtime/repository_source_revision.py` |
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
redirect native terminal tools. Provider routing, incremental index refresh
and proof admission remain separate integration steps. Smaller package views
are representation measurements; session token savings require actual agent
use, complete usage accounting and maintained task success.

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
