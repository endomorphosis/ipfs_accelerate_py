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
