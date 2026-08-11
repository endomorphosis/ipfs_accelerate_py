# DuckDB / Quack legacy state import fixtures (DQP-010)

Hermetic recipes for `LegacyStateImport@1` / `ImportManifest@1` /
`ImportReceipt@1`. Tests generate compact temporary sources rather than
checking in bulk golden dumps.

## Purpose

Exercise deterministic import of legacy supervisor state:

| Media | Typical sources | Domain examples |
| --- | --- | --- |
| Markdown | objective heaps, taskboards | `objectives`, `taskboards` |
| JSON | task snapshots, plan bundles | `taskboards`, `plan_revisions` |
| JSONL | event / audit streams | `events` |
| SQLite | queues, leases, caches | `queues`, `leases`, `caches` |
| DuckDB | coordination / task stores | `leases`, `worktrees`, `artifacts` |

## Contract rules (fail-closed)

1. **Explicit manifest** — sources are declared; there is no directory-watch
   authority and no implicit import on process start.
2. **Provenance** — every accepted row binds `source_digest` (sha256 of source
   bytes) and `parser_version` (`legacy-state-import/1`).
3. **Preview default** — `ImportMode.preview` parses and reconciles without
   writing the target store.
4. **Strict atomic apply** — with `strict=true`, any rejected row aborts apply;
   a successful apply commits receipt + rows atomically or not at all.
5. **Exact replay** — re-applying the same `import_id` + manifest + unchanged
   source digests is a no-op that returns the same `receipt_cid`.
6. **Conflicts are not last-write-wins** — policies are only:
   - `select` (requires `selected_source_id`)
   - `merge` (complementary fields; contradictory fields quarantine)
   - `quarantine`
   - `reject`
7. **Source immutability** — importers never modify or delete declared sources;
   digest drift during apply/replay fails closed.

## Recipe shape (generated in tests)

```text
tmp/
  board.md              # ## TASK-ID headings + field list
  tasks.json            # { "records": [ { "id": ... }, ... ] }
  events.jsonl          # one JSON object per line
  queue.sqlite3         # table rows with id / task_id
  leases.duckdb         # table rows with id / owner / task_id
  control_import.duckdb # optional durable ImportStore target
```

## Validation

```bash
python -m pytest -q test/api/test_agent_supervisor_legacy_state_import.py
```

Evidence subset covered by the API tests: duplicate sources, conflicts,
corrupt/truncated input, unsupported schema, rejected rows, replay, and source
immutability.

## Authority note

Imported sources retain provenance only. After a successful apply, the durable
import store (and later the intent repository) is authoritative; legacy files
become non-authoritative exports unless a later explicit import is requested.
