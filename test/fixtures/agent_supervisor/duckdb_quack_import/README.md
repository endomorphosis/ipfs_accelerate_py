# DuckDB Quack legacy state import fixtures

Hermetic recipes for **DQP-010** (`LegacyStateImport@1`, `ImportManifest@1`,
`ImportReceipt@1`). Prefer compact generators and in-test file seeds over bulk
golden dumps that re-emit full envelopes per case.

## Purpose

These fixtures document the source shapes the importer must accept and the
reconciliation behaviors tests assert:

| Concern | Expected behavior |
| --- | --- |
| Exact replay | Same manifest + source digests → same `receipt_cid`, no double-insert |
| Conflicts | `select` / `merge` / `quarantine` / `reject` only (never last-write-wins) |
| Strict apply | Atomic commit of receipt + rows, or full rollback |
| Provenance | Every accepted row binds `source_digest` + `parser_version` |
| Source immutability | Importer never writes, renames, or deletes declared sources |

## Supported source kinds

| Kind | Typical path | Record model |
| --- | --- | --- |
| `markdown` | `*.md` taskboard / objective | `## RECORD_ID` headings + `- Field: value` lines |
| `json` | `*.json` | Object, array of objects, or `{schema, records:[...]}` |
| `jsonl` | `*.jsonl` | One JSON object per line; truncated final line fails closed |
| `sqlite` | `*.sqlite3` | Read-only table scan; source left untouched |
| `duckdb` | `*.duckdb` | Read-only table scan via DuckDB |

## Domains

Manifest sources declare one domain per file:

`objectives`, `taskboards`, `plan_revisions`, `queues`, `events`, `statuses`,
`worktrees`, `caches`, `artifacts`, `leases`, `databases`.

## Compact recipe (generate under a temp root)

```python
from pathlib import Path
import json
from ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import import (
    ConflictPolicy,
    ImportMode,
    LegacyStateImport,
    build_manifest,
)

root = Path("tmp-import-root")
(root / "board.md").write_text(
    "## TASK-1 Example\n- Status: todo\n- Priority: P0\n",
    encoding="utf-8",
)
(root / "events.jsonl").write_text(
    json.dumps({"event_id": "E1", "type": "claimed", "task_id": "TASK-1"}) + "\n",
    encoding="utf-8",
)

manifest = build_manifest(
    manifest_id="fixture-demo",
    sources=[
        {
            "source_id": "board",
            "path": "board.md",
            "kind": "markdown",
            "domain": "taskboards",
        },
        {
            "source_id": "events",
            "path": "events.jsonl",
            "kind": "jsonl",
            "domain": "events",
        },
    ],
    conflict_policy=ConflictPolicy.REJECT,
    mode=ImportMode.PREVIEW,  # default posture: preview before apply
)

importer = LegacyStateImport(root / "target.duckdb", root=root)
preview = importer.preview(manifest)
applied = importer.apply(manifest)  # strict, atomic
replay = importer.apply(manifest)   # no-op, same receipt_cid
assert replay.receipt_cid == applied.receipt_cid
```

## Conflict recipes

1. **reject** — two sources disagree on `task_id=T1` → both rejected, zero accepts.
2. **quarantine** — same disagreement → one quarantine bag holding both candidates.
3. **select** — set `selected_sources=("authoritative-source-id",)` or unequal
   `select_priority`; unselected peers become `conflict_select_not_chosen`.
4. **merge** — non-overlapping fields merge; any field-level value conflict
   quarantines the entity (`merge_field_conflicts:...`).

## Evidence subset covered by API tests

- duplicate identical sources
- conflicting authorities
- corrupt / truncated input
- unsupported schema identity
- rejected non-object rows
- exact replay
- source byte immutability

Authoritative tests live in
`test/api/test_agent_supervisor_legacy_state_import.py`. This directory holds
documentation and optional future compact recipe generators only; do not commit
large golden envelopes here.
