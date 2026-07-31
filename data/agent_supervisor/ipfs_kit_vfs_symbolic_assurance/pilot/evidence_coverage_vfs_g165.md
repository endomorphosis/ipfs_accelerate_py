# Evidence coverage — VFS-G165 / `vfs/swissknife-vfs-pilot@1`

**Gap task:** VFS-085  
**Parent goal:** VFS-G131 — Run the frozen SwissKnife and IPFS Kit VFS pilot  
**Evidence term:** `vfs/swissknife-vfs-pilot@1`  
**Discovery:** `projection/discovery/2026-07-30-vfs-085-objective-gap-b4f0ed91b948.md`

## Gap

Objective scan reported **no present evidence** for `vfs/swissknife-vfs-pilot@1`
even though the pilot producer (`vfs_symbolic_pilot.py`) and unit tests already
existed (VFS-037). The durable declared outputs — the pilot artifact tree and
findings board — were incomplete or absent from the worktree.

## Closure

This package materializes the full hermetic pilot pipeline with seeded contract
break + inconclusive fixtures so acceptance criteria are observable:

| Criterion | Evidence |
| --- | --- |
| Every admitted file accounted for | `manifest.json`, `coverage.json` (7 admitted; 5 SwissKnife; 2 VFS closure) |
| Findings reproducible from CIDs | `finding_ledger/`, `findings.json`, report artifact CIDs |
| Inconclusive non-actionable | taskboard review record; `executable: false` for inconclusive family |
| Board bounded / deduplicated / goal-backed | `taskboard.json`, findings board projection under VFS-G101 lineage |
| No provider / mutation / repair authority | report + board: `provider_calls=0`, `source_mutations=0`, `authorizes_repair=false` |

## Key CIDs (this materialization)

See `report.json` and `swissknife_vfs_pilot.receipt.json` for the binding set.
Primary report CID is recorded in the receipt as `report_cid`.

## Heap / backlog alignment

- No objective-heap child goals added: the existing VFS-G131 → VFS-G165
  refinement already names this single evidence term.
- Supervisor-fed backlog task VFS-085 remains the gap-close vehicle; protected
  plan / objectives / todo / validator files were not edited.

## Validation

```bash
python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot --verify
```
