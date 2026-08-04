# SwissKnife / IPFS Kit VFS Pilot Artifacts

**Task:** VFS-085  
**Goal:** VFS-G165 (parent VFS-G131)  
**Evidence term:** `vfs/swissknife-vfs-pilot@1`  
**Bundle:** `vfs-assurance/rollout`  
**Track:** assurance-rollout  

This directory is the durable declared output for the frozen SwissKnife and
IPFS Kit VFS pilot. It proves `vfs/swissknife-vfs-pilot@1` by materializing the
deterministic pipeline artifacts produced by
`ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot`.

## Evidence bindings

| Artifact | Role |
| --- | --- |
| [`swissknife_vfs_pilot.receipt.json`](./swissknife_vfs_pilot.receipt.json) | Typed gap-close receipt for `vfs/swissknife-vfs-pilot@1` |
| [`report.json`](./report.json) | Full pilot report (`schema` / `evidence` = `vfs/swissknife-vfs-pilot@1`) |
| [`artifacts.json`](./artifacts.json) | Published artifact CID set |
| [`manifest.json`](./manifest.json) | Admitted-file manifest (SwissKnife + VFS closure) |
| [`coverage.json`](./coverage.json) | Exhaustive coverage accounting |
| [`inventory.json`](./inventory.json) | Forest inventory binding |
| [`forest.json`](./forest.json) | Frozen multi-repository forest |
| [`graph.json`](./graph.json) | Program graph for admitted files |
| [`contract.json`](./contract.json) | Contract-stage summary |
| [`cache_receipt.json`](./cache_receipt.json) | Analysis-cache receipt |
| [`proof.json`](./proof.json) | Shadow proof stage (non-authoritative) |
| [`zk_shadow.json`](./zk_shadow.json) | ZK shadow stage (non-authoritative) |
| [`findings.json`](./findings.json) / `finding_ledger/` | Content-addressed finding ledger |
| [`taskboard.json`](./taskboard.json) / `taskboard/` | Bounded, goal-backed repair taskboard |
| [`findings_board.md`](./findings_board.md) | Local copy of the durable findings board |

Durable findings board (sibling declared output):

- `docs/architecture/ipfs_kit_vfs_symbolic_assurance.findings.todo.md`

## Exact-term identifier

- `vfs/swissknife-vfs-pilot@1`

## Acceptance demonstrated

- Every admitted file is accounted for in the pilot manifest and coverage.
- Every finding is reproducible from content-addressed evidence (ledger + CIDs).
- Inconclusive results remain non-executable review records.
- The repair board is bounded, deduplicated, goal-backed, and independently reviewable.
- Provider calls and source mutations are zero; repair is not authorized.
- SwissKnife remains read-only under the initial forest policy.

## Authority limits

- This package is **not** completion-authoritative.
- Receipts set `safe_for_completion_reasoning=false`, `authoritative=false`,
  `authorizes_repair=false`, and `is_completion_evidence=false`.
- Proof and ZK stages remain shadow / non-authoritative.

## Validation

```bash
python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot --verify
test -d data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/pilot
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/pilot/report.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/pilot/swissknife_vfs_pilot.receipt.json
grep -F 'vfs/swissknife-vfs-pilot@1' \
  data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/pilot/swissknife_vfs_pilot.receipt.json
grep -F 'vfs/swissknife-vfs-pilot@1' \
  docs/architecture/ipfs_kit_vfs_symbolic_assurance.findings.todo.md
```

## Backlog alignment

- Objective heap goal VFS-G131 owns the pilot execution; VFS-G165 is the
  criterion-level evidence obligation for `vfs/swissknife-vfs-pilot@1`.
- Supervisor task VFS-085 closes that gap without editing protected plan,
  objectives, todo, or validator files.
- No further child-goal refinement is required: the full freeze → inventory →
  scan → graph → contract → cache → proof → zk_shadow → findings → taskboard →
  publish pipeline is already present and verified.
