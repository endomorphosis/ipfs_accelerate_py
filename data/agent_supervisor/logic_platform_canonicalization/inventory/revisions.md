# LPC-001 — Exact source revisions and intervening changes

Machine-readable companion:
`data/agent_supervisor/logic_platform_canonicalization/inventory/revisions.json`.

Task: **LPC-001** · Goal: **LPC-G010** · Program: logic-platform-canonicalization

## Implementation authority rule

**Current heads are implementation authority.**

Reviewed baselines are comparison pins only. Compare against them, then edit
from the current checked-out heads. Do not treat a newer reviewed baseline tip
as the production edit base when the campaign head is behind it. Record
intervening changes (this inventory) before any production edit.

## Reviewed baselines

| Repository | Reviewed baseline commit |
| --- | --- |
| `endomorphosis/ipfs_datasets_py` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| `endomorphosis/ipfs_accelerate_py` | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` |

Sources: plan §2, objectives heap, formal work plan metadata.

## Current heads (implementation authority)

| Repository | Branch | Current head | Checkout |
| --- | --- | --- | --- |
| `ipfs_datasets_py` | `agent/logic-platform-canonicalization` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` | `/home/barberb/lift_coding/external/ipfs_datasets` |
| `ipfs_accelerate_py` | `agent/logic-platform-canonicalization` | `ea11293bb996f052d620eae989f5377a956764b1` | `/home/barberb/lift_coding/external/ipfs_accelerate` (campaign worktrees under this repository) |

Scheduler binding: accelerate required branch
`agent/logic-platform-canonicalization`, required ancestor
`ea11293bb996f052d620eae989f5377a956764b1`.

## Ahead / behind counts

Convention: current head relative to reviewed baseline
(`left = baseline`, `right = current_head`).
`ahead` = commits only on current head; `behind` = commits only on baseline.

| Repository | Baseline | Current head | Merge-base | Ahead | Behind |
| --- | --- | --- | --- | ---: | ---: |
| `ipfs_datasets_py` | `ac82107e…` | `ac82107e…` | `ac82107e…` | 0 | 0 |
| `ipfs_accelerate_py` | `485edc08…` | `ea11293b…` | `ea11293b…` | 0 | 1245 |

- Datasets: exact match; no intervening commits.
- Accelerate: current head is a pure ancestor of the reviewed baseline
  (**1,245 commits behind**; merge-base is current HEAD).

## Dirty paths

| Checkout | Dirty | Paths |
| --- | --- | --- |
| datasets authority (`/home/barberb/lift_coding/external/ipfs_datasets`) | no | _(none)_ |
| accelerate authority (`/home/barberb/lift_coding/external/ipfs_accelerate`) | no | _(none)_ |

Aggregate dirty paths: **none** recorded for production sources at seal.
LPC-000 preflight required a clean checkout and tracked control files.
Inventory outputs under
`data/agent_supervisor/logic_platform_canonicalization/inventory/` are campaign
artifacts, not source dirty paths.

## Intervening changes

1. **Accelerate baseline delta** — Reviewed baseline
   `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` is 1,245 commits ahead of
   implementation-authority head
   `ea11293bb996f052d620eae989f5377a956764b1`. Baseline material is
   comparison-only until deliberately adopted.
2. **Datasets baseline delta** — None. Baseline and authority are identical at
   `ac82107e246b30e35a2bbdcf75e01370d22350c6`.
3. **LPC-000 control seal** — Operator-protected control files sealed on
   `agent/logic-platform-canonicalization` at
   `4672adcccf5c7f2106dcfd39489fd545396a2c78` after required ancestor
   `ea11293bb996f052d620eae989f5377a956764b1`. Control-plane seal is not a
   production semantic rewrite; protected paths stay operator-owned.
4. **Host campaign boundary** — Do not join or steal state from
   `incremental-proof-sealer-v1` or `state-laws-reindex`.

## Observation basis

- `docs/architecture/LOGIC_PLATFORM_CANONICALIZATION_PLAN.md` §2
- `docs/architecture/logic_platform_canonicalization.objectives.md`
- `docs/architecture/logic_platform_canonicalization.todo.md`
- `config/agent_supervisor_logic_platform_canonicalization_scheduler.json`
- `data/agent_supervisor/logic_platform_canonicalization/notes/lpc000_seal_receipt.md`
- `data/agent_supervisor/logic_platform_canonicalization/formal_work_plan.json`

Related identities: LPC-000 seal `4672adcccf5c7f2106dcfd39489fd545396a2c78`;
task dispatch tree `f4cbfa023da5353143a05e6d3e209de58df8ff0b`.
