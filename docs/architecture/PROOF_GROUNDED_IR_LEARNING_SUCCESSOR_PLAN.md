# Proof-Grounded IR Learning Fabric successor execution plan

Status: admitted continuation candidate

This plan moves the `PGIR-200..207` successor board onto the current
`ipfs_accelerate_py` agent-supervisor control plane without rewriting the
historical campaign. The 2026-08-18 `RESULT(PGIR-111)` decision remains a
truthful `no_go`; it is an input and completion anchor, not evidence that a
learned campaign ran.

## Source generation

- Accelerator base: `22173f9cf4f357ab20040024f87af53c1cd89c9a`.
- Datasets base and accelerator gitlink: `c30ccbec997868b061c4cadac38d30468c46ea2d`.
- Both bases contain their repository's accepted Proof-Grounded IR Learning
  merge history.
- Historical taskboard, objective heap, final report, and next board are
  operator-protected inputs.

Changing either source generation requires a new inventory receipt and a new
configured-board revision before a worker may claim a task.

## Execution projection

`docs/architecture/proof_grounded_ir_learning/successor.todo.md` is the
executable projection. It contains four completed historical anchors needed
to close cross-board dependencies and eight open task revisions. The task IDs
are retained, while source bindings and output paths are revised for the
current checkout.

All new immutable artifacts live below a `successor-v1` directory. Workers
must never replace the historical freeze, experiment, qualification, report,
or next-board bytes in place. The runtime lives under
`data/agent_supervisor/proof_grounded_ir_learning/successor-v1/runtime` and is
not completion authority.

## Supervisor contract

The sealed scheduler config uses the current
`configured_board_scheduler.py` entry point, an explicit legacy-Markdown
task-source projection, two fenced implementation lanes, isolated worktrees,
the initialized `ipfs_datasets_py` submodule, and a direct Codex provider
route. Markdown is retained only as the admitted source format for this
eight-task legacy successor; task identity, readiness, leases, validation,
and merge admission use the current supervisor implementation. Every task
declares the `proof-grounded-ir-learning-successor-v1` board namespace and an
explicit goal identity so no runtime identity falls back to the board filename.

Launch is fail-closed:

1. The current branch, required ancestor, nested gitlink, and nested HEAD must
   match the sealed config.
2. The continuation validator and configured-board preflight must pass from a
   clean checkout.
3. Provider dispatch requires the operator-visible `--implement` flag.
4. Training remains forbidden until `PGIR-205` emits a superseding freeze
   whose gates explicitly authorize descendant execution.

## Acceptance path

The dependency path is:

`PGIR-200 -> PGIR-201 -> PGIR-202 -> (PGIR-203 || PGIR-204) -> PGIR-205 -> PGIR-206 -> PGIR-207`.

`PGIR-200` may admit rows only from exact, cited source and transformation
rights. Ambiguous metadata remains quarantined. A permanent-zero decision is
a valid task result but keeps learned training ineligible. Likewise,
`PGIR-203` may issue a deterministic-only restriction and `PGIR-204` may
retire the historical baseline. These outcomes can close the board honestly
but do not qualify a learned model.

`PGIR-206` may run R1-R6 only when the superseding freeze authorizes it.
Otherwise it must emit typed `not_run` evidence. `PGIR-207` must resolve all
sixteen criteria and thirty-two report sections and may publish only with an
independent current authorization.

## Evidence portability

The historical freeze, experiment, and qualification replayers do not all
pass from current main. The successor generation therefore must:

- replay the unmodified predecessor freeze in temporary shared clones at
  accelerator commit `04fbb09b4a8b34e77d11bd8da6642e0978baa02c` and datasets
  commit `b20bd9e3cfae79e8888929daf64f52b2f8a5689a`, using
  `scripts/verify_proof_grounded_ir_learning_predecessor.py`;
- bind current Git commits and the recursive repository forest;
- track every immutable JSON named by its manifest despite the repository's
  general JSON ignore policy;
- make build and verify commands agree on the artifact population;
- verify from a fresh recursive checkout before completion; and
- preserve a no-go, rejection, or resource-exhausted result without invented
  metrics whenever an admission gate stays closed.
