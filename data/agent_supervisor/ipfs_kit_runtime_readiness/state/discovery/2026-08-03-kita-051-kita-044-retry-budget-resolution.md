# KITA-051 validation retry-budget resolution for KITA-044

Date: 2026-08-03
Status: supervisor blocker repaired; semantic candidates rejected
Source task: KITA-044
Follow-up task: KITA-051
Failure kind: validation (`proposal_gate_failed` and sealed command environment)

## Root causes

The preserved KITA-044 attempts exposed two independent supervisor defects:

1. A task validation beginning with `cd ipfs_kit_py &&` received an inline
   `PYTHONPATH` assignment for only its first Python process.  A later Python
   command in the same validated chain lost the approved worktree package
   roots and failed with `ModuleNotFoundError`.
2. Scope adjudication derived Python module names only from the superproject
   root.  It could not prove an exact package-root import such as
   `ipfs_kit_py.core.performance` when the task's safe validation command
   established `ipfs_kit_py` as its repository-relative working root.

A third policy gap allowed dependency evidence to override a task whose own
conflict policy requires an exact predicted-file amendment.

## Supervisor repairs

- `17f612f86` exports the approved validation `PYTHONPATH` across the entire
  command chain, including safe leading-`cd` commands.
- `eab0e2d67` derives exact import aliases only from validated safe leading-`cd`
  roots; it does not trust candidate `sys.path`, arbitrary `PYTHONPATH`, parent
  traversal, absolute paths, or shell grouping.
- `79e0943b7` adds a fail-closed exact task-scope policy and preserves it on
  generated retry tasks.

## Verification

The focused and aggregate supervisor suites passed on the repaired tree:

```text
python -m pytest -q \
  test/api/test_agent_supervisor_scope_adjudication.py \
  test/api/test_agent_supervisor_validation_scheduler.py \
  test/api/test_agent_supervisor_todo_daemon_port.py

683 passed
```

The exact KITA-044 attempt-2 candidate was also replayed in an isolated
worktree: its 28 declared tests and benchmark command completed once the
validation environment repair was applied.  That replay proves the supervisor
mechanics only; it is not task-completion evidence.

## Semantic review and safe board revision

Independent review rejected both preserved candidates (`cef7f147` and
`f7a41db2`).  They construct a new synthetic baseline with 192 random draws per
operation while the KITA-043 fixture uses eight, amortize the new cost across a
batch, and report roughly 10x.  They do not call the production WAL, VFS,
bucket, ARC, GraphRAG, replica, router, or adapter paths.  Their durability,
authorization, integrity, replication, and consistency fields are largely
labels rather than the production contracts.

The review also found that KITA-043 routes every non-import/resource workload
through `MemoryTransactionEngine`, even for VFS, WAL, ARC, GraphRAG, replica,
and interface workload names.  No frozen baseline result artifact exists and
the reference floors are null.

KITA-043 is therefore reopened with an exact production-binding/SLO output
scope and a protected operation-level call and raw-evidence gate.  KITA-044
now owns the exact reviewed production hot paths and must preserve that gate,
the frozen harness and baseline, the ARC oracle, and all owning
correctness/security tests.  A second protected gate reruns the production
benchmark live and recomputes the required 2x aggregate committed-TPS result
from raw timings, so a hand-authored optimization artifact cannot pass.

## Unblock

This receipt closes the repeated supervisor validation failure represented by
KITA-051 and authorizes the normal one-time retry-budget reset for KITA-044. It
does not complete KITA-043 or KITA-044, approve either rejected candidate, or
relax any production correctness, durability, authorization, or evidence gate.
