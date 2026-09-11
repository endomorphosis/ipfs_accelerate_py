# Retained workspace custody

A board with an unresolved callback can preserve its whole configured workspace
root without guessing which historical workspace belongs to that callback.
The registry lives in the repository's Git common directory, so sibling
worktrees observe the same fences. A freeze records the exact bounded pool,
lifecycle, and shared-claim population and reserves a fresh sibling root for
independent work.

Native pool mutations, lifecycle transitions and exact native deletion, provider
workspace operations, and shared claim updates hold a shared custody lock.
Installing a freeze requires the exclusive lock. Retained claims remain occupied
even after their original owner exits. Workspace mutations and repository-wide Git cleanup also hold enclosing Git
store locks, so a nested supervisor cannot change a parent-frozen workspace or
prune retained parent-board registrations or objects.

The filesystem protocol is compatible with the native PCTDD quarantine branch.
It can be deployed with an empty registry before a board installs a freeze.
Every process that can mutate the affected Git stores must load these guards
before the first freeze; changing source files does not update an existing
Python process. The board's central task fence, execution acknowledgement,
callback custody audit, and independent-task admission remain separate native
requirements. Filesystem custody grants no task completion or callback retry.

The tests use real Git worktrees, lifecycle records, shared claims, subprocess
locks, and FIFO rejection. DOEP's exact deletion, dead-owner adoption and
finalization, direct supervisor rescue, and completed rescue pruning preserve
frozen custody. Task-projection peer cleanup keeps its existing inert refusal;
canonical background cleanup enters shared custody before any Git operation.
The supervisor's existing refusal to clean canonical callback workspaces without
its guarded native cleanup API remains unchanged. Native lifecycle APIs and
canonical callback authorities from other forks are not imported.
