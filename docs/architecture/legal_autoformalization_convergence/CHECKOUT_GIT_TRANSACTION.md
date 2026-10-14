# Native checkout Git transactions

The phase 6 interrupted merge exposed an index race: native status scans could
refresh the shared index outside the board merge lease, and a completion rewrite
could begin before commit serialization. A Git `index.lock` failure was then
treated as a content conflict, leaving partially staged output and invoking the
merge-resolver compatibility process. Its existing runner-identity guard refused
before provider dispatch. This repair leaves that provider guard unchanged.

## Guard and API

The owner is
[`merge/git_checkout_transaction.py`](../../../ipfs_accelerate_py/agent_supervisor/merge/git_checkout_transaction.py).
Its public helpers are:

```python
git_checkout_directory(repo: Path) -> Path
checkout_git_mutation(repo: Path, *, timeout_seconds: float = 30.0)
serialize_checkout_git(*, repo_parameter: str | None = None)
git_lock_contention(result: subprocess.CompletedProcess) -> bool
capture_checkout_index(repo: Path) -> dict
retain_git_contention_custody(repo: Path, *, before_index, task_id,
                             branch, before_head, result) -> dict
```

`checkout_git_mutation` is a context manager. It resolves
`git --no-optional-locks rev-parse --absolute-git-dir` and holds the persistent
`agent-checkout-git-mutation.guard` inode in that **per-worktree** Git directory.
A process-local thread lock and POSIX advisory lock serialize cooperating workers.
Nested use in the same thread is reentrant. Different board namespaces sharing
one checkout use the same physical guard; sibling worktrees have separate indexes
and guards, preserving their concurrency. The existing board authority lease
remains in force and can impose its own repository-wide constraints.

The guard requires a regular, singly linked inode, refuses symlinks and inode
replacement, and never unlinks its advisory file. Existing `index.lock` and
`HEAD.lock` paths cause refusal before mutation; unknown lock ownership is not
deletion authority. `GIT_DIR`, `GIT_WORK_TREE`, or `GIT_INDEX_FILE` overrides also
cause refusal. This coordinates the instrumented native paths; it does not lock
out an external Git process that ignores the advisory guard.

## Coverage and ordering

The integration in
[`todo_daemon/implementation_daemon.py`](../../../ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py)
guards these complete operations:

- `_run_checkout_mutation_transaction`: physical checkout guard, existing board
  authority/recovery lease, then callback and publication/release checks.
- `_mark_tasks_completed_in_todo` and `_mark_tasks_ready_in_todo`: the full
  completion/reopen path, including Markdown rewrite and its generated commit.
- `_commit_specific_path`: inspection, staging, and generated-path commit.
- `_commit_worktree_changes_unchecked`: candidate worktree staging and commit.

The merge transaction additionally guards an already checked-out target before
workspace preparation can restore paths, and guards the selected merge workspace
through mutation and cleanup. Those target guards are acquired under the existing
board lease; nested acquisition of the same physical checkout is reentrant.
Child/target checkouts retain their own guards rather than sharing an index guard
by Git common directory alone. Keep this ordering when extending these paths;
holding a taskboard write lock before entering an unrelated checkout transaction
can introduce a lock inversion.

Native daemon, supervisor, merge-train, activity-snapshot, pytest-ledger, and
interrupted-validation status readers explicitly use `--no-optional-locks`.
The daemon's `_run_git` also supplies that option. Control/launcher Git metadata
calls retain `GIT_OPTIONAL_LOCKS=0`. These prevent optional status index refreshes;
they do not disable mandatory locks for mutating Git commands.

## Deferral and retained custody

`checkout_git_mutation_busy` and `checkout_git_foreign_lock_present` are native
transient merge-reconciliation reasons. They preserve the validated candidate for
retry. They do not prove completion or authorize a provider call. The existing
MergeTrain non-consuming board-lease deferral still independently verifies the
exact live lease, repository, inode, PID, task and branch; a reason string alone
does not establish that proof.

The semantic checkout transaction accepts `timeout_seconds`, defaulting to zero:
ordinary operations allow at most 5 seconds; callback requalification allows at
most 30 seconds. Values must be finite numeric values, excluding booleans.
Only callback requalification can boundedly wait for a revalidated live foreign
protected owner. It cannot adopt or clear that owner's journal. Failure results
retain the exact owner lease ID and wait diagnostics. This bound covers semantic
lease acquisition, separately from the physical guard wait and callback runtime.

An explicit Git lock-file failure during merge is an infrastructure failure.
It bypasses content repair, model resolution, and automatic merge abort/reset.
The code preserves an ephemeral checkout **before** custody publication, so an
I/O failure cannot authorize force-removal of partial stages or foreign locks.

Custody records live in the affected per-worktree Git directory:

- `agent-git-contention-custody.pending.json` is published and synced first.
- `agent-git-contention-custody/<id>/` retains exact bounded before/after index
  snapshots and an observational `receipt.json`. An unstable after-snapshot is
  explicitly unavailable, not certified.

The pending marker yields
`checkout_git_retained_contention_requires_recovery`, with
`checkout_mutation_recovery_required=True`. It retains the candidate for native
reconciliation but is **not** an ordinary transient deferral. Removing the
unrelated foreign writer's lock does not clear this barrier. A receipt-publication
failure returns `git_lock_contention_custody_failed` and preserves the checkout.

Before any future recovery/replay, independently revalidate the candidate commit,
target HEAD, current index/worktree, retained snapshots, receipt lineage, and
owner/process state. Missing or contradictory evidence remains blocked. This
change supplies no automatic custody-clear operation and grants no authority to
remove foreign locks, reset user data, or mark a task complete.

## Qualification status

On 2026-10-13, **66 cases passed with zero skips, failures or errors** under the
unchanged native VALIDATION lease: 21 new concurrency/custody cases plus 45
existing checkout-lock and merge-train deferral cases. The new suite includes an
actual merge paused in a Git hook while another board attempts completion,
independent sibling-worktree mutation, foreign staged/unstaged preservation,
partial-stage custody, and failed-publication preservation of an ephemeral
checkout. Targeted peer review found no remaining concrete blocker.

The exact test files are
[`test_agent_supervisor_checkout_git_serialization.py`](../../../test/api/test_agent_supervisor_checkout_git_serialization.py),
[`test_agent_supervisor_checkout_lock.py`](../../../test/api/test_agent_supervisor_checkout_lock.py), and
[`test_agent_supervisor_merge_train_lock_deferral.py`](../../../test/api/test_agent_supervisor_merge_train_lock_deferral.py).
Evidence is retained under
`/home/barberb/lift_coding/artifacts/legal-autoformalization-source-custody-20261013-01/`
as `checkout-git-final.stdout.log`, `checkout-git-final.junit.xml`, and
`checkout-git-final-lease.json`. The lease was released and scheduler policy
unchanged. Earlier failed runs remain retained.

Phase 7 native three-lane qualification is **pending**. These checks establish
the bounded checkout contracts; they do not establish legal semantics, global
graph consistency, training quality, or Lean admission.
