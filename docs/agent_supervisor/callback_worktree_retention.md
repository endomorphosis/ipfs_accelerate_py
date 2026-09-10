# Preserve callback recovery workspaces

An implementation branch can still point at the merge target while a provider
callback has an unknown outcome. A terminal worktree lifecycle record may mean
only that a dead process was fenced during restart. Neither fact proves task
acceptance or permission to discard its source evidence.

Portal workers with a task-isolated merge queue therefore skip background
cleanup of other worktrees. This includes the disposable projections used by
DuckDB execution callbacks. Their exact task/queue completion and qualified
no-effect cleanup paths remain available through `_cleanup_merged_worktree`.
The worker's requested cleanup count cannot grant authority over peer tasks.

For database-backed boards, supervisor backlog cleanup now calls the existing
canonical Quack completion verifier before inspecting or modifying a candidate.
Quarantined tasks, missing receipts, missing tasks, and unavailable owners keep
their workspaces. A Markdown completion mark is not a substitute. The proof is
read again under the mutation guard immediately before removal; a changed or
unavailable binding defers cleanup.

The existing verifier admits exact completed rescue identities. Those clean
rescue worktrees can be removed without force while their branch and commit are
retained. Ordinary database task branches continue through their native task
and merge-queue completion handlers; branch ancestry alone does not admit them
to background deletion. The cleanup event records the source HEAD and completion
proof. Existing Git identity, process, pool quarantine and mutation checks apply.

This prevents recurrence of source loss; it does not reconstruct already deleted
work or prove that an unknown callback had no effect. Such recovery still needs
the original preserved source and the existing native completion/no-effect proof.
Do not reset a quarantine or rerun a callback based on a missing worktree.

## Older SAWM native runtime backport

This deployment predates both the canonical cleanup verifier and the shared
pool mutation guard API. Its database supervisor therefore retains all
background candidates before taking a cleanup lease. Its Portal daemon uses
its native database-attempt authority binding to disable peer cleanup.
Exact task and merge-queue completion cleanup keeps its existing handlers.
No absent verifier is emulated and no completion is inferred from ancestry.
The compatibility tests exercise both native entry points and verify retained
source evidence. The full guarded verifier remains available on upstream main.
