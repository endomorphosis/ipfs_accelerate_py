# Native dependency preflight recovery

A failed dependency probe must remain a known environment wait when Native
proves it ran before provider dispatch. Generic deferrals, empty receipt tables
and a missing worktree do not establish that boundary.

`todo_daemon.dependency_preflight_retry` joins the exact sealed task projection,
database claim, intact Native event chain, stopped execution state and completed
workspace cleanup. It requires either the exact terminal lifecycle or Native's
matching terminal compare-delete cleanup event, absent branch and absent task
index. Any provider-start, merge or task-completion event refuses this recovery.
An older unknown callback can be settled only with this same positive evidence.

The database owner replaces only its exact pending intent through a CAS. The
known wait keeps the original attempt, paid history and implementation deadline,
honors Native's backoff, and permits at most eight observations. Before retrying,
the owner rechecks its claim and Native's refusal, then durably restores dispatch
intent. A later interruption therefore retains unknown custody normally.

Deadline or retry exhaustion reports `validation_dependency_repair_required`.
Repair packages in the existing approved validation environment, run its sealed
metadata probe again, and use the normal operator recovery after an exhausted
deadline. This mechanism does not enable automatic global package installation,
reset task budgets or weaken validation, merge or completion requirements.

Regression coverage uses a real bounded child-probe timeout, real database
leases, actual worktrees, pytest validation and Native merge/completion. The
provider in these isolated tests is an explicitly authored local shim. Tests also
cover legacy unknown intent recovery, unchanged custody after a modified event
chain and refusal to dispatch after the original deadline.
