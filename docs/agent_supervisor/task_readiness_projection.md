# Task readiness compatibility diagnostics

`DatabaseImplementationDaemon.materialize_task_state_compatibility_projection`
reads the canonical task source and writes a disposable status file. Its
`projection_authority` remains false. The file cannot authorize a claim,
release retained callback custody, accept a completion, or permit a source
transition.

| Field | Meaning |
| --- | --- |
| `todo_count`, `todo_task_ids` | Backlog with normalized todo status, including work with unmet dependencies. |
| `ready_count`, `ready_task_ids` | The canonical source's bounded readiness selection, including dependency and cooldown checks. |
| `eligible_ready_count` | Legacy numeric alias of `ready_count`, retained for existing supervisor consumers. |
| `eligible_ready_count_scope` | `source_readiness_only` on a complete observation; `unknown` on a failed observation. |
| `claim_eligibility_known`, `claim_eligible_count` | Always false and null: this diagnostic does not evaluate the complete claim-admission path. |

For example, a source-ready manual task can remain in `ready_task_ids` while an
automatic daemon reports `no_ready_tasks`. Lane selection, manual-task policy,
attempt budgets, existing claims, and retained callbacks have separate admission
checks. A dependency-blocked todo task belongs only in the backlog.

The native branch backports the canonical readiness scan from generic commit
`450ad3da99a01071e9faab61dcc83fa7320392de`. The existing task-projection validator
checks the ready set against the complete task population; the source revision
and content identity are checked around the read. Unavailable, changed, foreign,
or potentially truncated ready results leave a nonterminal incomplete marker.
For such a marker, `todo_count` is null and `readiness_scope` is `unknown`.

The DOEP observation that all 42 projected todo tasks had unmet declared
dependencies is conditional on the mutable lane statuses and declared Markdown
graph used for that comparison. It neither admits the projected 40 completions
nor settles the original DOEP031 callback. Tests use disposable real DuckDB
sources and the existing validator; no live task or owner mutation is required.
