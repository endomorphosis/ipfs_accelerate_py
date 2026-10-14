# Historical inventory nominations for bounded task source scopes

The [source-scope adapter](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_source_scope.py)
selects an explicit reviewed subset from a complete
[retained committed-source inventory](repository_source_inventory.md). It reuses
the caller's existing CAS and cold loader, preserving the full repository,
commit/tree/population and inventory identities. Its output contains metadata,
never source bodies. It creates no catalog, session, model or native task grant.

This is a historical nomination. The capture request's `task_id` remains its
original provenance label; it is not a current native target-task identity.
Current-source, allocation, task custody and accepted proof checks remain with
their existing owners. A caller cannot use this result to broaden an already
admitted worker scope or rewrite a signed task profile.

## Nominate an explicit subset

```python
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_scope import (
    nominate_source_scope,
)

scope = nominate_source_scope(
    cas=existing_cas,
    expected_inventory_cid=source_ref.inventory_cid,
    request=source_request,
    limits=source_limits,
    selected_paths=tuple(sorted(path.encode().hex() for path in (
        "instruction.md", "main.py", "smoke.py",
    ))),
    required_paths=tuple(sorted(path.encode().hex() for path in (
        "instruction.md", "smoke.py",
    ))),
    program_paths=(b"main.py".hex(),),
)
metadata = scope.observation()
frontier = scope.frontier_page(offset=0, limit=64, max_bytes=16384)
arguments = scope.semantic_inputs()
```

Selection is an exact sorted, unique tuple of canonical lowercase raw-path hex.
Required paths are a nonempty selected subset. Program paths are another
explicit subset, which may be empty; no roles are inferred. Selected files must
be regular committed `100644`/`100755` entries with printable canonical UTF-8
paths and UTF-8 content without NUL. Absolute paths, aliases, `.git` components,
backslashes, control bytes and paths over 4,096 bytes refuse. Language support
and semantic correctness are not established by eligibility for nomination.

The maximum is 64 selected files and **1,000,000 total original bytes, counted
per selected path**. This
uses the semantic producer's stricter default rather than the reader's 1-MiB
ceiling. Duplicate content counts once for each selected path. Mandatory
instruction, smoke and profile support consume the same file/byte limits and
must be selected explicitly. Unknown or unsupported requested files and any
overflow refuse the whole result; there is no truncation or implicit append.

`semantic_inputs()` returns detached `paths`, `required_raw_paths`,
`program_paths`, `max_files` and `max_source_bytes` arguments for the existing
semantic producer. It does not call that producer or supply a repository,
objective, task identity, current-source attestation or native grant. The caller
must first establish the actual current originals and reviewed mandatory scope.
The independent typed source channel has different limits: 256 host-admitted
paths, narrowed to 32 by its coding-child broker. A nomination does not override
any downstream channel's smaller limit.

## Keep the complete frontier

Every committed entry is either selected or retained in the frontier. Rows bind
raw path, Git mode/OID, source CID, SHA-256 and original length; unsafe display
paths are null. Gitlink source fields are null, retaining the child commit
reference without following it.

| Frontier status | Meaning |
| --- | --- |
| `not_selected_eligible` | Readable historical text outside the reviewed subset |
| `symlink_requires_separate_scope` | Retained link-target bytes, never followed |
| `gitlink_requires_separate_repository` | Child commit requires its own repository scope |
| `unsupported_path` | Raw path is outside the common reader's UTF-8/path rules |
| `unsupported_source_encoding` | Source is non-UTF-8 or contains NUL |
| `source_exceeds_task_byte_bound` | One original is larger than the task ceiling |

The scope and complete frontier receive content identities. These identify the
metadata; this adapter does not persist those new objects or promise CAS
availability. Observation and page exports are detached and retain false proof,
execution, completion, omission, current-source and task-admission flags.
Direct construction of the immutable value checks metadata self-consistency;
historical provenance requires the factory with independently pinned CAS inputs.

Pages accept offsets, up to 64 rows and a 2-KiB–64-KiB canonical JSON body budget.
They shorten only the number of complete rows and report exact remaining
positions. A budget that cannot hold one complete row refuses. Empty final
pages are explicit. These are host metadata pages, not authenticated worker
requests or one-use source cursors.

Cold loading verifies **every retained blob**, including unselected source.
Each page also revalidates and hashes the complete in-memory frontier. Output
pagination makes neither acquisition nor host work lazy. No latency, native
token reduction, task advantage or proof-cache reuse is established here.

## Join an existing live native task

The optional [native nomination binder](current_source_task_admission.md#bind-a-pinned-historical-nomination)
cold-replays this exact scope with independent pins before deriving a selected
path binding through the existing task owner. Its detached observation joins
historical identities to that actual live task, reader and context; it does not
change this object's historical flags or establish allocation, coding-child or
signed Terminal profile admission. The existing native broker's 32-path ceiling
remains independent of the nomination's 64-file ceiling.

## Join the existing managed allocation

The optional [allocation nomination binder](current_source_worktree_allocation.md#bind-a-pinned-historical-nomination)
now supplies its retained managed workspace to the existing task nomination
binder. It reuses the actual sealed Portal projection, lifecycle records,
native claim, original CAS and source reader. The caller independently pins the
scope and inventory inputs; a detached nomination or allocation receipt cannot
construct a live handle.

`require_nominated_source_scope()` on that allocation returns
`supervisor-native-nominated-source-worktree-allocation@1`, joining the source
and frontier identities to the actual `allocation_cid` and reporting allocation
ownership true. The task-only observation still reports allocation ownership
false, and the nomination remains historical. Bootstrap, compiled-context,
coding-process and proof/execution/completion/omission authority are not supplied
by this join.

The allocation checks current native task/lifecycle custody and complete
allocated originals. The broker independently joins full canonical and allocated
source-byte currentness and enforces its 32-path limit before granting its proxy
route. The nomination's 64-file/1,000,000-byte ceiling includes selected required
supports; it does not increase that worker bound or allow frontier reads.
Already published readers remain bind-once even if a later check fails.

The [composition evidence](evidence/nominated-source-allocation-20261014/README.md)
retains the new binder and observer boundary. Actual coding-child authentication,
signed-profile-to-nomination identity binding, source expansion and complete
session token measurements remain separate integration work.

## Reconcile the parallel repository stack

A read-only local survey found repository-planning commit
`24bafa1179d4d754b58d7d77080d811424404cca`, including context compiler
`ce2cd1026f21e52384ddf29c50c4a459baef927e`, on clean parallel worktrees.
Neither is contained in the source base
`5ad92f61bb643d4138749d3df16663a2af1c79c1`. The stack already implements
repository sessions, capsule queries, bounded compilation and task-proposal
consumers; these should be reused rather than recreated. The specific matching
remote branch names were absent when queried, so local references alone do not
establish GitHub publication under those names.

Its 21 commits diverge from 89 main-side commits and overlap 22 files. The older
`source_view_policy.py` helper and current JSON recipe class use the same module
and schema name for different contracts. Its `--source-view-policy` selection
also differs from the current preparation recipe. A blind merge would conflate
these meanings. The [survey receipt](evidence/repository-source-scope-20261013/parallel-work-survey.json)
retains exact commits, hashes, worktree state and queried refs.

The concrete integration order is:

1. Separate the older helper module, wire schema, CLI option and keyword from the
   current `SourceViewPolicy` recipe, with explicit historical readers and no
   ambiguous fallback.
2. Merge the existing repository packages, sessions, compiler and planning
   consumer against current role-context work, preserving required evidence,
   default configuration shapes and family/schema/task checkpoint inventories.
3. Review native catalog/index ownership and reuse the admitted owner's
   interfaces; distinguish session metadata from each IR/dimension inventory.
4. Reuse the opt-in [signed Terminal context selection](signed_terminal_source_scope.md)
   for single Python-code tasks. Its selected context/vector subset lives beside
   complete signed `task_profile.input_paths` and includes mandatory supports.
   Join historical nomination identities explicitly; data/multitask selection
   and the production coding-child read handoff remain separate work.
5. Qualify native task/allocation fences, actual coding-child adoption and stale
   source refusals before matched full-session token/task experiments.

The new nomination API supplies only the distinct metadata-selection seam. The
parallel stack's authored host-side experiments do not establish live Terminal
worker adoption or billed token savings. No foreign worktree, decoder weights,
embeddings or model-manager registration is changed by this increment.

The [qualification bundle](evidence/repository-source-scope-20261013/README.md)
retains this increment's selected regression results and limits separately.
