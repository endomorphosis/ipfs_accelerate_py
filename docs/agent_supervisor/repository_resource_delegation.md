# Private repository worker resource delegation

`repository_resource_handoff` lets an independently admitted local worker acquire
a child of the existing datasets global resource lease. The grant is a private
resource capability, separate from the signed repository/task evidence. It binds
the repository, task, owner process, host boot, scheduler state and monotonic
deadline. Native token validation remains authoritative.

The grant directory must be owner-only and outside the worker repository. The
exclusive, bounded regular file is read-only to its owner; symlinks, hard links,
incorrect permissions, changed bindings and expired or released parents are
refused. Never add grant files to model context, logs, Git or evidence archives.
Public receipts contain the parent lease ID and released child observation.

`repository_budgeted_behavioral_runner` checks its declared thread environment,
acquires the delegated native child and runs the existing signed behavioral
source fences before and after candidate materialization. It preserves the
independent task admission and complete task population.

The qualification CLI adds `--full-trial-resources`, which selects behavioral
evidence and explicit preparation. This model-off fixture declares one 3 GiB,
two-CPU/two-process parent with 1 GiB preparation and worker phases. All proof
consumers use children of that parent. Contiguous wall accounting includes
setup, queue waits, preparation, planning, worker startup through shutdown,
publication verification, successor reproof, cold comparison and cleanup.
Setup is charged to the trial deadline before resource admission. Enforcement
is cooperative with bounded native subprocesses, not a kernel cgroup.

Nine helper controls passed, including an actual subprocess and simultaneous
sibling validation, private-file failures, cancellation and released-parent
refusal. The full trial is **not yet qualified**: attempt 02 exhausted its
original oversized memory request; attempts 03 and 04 were refused by the real
95% host disk watermark. Attempt 04 also verified complete wall accounting on
admission failure. No host limits were relaxed. See the
[retained evidence](evidence/repository-resource-delegation-20261002/manifest.json).

Global disk reservation, retained payload queue bounds, per-device GPU budgets,
kernel enforcement and automatically protected capacity within each parent
remain separate RPI-022 work. Explicit sibling capacity in this fixture does
not establish those features.
