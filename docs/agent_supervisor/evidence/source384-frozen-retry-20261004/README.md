# Frozen archive retry: post-worker memory-pressure refusal

The user-requested retry uses independent copies of the exact prior runtime
archive and manifest, with the same checkpoint, reviewed intent mapping,
five CPUs, 12 GiB and existing deadlines. The earlier archive member audit is
reused after checking archive/manifest hashes and live production pins. No
additional live container diagnostics are injected into this ordinary run.

Deployment passes in 258.012s. Native preparation takes 11.469s; initial context
fails after 99.094s at the final observe_current call in infer_shared_parent_units,
after inference has been saved and reloaded. The native probe takes 111.573s;
the outer controller takes 449.676s. These intervals are nested, not additive.
The new qualifier bulk-copy release and warm replay boundary is not reached.

After unwind, the scheduler retains proof_memory_stall backoff. The later
resource sample reports 14.59 percent memory stall, 3.45 percent I/O stall and
11320 MiB available. Those later observations cannot establish the exact
admission-time values or cause. No completed inference artifact was exported,
so this run supplies no new inference coverage or model-load count.

All frozen source and public task pins are unchanged and container cleanup is
verified. No full task trial is launched after the failed prerequisite: there
are zero provider calls, no official verifier execution and no new reward or
completed token score. The backlog remains 18/32 production-closed. The later
admission-observation change is separate component evidence; it was not part
of this retained run and has not qualified the frozen runtime.
