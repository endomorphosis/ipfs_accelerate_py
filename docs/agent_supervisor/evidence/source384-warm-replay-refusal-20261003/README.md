# Ordinary deferred-import retry: warm replay lease refusal

This retry reuses the exact archive built for the first deferred-import attempt.
Its build command/exit records retain the original build location and time.
The archive changes only the full driver imports. Deployment and empty
native START/STOP pass, preserving the exact checkpoint, embedding assets,
solver profile, original public source population and declared deadlines.
The ordinary preparation qualifier does not import the full driver, so this
attempt does not measure the deferred imports' effect in a full task.

Initial context returns in 156.106s, including the probe's checkpoint/result
checks. The independent warm observation then refuses a child resource lease
after 30.316s. Preparation takes 10.953s and the probe takes 198.410s;
deployment takes 170.766s and the outer controller takes 437.029s. These are
nested scopes. No completed inference artifact is exported after the warm
failure. The full task is not launched after this failed prerequisite.

Actual cgroups enforce five CPUs and 12288 MiB. Initial availability is 9340
MiB; after unwind the sample is 8615 MiB and the scheduler retains a
proof_memory_headroom backoff reason. This is not an admission-time sample
or causal proof. Source pins and task input hashes remain unchanged, and
container removal is verified. There are no provider calls, official reward,
completed token score or claims of benchmark improvement.
