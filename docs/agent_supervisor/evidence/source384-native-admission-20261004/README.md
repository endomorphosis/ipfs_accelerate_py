# Native admission observation: post-worker headroom refusal

The new archive includes the request-local diagnostic producer/consumer and
three concurrent training-module changes; no training is performed. Checkpoint,
GTE assets, reviewed intent, five CPUs, 12 GiB and deadlines remain unchanged.
All 26,928 archive members pass streamed hash, size, mode and ownership checks.

Deployment passes in 170.910s. Native preparation takes 10.931s; initial context
fails after 96.855s at the first source-currentness observation after worker
return and output validation. The probe takes 108.758s and the controller
348.717s. Source/task pins and container cleanup are verified unchanged.

The attached request-local record captures the last fresh primary-gate sample:
8601 MiB available, 6144 MiB reserved by roots, zero additional child demand,
and 2458 MiB required headroom. This is one MiB below the gate's 8602 MiB
requirement. Memory stall is 0.01 percent, I/O stall 0.07 percent and CPU stall
33.5 percent, all below their respective configured thresholds. The final gate
status is the resulting cooldown, 1.210s after that fresh sample. This explains
that recorded refusal, not the source of charged memory or every earlier poll.

The later post-unwind sample reports 8669 MiB available and remains separate.
The new diagnostic transport is exercised in the actual container; the complete
qualification still fails. No full task, official verifier or provider call is
launched, and no new benchmark reward or completed token score is available.
The subsequent construction-object lifetime patch is separate evidence and was
not present in this run. No admission policy, resource limit or deadline changes.
