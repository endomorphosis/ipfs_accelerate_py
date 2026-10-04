# Pressure-attributed runtime: Source384 deadline failure

The ordinary Docker qualification of archive
`55069552545d8332fced55dab5be23c967466c8d7ef812e3d6bd0f6bbec52b4a`
fails during the final source-currentness observation after inference has been
saved and reloaded. The deployed sources are accelerate
`e38a45781e6c3257c71739f055363f8fa56da816` and datasets
`8ca1ceed01f9b2b2667de5b41fb7672d917911e8`. The archive stream audit checks all
26,932 members; seven runtime files changed and three upstream training files
were added relative to the preceding retained archive. Checkpoint, embedding,
solver, reviewed intent and resource/deadline selections remain unchanged.

Deployment takes 285.209s. Source preparation takes 11.323s and initial context
fails after 98.113s with `LeaseTimeoutError` at the final `observe_current`
checkpoint under the unchanged 90-second Source384 work deadline. The native
probe takes 110.534s; the controller exits 1 after 466.765s. These nested stage
times are not additive. The failure receipt has no completed Source384 duration
or coverage record, so it cannot qualify the archive's inference/replay result.

No request-local admission observation is attached (`no_attached_observation`).
The later post-unwind sample reports 9170 MiB available out of 12288 MiB, memory
stall 0.64 percent, CPU stall 18.78 percent and I/O stall 0.89 percent. The scheduler
then reports no active leases, allocations, waiters, proof backoff or recovery.
Those later values do not establish the resource state at the timeout or explain
its cause. The new bounded host/cgroup attribution is not exercised by this
failure; its component controls remain separate evidence.

Exact source and public task input pins are unchanged, and container cleanup
passes. There are no provider calls, training steps, downloads or official
verifier execution in the Source384 probe. No full task follows the failed
qualification; no new reward, completed token score or matched-arm advantage is
claimed. Zero provider calls describe a failed preparation prefix. The backlog
remains 18 of 32 closed.

Only bounded reviewed metadata and traceback frame identities are exported.
Raw exception/source/model/credential/hidden-verifier bodies are excluded.
Retained-record hashes refer to local artifacts rather than package members.
The runtime's timeout checks are cooperative; an observed 98.113-second failure
does not indicate that the configured 90-second deadline was raised.
