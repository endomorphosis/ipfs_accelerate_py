# Process-local pipeline lifecycle cap

Seventy-two distinct controlled tests pass under final producer `894d82722446d67c48df76d4fb7821135c24eff663a530dcbd14743f25a0cdbf`: 39 pipeline/lifecycle controls, 30 managed preparation/delegation controls, and three driver fault controls. The first two groups use actual file-backed native owners/processes with explicitly injected host telemetry. The lifecycle overflow fixture uses two real unsafe parents and a reduced test cap of three; production fixes the cap at 32. Native lease expiry leaves the old Python contexts counted.

The actual default-host probe was refused by the unchanged `host_disk_high_watermark` policy before work (one failed test, 0.78 seconds). It is retained separately and provides no live acceptance claim. No threshold was weakened and no default host capacity was replaced by test telemetry.

The adapter now reserves a process-local slot before native admission. Thirty-one ordinary lifecycles and one cleanup-only lifecycle are permitted. Queued entries count; unsafe contexts stay strongly reachable and counted after native expiry. Explicit recovery preserves disk claims and never releases live children to make room. The recovery-only public APIs reject workload phases, process execution and native capacity delegation.

The initial controlled attempt exposed a fixture bookkeeping-capacity refusal, not a production failure; it remains in the archive. Existing earlier evidence is untouched. The exact former pipeline producer bytes are copied here as historical provenance, alongside current source hashes, the current producer, and all logs/XML.

See [the API rules](../../REPOSITORY_PIPELINE_RESOURCES.md) and [managed preparation scope](../../REPOSITORY_MANAGED_BENCHMARK_PREPARATION.md). Hard enforcement, GPU/device accounting, full host object accounting and gradual recovery remain outside this profile.
