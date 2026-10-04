Pressure-source attribution component controls

76 tests passed with no skips/failures/errors. This package retains the exact six tested source/test files, two pre-edit owners, command, source hashes, exit receipt, XML, and log. It contains no scheduler/key/seal stores, model payloads, benchmark verifier bodies, or credentials.

The sampler still takes the original host and visible cgroup/ancestor pressure reads. It retains host plus at most eight anonymous cgroup rows and omitted-scope counts/maxima. Memory/I/O use full avg10; CPU uses some avg10. Missing/malformed optional PSI still has the original scalar fallback while diagnostic values are explicitly null. Existing max aggregation, thresholds, backoff, sampling cadence, and root/child accounting remain unchanged. Optional metadata is excluded from resource snapshot equality and not written to the shared ledger or scheduler configuration.

The scheduler extension is request-local and diagnostic only. Legacy/custom snapshots omit attribution. A projection failure omits only the optional addition while retaining the valid scalar sample/gate. These controls do not establish new native qualification, speedup, changed memory needs, or attribution of the prior aggregate failure.
