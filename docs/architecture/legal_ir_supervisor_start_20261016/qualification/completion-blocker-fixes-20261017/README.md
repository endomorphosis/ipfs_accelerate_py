# Completion blocker fixes

The ordinary provider observer treats process disappearance (`ENOENT`/`ESRCH`) during `/proc` reads as normal exit while preserving refusal for permission and identity failures. The legacy Docker watchdog removes the exact generated launcher files after successful container cleanup and refuses unexpected entries.

45 focused tests passed, followed by actual Docker normal-exit and abrupt-worker cleanup regressions with both generated launchers bound read-only. The native parent independently verified the exact container, watchdog and lease directory were gone, and finalized successfully.

The dead launch03 parent was stopped only after its native trees and exact provider container/watchdog were already gone. The datasets SDK subsequently reconciled only its dead active storage claim and verified a fresh actual postwrite census. The old failure and artifacts remain preserved.

68 additional SDK/acceptance cases and 31 subtests passed. The first result wrapper failed because it expected 68 in the JUnit suite total, which includes 31 subtests; the preserved actual XML has 68 testcase elements, zero errors/failures/skips and total 99. A separate assessment records this without relabeling the failed wrapper.

First-task native factual acceptance still needs a genuine fixture roundtrip and a fresh execution packet with supported report instructions. Preserve the two consumed attempts and canonical task identity. The next 63 tasks stay blocked until their own validators and phase prerequisites are materialized. No new training, weights, legal semantic review or Lean build was performed by these infrastructure fixes. GitHub hosted CI remains blocked by account billing.
