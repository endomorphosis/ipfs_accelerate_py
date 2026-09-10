# Worker argument identity

Worker diagnostics prefer exact procfs arguments over empty or truncated process-list display text. A display string cannot override an available exact executable, module or script position. The decoder preserves empty downstream values, and recognition supports ordinary bundled Python flags while rejecting inline Python commands and malformed argument records.

This changes existing diagnostic recognition only. Generic main retains its existing sealed-runner receipt checks and ordinary worker diagnostic contract; it does not acquire the separate native PCTDD ordinary-provider receipt subsystem. No diagnostic match grants task completion, callback closure, retry or restart authority. The native PCTDD counterpart separately retains its stronger active-attempt ordinary receipt checks.

Regression tests cover missing and conflicting display text, empty arguments, interpreter/module/script positions and procfs decoding.

Both outer-supervisor and worktree-phase detection carry the supplied Portal task state into sealed-runner verification. A missing or mismatched attempt, revision, workspace, owner birth or runner birth cannot inherit another attempt's receipt. Worktree disappearance tracking includes the exact active implementation identity and phase launch boundary; heartbeat, progress and receipt observation changes do not refresh that tracking generation.

A provider launch can precede its new receipt and log. The outer supervisor therefore allows the existing bounded launch interval before considering an older log stalled, only for a valid exact attempt with an implementing/provider_launch_birth boundary and a nonfuture launch time. This grace does not recognize a worker or establish callback closure. Expired, absent and malformed launch clocks receive no log exemption.

A future attempt timestamp, like an absent one, cannot establish attempt-age grace. An independently checked live worker birth can still supply the existing bounded fallback using boot time; a missing, reused, expired or future process birth cannot. This preserves live work during wall-clock skew without treating the skewed timestamp as progress.

Outside the existing bounded implementation and merge handling, heartbeat freshness is not progress evidence. A task with queued work and no progress timestamp is stalled unless the caller supplies an explicit, unexpired projection grace. Evaluating these diagnostics does not write task state, progress, receipts or completion evidence.
