# Applicability replay budget: existing regression controls

The final existing regression group passed 162 tests, with one explicitly skipped optional checkpoint/GTE case and no failures or errors. Wall time was 186.986 seconds. All seven changed production owners and the selected dependency/test pins remained unchanged. The command records fresh isolated seal, key, scheduler and orchestration paths. It ran the driver/profile controls, captured-header applicability, local planning admission, symbolic intent planning and terminal intent planning.

The omitted case is `test_normal_initialization_with_real_checkpoint_produces_nomination_before_planning`, which requires an explicit pinned checkpoint and cached GTE snapshot. This package does not qualify checkpoint inference or a new Docker benchmark generation. The separate mandatory native qualification must cover that boundary.

The separate replay-budget component (manifest `dd8a2c6711f47144cf83f04b49a826c47ce705e25c7d01fa3d0239a6ee94ce11`) has 40 distinct passing new controls, including actual captured-source/Z3 admission recovery after a46-second reservation and native constructor/START/bootstrap/STOP. Its membership and digests were independently checked here. Those40 are not duplicated in this163-execution existing-test group. Combined there are202 distinct passing controls and one explicit skip.

Changes preserve ordinary45-second applicability and existing60-second validation-command limits. Explicit local benchmark replays may use120 seconds, bounded by the original work/lifetime and per-START deadlines. Native scheduler pressure/headroom/source/proof gates remain mandatory, and solver execution remains capped at five seconds. No benchmark score, host-pressure recovery, token saving or comparative performance advantage is established by these controls.

A source audit identifies a remaining handoff limitation: the separate typed-owner completion RPC callback has no captured replay scope and retains ordinary45-second checker behavior. Adding only a callback context would be insufficient because validation result publication also needs deadline checks. That follow-up is unimplemented; the current measured failure was constructor admission, not this later callback.

The original and final owner hashes are retained. Test XML contains outcomes and the explicit skip without stdout/stderr streams. No private stores, credentials, model bodies, hidden verifier bodies, benchmark task source bodies or test logs are exported.
