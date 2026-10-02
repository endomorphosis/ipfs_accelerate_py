# Repository pipeline resource evidence

The final controlled suite passes 31 cases in 10.81 seconds with injected host
telemetry, real file-backed owners and actual subprocesses. All six development
attempt logs are retained; earlier failures are not counted as qualification.

One actual-host attempt was refused by `host_disk_high_watermark` before work,
in 0.60 seconds. No admission limit was relaxed. This attempt preceded the final
unsafe-exit GC guard and qualifies neither that guard nor the complete pipeline.

`producer-sources.json` pins the three additive files and existing owner
dependencies. No checkpoint, raw owner database, lease capability key or policy
private key is included. Full RPI-022 remains open; this evidence concerns only the
explicit CPU sampled composition and its separately identified controlled checks.
