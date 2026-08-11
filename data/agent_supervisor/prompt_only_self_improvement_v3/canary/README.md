# ASE3-013 Self-Host Canary Evidence

Disposable canary namespace artifacts for the prompt-only v3 self-host gate.

## Policy

- `monitor_policy.canary_observation_seconds`: **900**
- Observation clock is **monotonic** and starts only after the final recovery is healthy
- Any unhealthy sample **resets** the window
- No seed board / preseeded objectives
- ASE3-026 dual activation evidence required before start

## Layout

- `promotion_evidence_schema.json` — promotion evidence schema binding
- Runtime evidence files are written by hermetic tests under temp roots; release
  packaging may copy accepted promotion receipts here.

## Safety

Canary evaluation must never target the operator dirty checkout or authorize
merge/push/deploy.
