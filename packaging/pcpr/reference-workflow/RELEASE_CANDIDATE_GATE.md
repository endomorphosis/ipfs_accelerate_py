# PCPR-093 Accelerate release-candidate gate

These files record the Accelerate-owned hermetic release-candidate gate
run for the PCPR objective-to-release path. Supervisor, Datasets, Kit,
Accelerate, packaging, reference-workflow, interoperability, and
hard-safety gates are evaluated as named bounded records.

- `cpython312/reference.release-candidate-gate.json` is the declared
  gate-run document. Exact commit and tree are bound by the PCPR-093
  receipt `current_tree_binding`.
- `platform-release-candidate-gate-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, PCPR-090 threat-model, PCPR-091 TCB-inventory, and
  PCPR-092 audit-package identities are bound and not reminted.
- Failed, stale, partial, simulated, estimated, or missing evidence is
  recorded and never hidden. Live claims require measured_live evidence.
- This gate run is not a closed PCPR release outcome. Closed release
  decision remains PCPR-094.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-093-operator-live-release-candidate-gate`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
