# PCPR-095 Accelerate residual-gap report receipt

These files record the Accelerate-owned hermetic residual-gap report receipt
run for the PCPR objective-to-release path. Supervisor, Datasets, Kit,
Accelerate, packaging, reference-workflow, interoperability, and
hard-safety gates are evaluated as named bounded records.

- `cpython312/reference.residual-gap-report.json` is the declared
  promotion-decision document. Exact commit and tree are bound by the PCPR-095
  receipt `current_tree_binding`.
- `platform-residual-gap-report-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, PCPR-090 threat-model, PCPR-091 TCB-inventory,
  PCPR-092 audit-package, and PCPR-093 gate-run identities are bound
  and not reminted.
- Failed, stale, partial, simulated, estimated, or missing evidence is
  recorded and never hidden. Live claims require measured_live evidence.
- This receipt is an honest non-promotion. It does not claim a closed
  PCPR release outcome. Residual-gap reporting remains PCPR-096.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-095-operator-live-residual-gap-report`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
