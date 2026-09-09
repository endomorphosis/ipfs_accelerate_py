# PCPR-094 Accelerate promotion or non-promotion receipt

These files record the Accelerate-owned hermetic promotion or non-promotion receipt
run for the PCPR objective-to-release path. Supervisor, Datasets, Kit,
Accelerate, packaging, reference-workflow, interoperability, and
hard-safety gates are evaluated as named bounded records.

- `cpython312/reference.promotion-or-non-promotion.json` is the declared
  promotion-decision document. Exact commit and tree are bound by the PCPR-094
  receipt `current_tree_binding`.
- `platform-promotion-or-non-promotion-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, PCPR-090 threat-model, PCPR-091 TCB-inventory,
  PCPR-092 audit-package, and PCPR-093 gate-run identities are bound
  and not reminted.
- Failed, stale, partial, simulated, estimated, or missing evidence is
  recorded and never hidden. Live claims require measured_live evidence.
- This receipt is an honest non-promotion. It does not claim a closed
  PCPR release outcome. Residual-gap reporting remains PCPR-095.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-094-operator-live-promotion`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
