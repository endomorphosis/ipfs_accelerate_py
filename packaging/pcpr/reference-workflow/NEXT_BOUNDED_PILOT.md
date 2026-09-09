# PCPR-096 Accelerate next-bounded-pilot recommendation

These files record the Accelerate-owned hermetic next-bounded-pilot
recommendation for the PCPR objective-to-release path. The PCPR-095
residual-gap report is bound and not reminted. The recommendation is a
synthetic R&D pilot. A customer pilot is not eligible. The pilot is not
created. No successor campaign is materialized.

- `cpython312/reference.next-bounded-pilot.json` is the declared
  recommendation document. Exact commit and tree are bound by the PCPR-096
  receipt `current_tree_binding`.
- `platform-next-bounded-pilot-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, PCPR-090 threat-model, PCPR-091 TCB-inventory,
  PCPR-092 audit-package, PCPR-093 gate-run, PCPR-094 promotion-decision,
  and PCPR-095 residual-gap-report identities are bound and not reminted.
- Failed, stale, partial, simulated, estimated, or missing evidence is
  recorded and never hidden. Live claims require measured_live evidence.
- This receipt is an honest non-promotion. It does not claim a closed
  PCPR release outcome. It does not create a successor campaign.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-096-operator-live-next-bounded-pilot`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
