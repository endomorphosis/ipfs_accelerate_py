# PCPR-090 Accelerate objective-to-release threat model

These files record the Accelerate-owned hermetic threat model for the
PCPR objective-to-release path. Actors, assets, trust boundaries, state
owners, canonicalization, CIDs, events, leases, fences, unknown
outcomes, confirmations, proof admission, dependencies, and builds are
bounded.

- `cpython312/reference.threat-model.json` is the declared threat-model
  document. Exact commit and tree are bound by the PCPR-090 receipt
  `current_tree_binding`.
- `platform-threat-model-catalog.json` observes sibling Datasets and Kit
  bindings when present. Sibling source is never required.
- Pack, root, chain, objective, Python-client, MCP-client, parity, and
  PCPR-083 bypass identities are bound and not reminted. The trusted
  computing base remains PCPR-091. The audit package remains PCPR-092.
  Closed release decision remains PCPR-093/094.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-090-operator-live-threat-model`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
