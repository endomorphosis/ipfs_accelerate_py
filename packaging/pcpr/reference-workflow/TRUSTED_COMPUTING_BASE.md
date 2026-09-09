# PCPR-091 Accelerate objective-to-release trusted computing base

These files record the Accelerate-owned hermetic trusted-computing-base
inventory for the PCPR objective-to-release path. Actors, assets, trust
boundaries, state owners, canonicalization, CIDs, events, leases, fences,
unknown outcomes, confirmations, proof admission, dependencies, and
builds are bounded as named TCB components.

- `cpython312/reference.trusted-computing-base.json` is the declared TCB
  inventory document. Exact commit and tree are bound by the PCPR-091
  receipt `current_tree_binding`.
- `platform-trusted-computing-base-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, and PCPR-090 threat-model identities are bound and
  not reminted. The audit package remains PCPR-092. Closed release
  decision remains PCPR-093/094.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-091-operator-live-tcb-inventory`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
