# PCPR-092 Accelerate security and correctness audit package

These files record the Accelerate-owned hermetic security and
correctness audit package for the PCPR objective-to-release path.
Architecture, state-machine, authority, canonicalization, recovery,
proof, build, negative-test, and limitation evidence are assembled as
named reproducible artifacts.

- `cpython312/reference.security-and-correctness-audit-package.json` is
  the declared audit-package document. Exact commit and tree are bound
  by the PCPR-092 receipt `current_tree_binding`.
- `platform-security-and-correctness-audit-package-catalog.json`
  observes sibling Datasets and Kit bindings when present. Sibling
  source is never required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, PCPR-090 threat-model, and PCPR-091 TCB-inventory
  identities are bound and not reminted. Closed release decision
  remains PCPR-093/094.
- The package status is `external_audit_ready` and is never
  `externally_audited`.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-092-operator-live-audit-package`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
