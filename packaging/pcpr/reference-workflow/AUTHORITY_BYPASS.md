# PCPR-083 Accelerate external-client authority-bypass proof

These files record the Accelerate-owned hermetic negative proof that the
PCPR-080 Python client and the PCPR-081 generic MCP client cannot bypass
platform authority. Both principals are intent, never authority.

- `cpython312/reference.authority-bypass.json` is the declared bypass
  document. Exact commit and tree are bound by the PCPR-083 receipt
  `current_tree_binding`.
- `platform-authority-bypass-catalog.json` observes sibling Datasets and
  Kit bindings when present. Sibling source is never required.
- Every forbidden operation (task update or terminalization, policy
  forgery, confirmation/lease/fence bypass, scope broadening, current-root
  or policy-pointer write, self-promotion) fails typed on both clients.
  No unauthorized effect is observed. Every hard-zero counter remains
  zero. Client CIDs, pack, root, chain, objective, and PCPR-082 parity
  identities are bound and not reminted.
- Live Supervisor.run, live Quack, live MCP stdio/HTTP, and live CLI
  sessions stay typed unavailable. Threat-model preparation remains
  PCPR-090.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-083-operator-live-external-client-authority-bypass`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
