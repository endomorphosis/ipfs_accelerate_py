# PCPR-082 Accelerate cross-client objective identity parity

These files record the Accelerate-owned hermetic proof that the PCPR-080
Python client and the PCPR-081 generic MCP client submit the same
canonical PCPR-060 objective bytes to the same canonical service and
receive the same objective identity and equivalent goals, tasks, events,
candidate evidence, and final receipts.

- `cpython312/reference.objective-identity-parity.json` is the declared
  parity document. Exact commit and tree are bound by the PCPR-082
  receipt `current_tree_binding`.
- `platform-objective-identity-parity-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Python client CID and generic MCP client CID remain distinct and are
  not reminted. Both principals are intent, never authority. Live
  Supervisor.run, live Quack, live MCP stdio/HTTP, and live CLI sessions
  stay typed unavailable. Authority-bypass proof remains PCPR-083.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-082-operator-live-cross-client-objective-identity-parity`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
