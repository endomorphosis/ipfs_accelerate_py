# PCPR-080 Accelerate Python external-client demonstration

These files record the Accelerate-owned hermetic Python client that
uses the same canonical service MCP and CLI would use. The client
submits the PCPR-060 canonical objective, receives the same objective
identity, subscribes to events, inspects task state, submits candidate
evidence, and retrieves the PCPR-072 final proof-carrying receipt
chain. It does not remint those identities and does not claim a live
supervisor session.

- `cpython312/reference.python-external-client.json` is the declared
  client document. Exact commit and tree are bound by the PCPR-080
  receipt `current_tree_binding`.
- `platform-python-external-client-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- The Python principal is intent, never authority. Forbidden client
  operations are refused. Live Supervisor.run, live Quack, live MCP,
  and live CLI sessions stay typed unavailable. Generic MCP-client
  demonstration remains PCPR-081.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-080-operator-live-python-external-client`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
