# PCPR-081 Accelerate generic MCP-client demonstration

These files record the Accelerate-owned hermetic generic MCP client that
uses the same canonical service the Python client and CLI would use.
The client speaks JSON-RPC 2.0 MCP (`initialize`, `tools/list`,
`tools/call`) and executes the complete high-level objective path: it
submits the PCPR-060 canonical objective, receives the same objective
identity, subscribes to events, inspects task state, submits candidate
evidence, and retrieves the PCPR-072 final proof-carrying receipt
chain. It does not remint those identities and does not claim a live
MCP stdio or HTTP session.

- `cpython312/reference.generic-mcp-client.json` is the declared client
  document. Exact commit and tree are bound by the PCPR-081 receipt
  `current_tree_binding`.
- `platform-generic-mcp-client-catalog.json` observes sibling Datasets
  and Kit bindings when present. Sibling source is never required.
- The MCP principal is intent, never authority. Forbidden client
  operations are refused as JSON-RPC errors. Live Supervisor.run, live
  Quack, live MCP stdio/HTTP, and live CLI sessions stay typed
  unavailable. This is not MCP++. Cross-client identity parity remains
  PCPR-082.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-081-operator-live-generic-mcp-client`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
