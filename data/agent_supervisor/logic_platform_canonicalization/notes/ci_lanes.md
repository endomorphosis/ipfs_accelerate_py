# LPC-151 Required CI lanes (fail on failure)

Required blocking lanes (no `continue-on-error`, no `|| true`, no skipped required tests):

| Lane | Purpose |
| --- | --- |
| contracts | pure contract imports, no solver/network |
| datasets-unit | `ipfs_datasets_py` logic unit tests |
| parser | syntax_core / parser conformance |
| domain-slice | DomainLogicSlice@2 adapters including legal, security, intent, UI/UX |
| provider | provider protocol and BackendRequest@2 |
| tactician | proof-plan / advisor-authority tests |
| receipts | receipt adversarial tests |
| adapter | supervisor canonical adapter + client |
| manifest | cross-package LogicPlatformManifest handshake |
| parity | direct vs supervisor + Python/CLI/MCP |
| wheel-install | clean wheel install without sibling layout |
| doc-drift | generated documentation drift |
| catalog-drift | generated catalog snapshot drift |

Unavailable providers must be reported as unavailable, not passed.
Historical reports are not current evidence.
