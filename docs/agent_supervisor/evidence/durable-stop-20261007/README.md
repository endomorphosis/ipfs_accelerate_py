# Durable STOP qualification, 2026-10-07

The managed Popen owner retains cleanup-directory custody before launch,
preserves exactly bound cleanup processes, and independently observes completion
before reporting STOP success. The separate lifecycle owner persists refusal
when container cleanup requires an admitted owner it does not yet implement.

| Population | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Cleanup observer, managed STOP, native dispatch guards | 70 | 0 | 0 |
| Lifecycle, signed native START witness, startup repair | 76 | 0 | 0 |
| Current multi-supervisor health and startup | 76 | 0 | 0 |
| Total | 222 | 0 | 0 |

`qualification.json` records exact commands, environment overrides, frozen source
hashes, dependency revision and log/XML/catalog hashes. Each population used a
separate local AST-seal catalog and fixtures directory. Raw logs, XML, DuckDB
catalogs and fixtures remain at the recorded local artifact paths. The recorder
is included for reproducibility; adjust its checkout/output constants and use a
fresh label before rerunning.

The tests include real private bindings, canonical resource retirement, signed
terminal CAS joins, persisted lifecycle journals and local process recovery.
Docker observations are explicit doubles. This is not live Docker qualification,
cold-owner recovery, protected provider execution or a Terminal Bench result.
Native factory and CLI entrypoints remain unconditionally disabled.

Baseline classifications are separate from the passing populations:

- `ordinary-shutdown-baseline.json`: 18 older shutdown failures reproduce on the
  unmodified starting commit; those contracts still need migration.
- `generation-status-baseline.json`: the old generation-status suite fails
  collection on the same commit because an imported contract is missing.
- `status-retry-baseline.json`: a stale two-attempt health assertion predates the
  intentional four-attempt production contract. Only the test was corrected;
  the complete health suite then passed.

`owner-provenance.json` records the narrowly reused protocol and donor review.
`stop-path-and-image-audit.md` contains the independent review and the remaining
owner recovery, signed launch, callback, and approved-image prerequisites.

The parent repository concurrently advanced its datasets pin.
`datasets-pin-audit.json` records the preserved new pin and the read-only review
that found no STOP runtime dependency changes. Test results remain explicitly
bound to the older datasets revision recorded in the qualification.
