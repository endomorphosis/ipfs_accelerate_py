# Contributor guide: extending the agent supervisor

## Before you write code

1. Read [Design philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md).  
2. Find the owning package in [PACKAGE_MAP.md](PACKAGE_MAP.md).  
3. Prefer **semantic names** (what it does), not board prefixes (which ticket
   funded it).  
4. Keep program boards (`*.todo.md` / `*.objectives.md`) as the execution
   record; do not encode their IDs into public APIs.

## Package rules

### Dependency DAG

Allowed edges flow **up** from `core` toward `todo_daemon` / `integrations`
(see package map). Do not create cycles. Do not import implementation daemons
from `proof` or `core`.

### Adding a module

1. Place the file under the correct domain package.  
2. Update that package’s `README.md` public module table.  
3. Export only if it is an intentional public surface (package or root
   `__init__.py` policy).  
4. Add focused tests under `test/api/` (or package-local tests if that is the
   repo convention).  
5. Avoid thin re-export stubs at retired flat paths.

### Naming

| Do | Don’t |
| --- | --- |
| `code_proof_public_bindings` | `cbp_public_bindings` as the only name |
| `SrtHoldoutRegistry` | `Plat2HoldoutRegistry` as the only name |
| Document “landed under codebase-proof program” in an evidence footer | Name packages `cbp/` or `asi/` |

If you must rename a public symbol, add a short-lived deprecated alias and
update in-repo callers in the same change.

## Control surface stability

- New **operations** require contract updates, discovery manifest consistency,
  and parity expectations across Python/CLI/MCP where applicable.  
- Generation-2 stable export manifests are closed sets; do not casually add
  names.  
- Capability probes are separate from proof receipts.

## Adding a program (board + objectives)

A program is not a new top-level package named after the prefix.

1. Add/keep `docs/architecture/<program>.objectives.md` and `.todo.md`.  
2. Register glossary row in [PROGRAMS.md](PROGRAMS.md).  
3. Put implementation modules in existing domain packages.  
4. Document operator launch (protected paths, provider env) in the program plan
   or the operator guide.  
5. Keep task headers (`## PREFIX-###`) stable for the daemon.

## Validation expectations

- Targeted pytest for the modules you touch.  
- No protected-path rewrites in implementation attempts unless owned.  
- Import paths in docs/examples should use domain packages on `main`.

## Documentation checklist for a non-trivial change

- [ ] Philosophy still accurate (if you changed authority/evidence rules)  
- [ ] Package map / domain README updated  
- [ ] Agent capsule still true (invariants)  
- [ ] No new board-prefix names on public APIs  
- [ ] Guide examples still run  

## Related

- [Doc hub](README.md)  
- [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md)  
- [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md)  
