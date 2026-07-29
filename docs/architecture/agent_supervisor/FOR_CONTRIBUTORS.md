# Contributor guide: extending the agent supervisor

Checklist and conventions for humans (and agents) landing non-trivial changes.

## Before you write code

1. Read [Design philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) (authority & evidence).
2. Read [Developer guide](DEVELOPER_GUIDE.md) (imports, placement, pitfalls).
3. Find the owning package in [PACKAGE_MAP.md](PACKAGE_MAP.md).
4. Prefer **semantic names** (what it does), not board prefixes (which ticket funded it).
5. Keep program boards (`*.todo.md` / `*.objectives.md`) as the execution record;
   do not encode their IDs into public APIs.

## Package rules

### Dependency DAG

Allowed edges flow **up** from `core` toward `todo_daemon` / `integrations`
(see package map). Do not create cycles. Do not import implementation daemons
from `proof` or `core`.

### Adding a module

1. Place the file under the correct domain package.
2. Update that package’s `README.md` (module table + purpose if the role changes).
3. Update `docs/architecture/agent_supervisor/packages/<pkg>.md` if the semantic
   page still describes the old surface.
4. Export only if it is an intentional public surface (package or root
   `__init__.py` policy — **closed** for generation-2 manifests).
5. Add focused tests under `test/api/` (or package-local tests if that is the
   repo convention).
6. Avoid thin re-export stubs at retired flat paths.

### Naming

| Do | Don’t |
| --- | --- |
| `code_proof_public_bindings` | `cbp_public_bindings` as the only name |
| `SrtHoldoutRegistry` | `Plat2HoldoutRegistry` as the only name |
| `AGENT_SUPERVISOR_CORE_PACKAGES` | New public `AGENT_SUPERVISOR_G0xx_*` names |
| Document “landed under X program” in an evidence footer | Name packages `cbp/` or `asi/` |

If you must rename a public symbol, add a short-lived deprecated alias and
update in-repo callers in the same change.

## Control surface stability

- New **operations** require contract updates, discovery manifest consistency,
  and parity expectations across Python/CLI/MCP where applicable.
- Generation-2 stable export manifests (`AGENT_SUPERVISOR_PUBLIC_API_EXPORTS`)
  are closed sets; do not casually add names.
- Capability probes are separate from proof receipts and from import success.
- Cold import must remain free of process/network/provider side effects.

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
- If you touch layout inventories or public exports, run:

```bash
python -m pytest \
  test/api/test_agent_supervisor_asref_layout_evidence.py \
  test/api/test_agent_supervisor_semantic_layout_exports.py -q
```

## Documentation checklist for a non-trivial change

- [ ] [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) still accurate (if authority/evidence rules changed)
- [ ] [Developer guide](DEVELOPER_GUIDE.md) / [Package map](PACKAGE_MAP.md) updated if placement rules changed
- [ ] Domain package `README.md` module table updated
- [ ] Semantic package page under `packages/` updated if needed
- [ ] [Agent capsule](FOR_AGENTS.md) still true (invariants)
- [ ] No new board-prefix names on public APIs
- [ ] Operator [guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) examples still valid for control-surface changes
- [ ] Architecture doc cross-links still point at domain paths

## Related

- [Doc hub](README.md)
- [Developer guide](DEVELOPER_GUIDE.md)
- [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md)
- [Package code README](../../../ipfs_accelerate_py/agent_supervisor/README.md)
