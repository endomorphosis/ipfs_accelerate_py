# agent_supervisor

Autonomous agent supervisor for objective-driven work: control-plane
operations, formal planning/proof, merge lanes, and implementation daemons.

This package is a **proof- and policy-bounded control plane**. Models propose;
validation, leases, allowlists, and typed evidence admit mutations.

## Documentation (start here)

| Audience | Doc |
| --- | --- |
| Design philosophy | [`docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md`](../../docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md) |
| Doc hub | [`docs/architecture/agent_supervisor/README.md`](../../docs/architecture/agent_supervisor/README.md) |
| Package map | [`docs/architecture/agent_supervisor/PACKAGE_MAP.md`](../../docs/architecture/agent_supervisor/PACKAGE_MAP.md) |
| Domain package READMEs | [`docs/architecture/agent_supervisor/packages/`](../../docs/architecture/agent_supervisor/packages/) |
| Operators | [`docs/guides/AGENT_SUPERVISOR_GUIDE.md`](../../docs/guides/AGENT_SUPERVISOR_GUIDE.md) |
| Agents | [`docs/architecture/agent_supervisor/FOR_AGENTS.md`](../../docs/architecture/agent_supervisor/FOR_AGENTS.md) |
| Contributors | [`docs/architecture/agent_supervisor/FOR_CONTRIBUTORS.md`](../../docs/architecture/agent_supervisor/FOR_CONTRIBUTORS.md) |
| Architecture | [`docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md`](../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) |
| Program glossary | [`docs/architecture/agent_supervisor/PROGRAMS.md`](../../docs/architecture/agent_supervisor/PROGRAMS.md) |

## Domain packages (target layout on `main`)

```text
core → control, task_sources, context, analysis, proof
     → objectives, planning, validation, prompt
     → merge, rescue, runtime, self_improvement
     → todo_daemon, integrations
```

Import from domain packages when present:

```python
from ipfs_accelerate_py.agent_supervisor import (
    Operation,
    OperationRequest,
    SupervisorControlService,
)
```

Prefer domain paths for new code:

```python
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)
```

## Public API policy

1. Root `__init__.py` re-exports intentional symbols only (control surface,
   stable manifests, selected helpers).
2. Optional providers stay **lazy**; cold import must not start processes.
3. Discovery ≠ capability ≠ proof (see philosophy).
4. Do **not** encode taskboard prefixes into public API names.

## Programs vs packages

Self-improvement, codebase-proof, domain layout, catalog work, and similar
efforts are **programs** (boards + objectives). They layer on this package;
they are not alternate package trees named after board prefixes.

## Status note

Feature branches may still carry a transitional flat module tree while `main`
uses domain packages. Treat the package map and `packages/*.md` docs as the
canonical ownership model for both.
