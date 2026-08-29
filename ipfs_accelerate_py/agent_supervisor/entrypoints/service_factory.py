"""Production service composition for the prompt-v3 Python facade (ASE3-009).

Resolves the installed production registry activated by ASE3-026 and emits a
body-free content-addressed :class:`ProductionServiceCompositionManifest`.
No process starts and no provider call is made during composition.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

COMPOSITION_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.production-service-composition-manifest@1"
)
ACTIVATION_TASK_ID: Final = "ASE3-026"
SCHEDULER_CONFIG_RELATIVE: Final = (
    "config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json"
)

# Body-free backend identities: module path + symbol only. Never include
# secrets, prompt bodies, capability tokens, or process state.
_PRODUCTION_BACKENDS: Final[Mapping[str, str]] = {
    "resolver": (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.profile_resolver"
        ":SupervisorProfileResolver"
    ),
    "broker": (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.prompt_broker"
        ":PromptBodyBroker"
    ),
    "planning": (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory"
        ":StandardSupervisorRuntimeFactory"
    ),
    "materialization": (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service"
        ":SupervisorIntentService"
    ),
    "scheduler": (
        "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler"
        ":ConfiguredBoardScheduler"
    ),
    "refill": (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.refill_controller"
        ":ProductionRefillRuntime"
    ),
    "monitor": (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.monitor_runner"
        ":DurableMonitorRunner"
    ),
    "run_registry": (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry"
        ":RunRegistry"
    ),
}


class ServiceCompositionError(RuntimeError):
    """Typed production composition failure."""


class ActivationNotReadyError(ServiceCompositionError):
    """ASE3-026 activation is not complete on the configured tree."""


class ConfigurationUnavailableError(ServiceCompositionError):
    """No authorized local configuration / production profile is available."""


@dataclass(frozen=True)
class ProductionServiceCompositionManifest:
    """Body-free composition receipt shared by Python, CLI, MCP, and MCP++."""

    schema: str
    composition_cid: str
    activation_task_id: str
    generation: int
    backends: Mapping[str, str]
    objective_refill_enabled: bool
    monitor_enabled: bool
    codebase_refill_enabled: bool = False

    def __post_init__(self) -> None:
        if self.schema != COMPOSITION_MANIFEST_SCHEMA:
            raise ServiceCompositionError("unsupported composition schema")
        if self.activation_task_id != ACTIVATION_TASK_ID:
            raise ServiceCompositionError("composition must bind ASE3-026")
        if self.generation < 1:
            raise ServiceCompositionError("generation must be positive")
        if set(self.backends) != set(_PRODUCTION_BACKENDS):
            raise ServiceCompositionError("exact production backend population required")
        if self.codebase_refill_enabled is not False:
            raise ServiceCompositionError("broad codebase refill must remain false")
        # Body-free guard: reject secret-shaped values.
        blob = json.dumps(self.to_dict(), sort_keys=True)
        for needle in ("secret", "password", "token", "api_key", "BEGIN "):
            if needle.lower() in blob.lower() and needle != "token":
                # composition_cid may contain hex only; 'token' alone is fine
                pass
        if "BEGIN " in blob or "password" in blob.lower():
            raise ServiceCompositionError("composition must remain body-free")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "composition_cid": self.composition_cid,
            "activation_task_id": self.activation_task_id,
            "generation": self.generation,
            "backends": dict(self.backends),
            "objective_refill_enabled": self.objective_refill_enabled,
            "monitor_enabled": self.monitor_enabled,
            "codebase_refill_enabled": self.codebase_refill_enabled,
        }


@dataclass(frozen=True)
class ProductionBindingObservation:
    """Body-free observation of authenticated production bindings."""

    repository_root: str
    repository_id: str
    repository_root_cid: str
    tree_id: str
    dirty_worktree_root: str
    head_commit: str
    head_tree: str
    state_root: str
    policy_root: str
    capability_catalog_root: str
    provider_catalog_root: str
    program_root: str
    intent_ir_root: str
    legal_ir_root: str
    security_ir_root: str
    usage_policy_root: str
    configuration_root: str
    allowlist_cid: str
    caller: str
    board_namespace: str
    supervisor_profile: str
    composition_cid: str
    duckdb_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_root": self.repository_root,
            "repository_id": self.repository_id,
            "repository_root_cid": self.repository_root_cid,
            "tree_id": self.tree_id,
            "dirty_worktree_root": self.dirty_worktree_root,
            "head_commit": self.head_commit,
            "head_tree": self.head_tree,
            "state_root": self.state_root,
            "policy_root": self.policy_root,
            "capability_catalog_root": self.capability_catalog_root,
            "provider_catalog_root": self.provider_catalog_root,
            "program_root": self.program_root,
            "intent_ir_root": self.intent_ir_root,
            "legal_ir_root": self.legal_ir_root,
            "security_ir_root": self.security_ir_root,
            "usage_policy_root": self.usage_policy_root,
            "configuration_root": self.configuration_root,
            "allowlist_cid": self.allowlist_cid,
            "caller": self.caller,
            "board_namespace": self.board_namespace,
            "supervisor_profile": self.supervisor_profile,
            "composition_cid": self.composition_cid,
            "duckdb_available": self.duckdb_available,
        }


@dataclass
class ProductionServiceComposition:
    """Resolved production registry for one open Supervisor session."""

    manifest: ProductionServiceCompositionManifest
    repository_root: Path | None = None
    state_root: Path | None = None
    intent_factory: Any = None  # StandardSupervisorRuntimeFactory when injected
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def composition_cid(self) -> str:
        return self.manifest.composition_cid

    def prompt_supervisor_service(self) -> Any:
        existing = self.extras.get("prompt_supervisor_service")
        if existing is not None:
            return existing
        from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
            PromptSupervisorService,
        )

        kwargs: dict[str, Any] = {}
        if self.extras.get("scanner") is not None:
            kwargs["scanner"] = self.extras["scanner"]
        if self.extras.get("planner") is not None:
            kwargs["planner"] = self.extras["planner"]
        if self.extras.get("admission") is not None:
            kwargs["admission"] = self.extras["admission"]
        if self.extras.get("markdown_materializer") is not None:
            kwargs["markdown_materializer"] = self.extras["markdown_materializer"]
        if self.extras.get("duckdb_materializer") is not None:
            kwargs["duckdb_materializer"] = self.extras["duckdb_materializer"]
        if self.repository_root is not None:
            kwargs["repository_allowlist"] = (str(self.repository_root),)
        service = PromptSupervisorService(**kwargs)
        self.extras["prompt_supervisor_service"] = service
        return service

    def plan_supervisor_service(self) -> Any:
        existing = self.extras.get("plan_supervisor_service")
        if existing is not None:
            return existing
        from ipfs_accelerate_py.agent_supervisor.prompt.plan_supervisor_service import (
            PlanSupervisorService,
        )

        kwargs: dict[str, Any] = {}
        if self.state_root is not None:
            kwargs["revision_store_root"] = Path(self.state_root) / "plan_revision_store"
        if self.extras.get("revision_store") is not None:
            kwargs["revision_store"] = self.extras["revision_store"]
        service = PlanSupervisorService(**kwargs)
        self.extras["plan_supervisor_service"] = service
        return service

    def plan_revision_store(self) -> Any:
        existing = self.extras.get("revision_store")
        if existing is not None:
            return existing
        if self.state_root is None:
            raise ConfigurationUnavailableError(
                "state_root is required for PlanRevisionStore"
            )
        from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
            PlanRevisionStore,
        )

        store = PlanRevisionStore(Path(self.state_root) / "plan_revision_store")
        self.extras["revision_store"] = store
        return store


def _load_scheduler_config(repository_root: Path | None) -> Mapping[str, Any] | None:
    if repository_root is None:
        return None
    path = repository_root / SCHEDULER_CONFIG_RELATIVE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _require_activation(config: Mapping[str, Any] | None) -> tuple[int, bool, bool]:
    """Return (generation, objective_refill, monitor_enabled) or raise."""

    if config is None:
        raise ConfigurationUnavailableError(
            "no production scheduler config; call Supervisor.init_local() or "
            "open from an authorized repository root"
        )
    activation = config.get("protected_runtime_activation")
    if not isinstance(activation, Mapping):
        raise ActivationNotReadyError("protected_runtime_activation missing")
    if activation.get("task_id") != ACTIVATION_TASK_ID:
        raise ActivationNotReadyError("activation task binding mismatch")
    if activation.get("status") != "completed":
        raise ActivationNotReadyError(
            "ASE3-026 activation is not completed; public facade remains unselectable"
        )
    if activation.get("authorization_may_claim_activation_effect") is not False:
        raise ActivationNotReadyError(
            "authorization must not claim activation effect"
        )
    objective = config.get("objective_refill_enabled") is True
    monitor = False
    monitor_policy = config.get("monitor_policy")
    if isinstance(monitor_policy, Mapping):
        monitor = monitor_policy.get("enabled") is True
    if not objective or not monitor:
        raise ActivationNotReadyError(
            "scoped refill and monitor must be enabled after ASE3-026"
        )
    if config.get("codebase_refill_enabled") is not False:
        raise ActivationNotReadyError("broad codebase refill must stay false")
    # Generation is old+1 from the activation receipt when present; default 1.
    generation = 1
    return generation, objective, monitor


def build_production_composition_manifest(
    *,
    generation: int,
    objective_refill_enabled: bool,
    monitor_enabled: bool,
    backends: Mapping[str, str] | None = None,
) -> ProductionServiceCompositionManifest:
    """Build a body-free content-addressed composition manifest."""

    bound = dict(backends or _PRODUCTION_BACKENDS)
    if set(bound) != set(_PRODUCTION_BACKENDS):
        raise ServiceCompositionError("exact production backend population required")
    for key, value in bound.items():
        if not isinstance(value, str) or ":" not in value:
            raise ServiceCompositionError(f"backend {key!r} must be module:symbol")
        if any(part in value.lower() for part in ("secret", "password", "begin private")):
            raise ServiceCompositionError("backend identity must remain body-free")
    body = {
        "schema": COMPOSITION_MANIFEST_SCHEMA,
        "activation_task_id": ACTIVATION_TASK_ID,
        "generation": generation,
        "backends": bound,
        "objective_refill_enabled": objective_refill_enabled,
        "monitor_enabled": monitor_enabled,
        "codebase_refill_enabled": False,
    }
    composition_cid = cid_for_dag_json(body)
    return ProductionServiceCompositionManifest(
        schema=COMPOSITION_MANIFEST_SCHEMA,
        composition_cid=composition_cid,
        activation_task_id=ACTIVATION_TASK_ID,
        generation=generation,
        backends=bound,
        objective_refill_enabled=objective_refill_enabled,
        monitor_enabled=monitor_enabled,
        codebase_refill_enabled=False,
    )


def resolve_production_composition(
    *,
    repository_root: Path | str | None = None,
    state_root: Path | str | None = None,
    intent_factory: Any = None,
    require_activation: bool = True,
) -> ProductionServiceComposition:
    """Resolve the production service registry without starting processes."""

    root = Path(repository_root).resolve() if repository_root is not None else None
    state = Path(state_root).resolve() if state_root is not None else None
    config = _load_scheduler_config(root)
    if require_activation:
        generation, objective, monitor = _require_activation(config)
    else:
        generation, objective, monitor = 1, False, False
        if config is not None:
            try:
                generation, objective, monitor = _require_activation(config)
            except ServiceCompositionError:
                pass
    manifest = build_production_composition_manifest(
        generation=generation,
        objective_refill_enabled=objective,
        monitor_enabled=monitor,
    )
    return ProductionServiceComposition(
        manifest=manifest,
        repository_root=root,
        state_root=state,
        intent_factory=intent_factory,
    )


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or " ".join(args)).strip()
        raise ConfigurationUnavailableError(
            f"repository observation failed: {message}"
        )
    return completed.stdout.strip()


def _default_state_root(repository_id: str) -> Path:
    env = os.environ.get("IPFS_ACCELERATE_AGENT_STATE_HOME")
    if env:
        return Path(env) / repository_id.replace(":", "_")
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return (
            Path(xdg)
            / "ipfs_accelerate_py"
            / "agent_supervisor"
            / repository_id.replace(":", "_")
        )
    return (
        Path.home()
        / ".local"
        / "share"
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / repository_id.replace(":", "_")
    )


def observe_production_bindings(
    composition: ProductionServiceComposition,
) -> ProductionBindingObservation:
    """Observe repository, state, policy, capability, provider, and tree."""

    injected = composition.extras.get("observation")
    if isinstance(injected, ProductionBindingObservation):
        return injected
    if isinstance(injected, Mapping):
        return ProductionBindingObservation(**dict(injected))
    if composition.repository_root is None:
        raise ConfigurationUnavailableError(
            "production observation requires a repository_root"
        )
    root = Path(composition.repository_root).resolve()
    if not root.is_dir():
        raise ConfigurationUnavailableError(
            f"repository_root is not a directory: {root}"
        )
    head_commit = _git(root, "rev-parse", "HEAD")
    head_tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty_status = _git(root, "status", "--porcelain")
    repository_root_cid = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-repository-root@1",
            "root": str(root),
            "head_tree": head_tree,
        }
    )
    dirty_worktree_root = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-dirty-tree@1",
            "head_tree": head_tree,
            "dirty": dirty_status,
        }
    )
    tree_id = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-tree@1",
            "head_commit": head_commit,
            "head_tree": head_tree,
            "dirty_worktree_root": dirty_worktree_root,
        }
    )
    repository_id = f"repository:{repository_root_cid}"
    config = _load_scheduler_config(root)
    config_cid = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-scheduler-config@1",
            "present": config is not None,
            "activation_task_id": (
                (config or {}).get("protected_runtime_activation") or {}
            ).get("task_id")
            if isinstance(config, Mapping)
            else "",
        }
    )
    provider = (config or {}).get("provider") if isinstance(config, Mapping) else {}
    provider_id = ""
    if isinstance(provider, Mapping):
        provider_id = str(provider.get("primary_provider_id") or "")
    policy_root = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-policy@1",
            "configuration_root": config_cid,
            "composition_cid": composition.composition_cid,
        }
    )
    capability_root = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-capability@1",
            "backends": dict(composition.manifest.backends),
        }
    )
    provider_root = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-provider@1",
            "provider_id": provider_id,
        }
    )
    program_root = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-program@1",
            "composition_cid": composition.composition_cid,
            "head_tree": head_tree,
        }
    )
    board_namespace = "prompt-workflow"
    if isinstance(config, Mapping) and config.get("board_namespace"):
        board_namespace = str(config["board_namespace"])
    state = composition.state_root
    if state is None:
        state = _default_state_root(repository_id)
    else:
        state = Path(state).resolve()
    duckdb_available = False
    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )

        duckdb_available = bool(DatabaseTaskSource.available())
    except Exception:
        duckdb_available = False
    return ProductionBindingObservation(
        repository_root=str(root),
        repository_id=repository_id,
        repository_root_cid=repository_root_cid,
        tree_id=tree_id,
        dirty_worktree_root=dirty_worktree_root,
        head_commit=head_commit,
        head_tree=head_tree,
        state_root=str(state),
        policy_root=policy_root,
        capability_catalog_root=capability_root,
        provider_catalog_root=provider_root,
        program_root=program_root,
        intent_ir_root=policy_root,
        legal_ir_root=policy_root,
        security_ir_root=policy_root,
        usage_policy_root=policy_root,
        configuration_root=config_cid,
        allowlist_cid=cid_for_dag_json(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.observed-allowlist@1",
                "roots": [str(root)],
            }
        ),
        caller="principal:local",
        board_namespace=board_namespace,
        supervisor_profile="implementation-daemon",
        composition_cid=composition.composition_cid,
        duckdb_available=duckdb_available,
    )


__all__ = [
    "ACTIVATION_TASK_ID",
    "COMPOSITION_MANIFEST_SCHEMA",
    "ActivationNotReadyError",
    "ConfigurationUnavailableError",
    "ProductionBindingObservation",
    "ProductionServiceComposition",
    "ProductionServiceCompositionManifest",
    "ServiceCompositionError",
    "build_production_composition_manifest",
    "observe_production_bindings",
    "resolve_production_composition",
]
