"""Production service composition for the prompt-v3 Python facade (ASE3-009).

Resolves the installed production registry activated by ASE3-026 and emits a
body-free content-addressed :class:`ProductionServiceCompositionManifest`.
No process starts and no provider call is made during composition.
"""

from __future__ import annotations

import json
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


__all__ = [
    "ACTIVATION_TASK_ID",
    "COMPOSITION_MANIFEST_SCHEMA",
    "ActivationNotReadyError",
    "ConfigurationUnavailableError",
    "ProductionServiceComposition",
    "ProductionServiceCompositionManifest",
    "ServiceCompositionError",
    "build_production_composition_manifest",
    "resolve_production_composition",
]
