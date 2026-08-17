"""Production Python facade: ``Supervisor.open()`` and typed run handles (ASE3-009).

Cold import of ``entrypoints`` does not load this module. Transports must share
the same composition CID. There is no simulated completion path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

from .service_factory import (
    ActivationNotReadyError,
    ConfigurationUnavailableError,
    ProductionServiceComposition,
    ProductionServiceCompositionManifest,
    ServiceCompositionError,
    resolve_production_composition,
)


class SupervisorError(RuntimeError):
    """Base typed facade failure."""


class SupervisorConfigurationError(SupervisorError):
    """Absent or invalid configuration; operator must init or authorize."""


class SupervisorAmbiguityError(SupervisorError):
    """Zero or multiple compatible runs; one typed continuation is required."""

    def __init__(self, message: str, *, candidates: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.candidates = candidates


class SupervisorUnavailableError(SupervisorError):
    """A required production backend is unavailable."""


@dataclass(frozen=True)
class SupervisorObservation:
    """Body-free run observation snapshot."""

    run_id: str
    state: str
    health: str
    event_cursor: str
    composition_cid: str
    summary: str
    values: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "health": self.health,
            "event_cursor": self.event_cursor,
            "composition_cid": self.composition_cid,
            "summary": self.summary,
            "values": dict(self.values),
        }


class SupervisorRun:
    """Typed durable run handle returned by :meth:`Supervisor.run`."""

    def __init__(
        self,
        *,
        run_id: str,
        run_revision: int,
        composition_cid: str,
        state: str,
        health: str,
        event_cursor: str,
        invocation_cid: str = "",
        handle: Any = None,
        supervisor: "Supervisor | None" = None,
        effect_receipt_cids: tuple[str, ...] = (),
    ) -> None:
        if not run_id:
            raise SupervisorError("run_id is required")
        self.run_id = run_id
        self.run_revision = int(run_revision)
        self.composition_cid = composition_cid
        self.state = state
        self.health = health
        self.event_cursor = event_cursor
        self.invocation_cid = invocation_cid
        self._handle = handle
        self._supervisor = supervisor
        self.effect_receipt_cids = effect_receipt_cids

    def steer(self, prompt: str) -> SupervisorObservation:
        if self._supervisor is None:
            raise SupervisorUnavailableError("run is detached from a Supervisor")
        return self._supervisor.steer(self.run_id, prompt)

    def status(self) -> SupervisorObservation:
        if self._supervisor is None:
            raise SupervisorUnavailableError("run is detached from a Supervisor")
        return self._supervisor.status(self.run_id)

    def follow(self) -> Iterator[SupervisorObservation]:
        if self._supervisor is None:
            raise SupervisorUnavailableError("run is detached from a Supervisor")
        return self._supervisor.follow(self.run_id)

    def explain(self) -> SupervisorObservation:
        if self._supervisor is None:
            raise SupervisorUnavailableError("run is detached from a Supervisor")
        return self._supervisor.explain(self.run_id)

    def doctor(self) -> SupervisorObservation:
        if self._supervisor is None:
            raise SupervisorUnavailableError("run is detached from a Supervisor")
        return self._supervisor.doctor(self.run_id)


class Supervisor:
    """Product facade: one open session bound to a production composition."""

    def __init__(
        self,
        composition: ProductionServiceComposition,
        *,
        runs: Mapping[str, SupervisorRun] | None = None,
    ) -> None:
        if not isinstance(composition, ProductionServiceComposition):
            raise SupervisorError("composition must be ProductionServiceComposition")
        self._composition = composition
        self._runs: dict[str, SupervisorRun] = dict(runs or {})

    @property
    def composition_cid(self) -> str:
        return self._composition.composition_cid

    @property
    def composition_manifest(self) -> ProductionServiceCompositionManifest:
        return self._composition.manifest

    @classmethod
    def open(
        cls,
        *,
        repository: Path | str | None = None,
        state_root: Path | str | None = None,
        services: ProductionServiceComposition | None = None,
        intent_factory: Any = None,
        require_activation: bool = True,
    ) -> "Supervisor":
        """Open a production Supervisor.

        After authorized local initialization, no expert constructor arguments
        are required when opened from the configured repository root.
        Injectable ``services`` / ``intent_factory`` are embedder hooks only;
        they cannot invent a simulated completion path.
        """

        if services is not None:
            if not isinstance(services, ProductionServiceComposition):
                raise SupervisorError(
                    "services must be a ProductionServiceComposition"
                )
            composition = services
            if intent_factory is not None:
                composition.intent_factory = intent_factory
        else:
            root = repository
            if root is None:
                # Infer sole enclosing Git root when unique would be ideal;
                # without an authorized root, fail typed rather than guess.
                cwd = Path.cwd()
                candidate = _nearest_git_root(cwd)
                root = candidate
            try:
                composition = resolve_production_composition(
                    repository_root=root,
                    state_root=state_root,
                    intent_factory=intent_factory,
                    require_activation=require_activation,
                )
            except ConfigurationUnavailableError as exc:
                raise SupervisorConfigurationError(str(exc)) from exc
            except ActivationNotReadyError as exc:
                raise SupervisorConfigurationError(str(exc)) from exc
            except ServiceCompositionError as exc:
                raise SupervisorConfigurationError(str(exc)) from exc
        return cls(composition)

    @classmethod
    def init_local(
        cls,
        *,
        repository: Path | str | None = None,
        consent: bool = False,
    ) -> Mapping[str, Any]:
        """One-time local profile bootstrap (explicit consent required).

        Does not start workers. Returns a body-free receipt; callers re-open
        with :meth:`open` after configuration is present.
        """

        if consent is not True:
            raise SupervisorConfigurationError(
                "init_local requires explicit consent=True"
            )
        root = Path(repository).resolve() if repository is not None else Path.cwd()
        git_root = _nearest_git_root(root) or root
        # Prefer the reviewed local_profile initializer when available.
        try:
            from .local_profile import initialize_local_profile

            profile = initialize_local_profile(repository_root=str(git_root))
            receipt = {
                "schema": "ipfs_accelerate_py.agent_supervisor.supervisor-init-local@1",
                "repository_root": str(git_root),
                "initialized": True,
                "profile_bound": True,
            }
            if hasattr(profile, "repository_cid"):
                receipt["repository_cid"] = str(profile.repository_cid)
            return receipt
        except Exception as exc:  # noqa: BLE001 — surface as typed config error
            # If local profile already exists or helper needs more args, report typed.
            raise SupervisorConfigurationError(
                f"local initialization failed: {exc}"
            ) from exc

    def run(self, prompt: str) -> SupervisorRun:
        """Start or resume a durable run from a prompt (intent, never authority)."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise SupervisorError("prompt must be a non-empty string")
        # Prompt bodies never enter durable composition or receipts.
        prompt_cid = cid_for_dag_json(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.prompt-ref@1",
                "length": len(prompt),
                "sha256_prefix": _short_digest(prompt),
            }
        )
        factory = self._composition.intent_factory
        if factory is not None:
            return self._run_via_intent(factory, prompt_cid=prompt_cid)
        # Production path without injectable factory: refuse simulated success.
        # Preview-safe continuation directing the operator to wire lifecycle
        # handlers or use the product CLI once configured.
        raise SupervisorUnavailableError(
            "production intent runtime is not bound for this session; "
            "inject a StandardSupervisorRuntimeFactory with real effect "
            "handlers, or use the product CLI after full launch wiring"
        )

    def _run_via_intent(self, factory: Any, *, prompt_cid: str) -> SupervisorRun:
        from .intent_service import (
            PromptToRunUnavailableError,
            SupervisorIntentService,
        )
        from .runtime_factory import (
            CompleteLaunchPlan,
            StandardSupervisorRuntimeFactory,
            reject_fixture_launch_plan,
        )

        if not isinstance(factory, StandardSupervisorRuntimeFactory):
            raise SupervisorUnavailableError(
                "intent_factory must be a StandardSupervisorRuntimeFactory"
            )
        # Caller must supply a pre-built complete plan via factory extras or
        # we refuse — no fixture plan is synthesized here.
        plan = self._composition.extras.get("complete_plan")
        if plan is None:
            raise SupervisorUnavailableError(
                "no CompleteLaunchPlan bound on composition; refuse simulated run"
            )
        plan = reject_fixture_launch_plan(plan)
        service = SupervisorIntentService(factory=factory)
        try:
            result = service.run(plan)
        except PromptToRunUnavailableError as exc:
            raise SupervisorUnavailableError(str(exc)) from exc
        handle = result.run_handle
        run = SupervisorRun(
            run_id=handle.run_id,
            run_revision=handle.run_revision,
            composition_cid=self.composition_cid,
            state=str(handle.state.value if hasattr(handle.state, "value") else handle.state),
            health=str(
                handle.health.value if hasattr(handle.health, "value") else handle.health
            ),
            event_cursor=handle.event_cursor,
            invocation_cid=result.invocation_cid,
            handle=handle,
            supervisor=self,
            effect_receipt_cids=tuple(result.effect_receipt_cids),
        )
        # Never mark complete without effect receipts on a fresh start.
        if run.state.lower() == "completed" and not run.effect_receipt_cids:
            raise SupervisorUnavailableError(
                "refusing simulated completion without effect receipts"
            )
        self._runs[run.run_id] = run
        # prompt_cid retained only as non-body identity in memory for this session
        self._composition.extras.setdefault("prompt_cids", {})[run.run_id] = prompt_cid
        return run

    def preview(self, prompt: str) -> SupervisorObservation:
        if not isinstance(prompt, str) or not prompt.strip():
            raise SupervisorError("prompt must be a non-empty string")
        prompt_cid = cid_for_dag_json(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.prompt-ref@1",
                "length": len(prompt),
                "sha256_prefix": _short_digest(prompt),
            }
        )
        return SupervisorObservation(
            run_id="",
            state="preview",
            health="unknown",
            event_cursor="",
            composition_cid=self.composition_cid,
            summary="preview-only; no durable effect authorized",
            values={"prompt_cid": prompt_cid, "effect_applied": False},
        )

    def steer(self, run_id: str, prompt: str) -> SupervisorObservation:
        run = self._require_run(run_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise SupervisorError("steer prompt must be a non-empty string")
        return SupervisorObservation(
            run_id=run.run_id,
            state=run.state,
            health=run.health,
            event_cursor=run.event_cursor,
            composition_cid=self.composition_cid,
            summary="steer accepted as intent only; no policy widening",
            values={
                "steering_prompt_cid": cid_for_dag_json(
                    {
                        "schema": "ipfs_accelerate_py.agent_supervisor.steer-ref@1",
                        "length": len(prompt),
                        "sha256_prefix": _short_digest(prompt),
                    }
                ),
                "effect_applied": False,
            },
        )

    def status(self, run_id: str | None = None) -> SupervisorObservation:
        run = self._resolve_run(run_id)
        return SupervisorObservation(
            run_id=run.run_id,
            state=run.state,
            health=run.health,
            event_cursor=run.event_cursor,
            composition_cid=self.composition_cid,
            summary=f"run {run.run_id} state={run.state}",
            values={"run_revision": run.run_revision},
        )

    def follow(self, run_id: str | None = None) -> Iterator[SupervisorObservation]:
        run = self._resolve_run(run_id)
        yield self.status(run.run_id)

    def explain(self, run_id: str | None = None) -> SupervisorObservation:
        run = self._resolve_run(run_id)
        return SupervisorObservation(
            run_id=run.run_id,
            state=run.state,
            health=run.health,
            event_cursor=run.event_cursor,
            composition_cid=self.composition_cid,
            summary="body-free explanation of bound composition and run cursor",
            values={
                "composition": self.composition_manifest.to_dict(),
                "run_revision": run.run_revision,
            },
        )

    def doctor(self, run_id: str | None = None) -> SupervisorObservation:
        run = self._resolve_run(run_id)
        return SupervisorObservation(
            run_id=run.run_id,
            state=run.state,
            health=run.health,
            event_cursor=run.event_cursor,
            composition_cid=self.composition_cid,
            summary="doctor snapshot; detection does not grant restart authority",
            values={
                "composition_cid": self.composition_cid,
                "activation_task_id": self.composition_manifest.activation_task_id,
                "generation": self.composition_manifest.generation,
            },
        )

    def _require_run(self, run_id: str) -> SupervisorRun:
        if run_id not in self._runs:
            raise SupervisorAmbiguityError(
                f"unknown run_id {run_id!r}",
                candidates=tuple(sorted(self._runs)),
            )
        return self._runs[run_id]

    def _resolve_run(self, run_id: str | None) -> SupervisorRun:
        if run_id is not None:
            return self._require_run(run_id)
        if len(self._runs) == 1:
            return next(iter(self._runs.values()))
        if not self._runs:
            raise SupervisorAmbiguityError(
                "no active run; supply run_id or call run(prompt) first",
                candidates=(),
            )
        raise SupervisorAmbiguityError(
            "multiple active runs; supply an exact run_id",
            candidates=tuple(sorted(self._runs)),
        )


def _nearest_git_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _short_digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "Supervisor",
    "SupervisorAmbiguityError",
    "SupervisorConfigurationError",
    "SupervisorError",
    "SupervisorObservation",
    "SupervisorRun",
    "SupervisorUnavailableError",
]
