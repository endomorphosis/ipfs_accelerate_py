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


# Create-plan reasons emitted when no owner IR materials and no fresh host
# capacity snapshot were supplied. They do not admit the parallel plan.
_UNBACKED_CREATE_PLAN_REASONS = frozenset(
    {
        "ir_admission_materials_absent",
        "execution_plan_not_admitted",
        "parallel_plan_rejected",
        "parallel:resource_infeasible",
        "parallel:stale_capacity",
        "parallel:invalid_snapshot",
    }
)


def _plan_create_ir_absence_only(envelope: Mapping[str, Any]) -> bool:
    """True when create-plan is review-only only for missing IR/capacity backing.

    Critique, structural, and other parallel failures still block the run.
    This does not mark the create receipt admitted and does not grant execution.
    """

    if str(envelope.get("status") or "") != "review_only":
        return False
    reasons = {
        str(item)
        for item in (envelope.get("rejection_reasons") or ())
        if str(item)
    }
    return (
        "ir_admission_materials_absent" in reasons
        and reasons <= _UNBACKED_CREATE_PLAN_REASONS
    )


def _scan_directory_for_repository(repository_root: str) -> str:
    """Return an existing in-repo scan directory.

    Tests and fixtures may provide ``pkg/``. Production checkouts usually
    do not. Scanning the repository root is valid; a missing ``pkg/`` must
    not fail closed as ``DirectoryResolutionError``.
    """

    root = Path(repository_root)
    packaged = root / "pkg"
    if packaged.is_dir():
        return str(packaged)
    return str(root)


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
        identities: Mapping[str, Any] | None = None,
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
        self.identities = dict(identities or {})

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
        profile_dir: Path | None = None,
        lifecycle_dir: Path | None = None,
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
            from .service_factory import observe_production_bindings

            # Observe real Git bindings before creating any local authority.
            # Bootstrap does not require an activated scheduler and does not
            # grant activation, merge, or worker-start authority.
            composition = resolve_production_composition(
                repository_root=git_root, require_activation=False
            )
            bindings = observe_production_bindings(composition)
            profile = initialize_local_profile(
                repository_cid=bindings.repository_root_cid,
                baseline_commit=bindings.head_commit,
                profile_dir=profile_dir,
                lifecycle_dir=lifecycle_dir,
            )
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

    def observe_bindings(self) -> Mapping[str, Any]:
        """Observe authenticated repository, state, policy, and tree bindings."""

        from .service_factory import (
            ConfigurationUnavailableError,
            observe_production_bindings,
        )

        try:
            return observe_production_bindings(self._composition).to_dict()
        except ConfigurationUnavailableError as exc:
            raise SupervisorConfigurationError(str(exc)) from exc

    def run(self, prompt: str) -> SupervisorRun:
        """Admit a prompt and start workers for its ready tasks.

        Prompt text is intent, never authority.  The supervisor process
        lives outside the repository.  Each worker runs the admitted task
        in the repository and does not grant execution or completion
        authority.  Fenced lifecycle START remains a separate control
        operation.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise SupervisorError("prompt must be a non-empty string")
        factory = self._composition.intent_factory
        if factory is not None and self._composition.extras.get("complete_plan") is not None:
            prompt_cid = cid_for_dag_json(
                {
                    "schema": "ipfs_accelerate_py.agent_supervisor.prompt-ref@1",
                    "length": len(prompt),
                    "sha256_prefix": _short_digest(prompt),
                }
            )
            return self._run_via_intent(factory, prompt_cid=prompt_cid)
        return self._run_from_prompt(prompt)

    def start(self, run_id: str) -> SupervisorObservation:
        """Authorized START of a previously materialized run."""

        run = self._require_run(run_id)
        factory = self._composition.intent_factory
        if factory is None:
            raise SupervisorUnavailableError(
                "configured-board runtime is not bound for START"
            )
        identities = dict(run.identities)
        task_source_cid = str(identities.get("task_source_cid") or "")
        task_source_revision_cid = str(
            identities.get("task_source_revision_cid") or ""
        )
        if not task_source_cid or not task_source_revision_cid:
            raise SupervisorUnavailableError(
                "START requires materialized task-source identities"
            )
        lease_id = str(identities.get("lease_id") or "")
        fencing = identities.get("fencing_epoch")
        if not lease_id or fencing is None:
            raise SupervisorUnavailableError(
                "START requires lease and fence bindings"
            )
        from types import SimpleNamespace

        from .runtime_factory import (
            RuntimeEffectError,
            complete_launch_plan_from_materialization,
            launch_plan_from_observation,
        )

        observation = dict(identities.get("observation") or self.observe_bindings())
        try:
            launch = launch_plan_from_observation(
                observation,
                invocation_cid=str(
                    identities.get("workflow_request_cid") or run.invocation_cid
                ),
                idempotency_key=str(
                    identities.get("start_idempotency_key")
                    or ("start:" + run.run_id)
                ),
            )
            complete = complete_launch_plan_from_materialization(
                launch_plan=launch,
                task_source_cid=task_source_cid,
                task_source_revision_cid=task_source_revision_cid,
                objective_cid=str(identities.get("objective_cid") or ""),
                objective_revision_cid=str(
                    identities.get("workflow_request_cid") or ""
                ),
            )
            handle = SimpleNamespace(
                run_id=run.run_id,
                run_revision=run.run_revision,
                state=run.state,
                health=run.health,
                event_cursor=run.event_cursor,
                task_source_cid=task_source_cid,
                task_source_revision_cid=task_source_revision_cid,
            )
            started = factory.invoke("start", complete, handle)
        except Exception as exc:
            raise SupervisorUnavailableError(str(exc)) from exc
        process_cid = str(started.values.get("process_cid") or "")
        started_lease = str(started.values.get("lease_id") or "")
        fencing = int(started.values.get("fencing_generation") or 0)
        if not process_cid or not started_lease or fencing < 1:
            raise SupervisorUnavailableError(
                "start receipt lacks process identity or fenced lease"
            )
        if not started.effect_applied:
            raise RuntimeEffectError("START handler did not apply its declared effect")
        run.state = "running"
        run.health = "healthy"
        run.event_cursor = str(
            started.values.get("event_cursor") or "lifecycle-started"
        )
        run.effect_receipt_cids = tuple(
            dict.fromkeys((*run.effect_receipt_cids, started.receipt_cid))
        )
        run.identities["start_receipt_cid"] = started.receipt_cid
        run.identities["process_cid"] = process_cid
        run.identities["start_event_cursor"] = run.event_cursor
        run.identities["start_lease_id"] = started_lease
        run.identities["start_fencing_generation"] = fencing
        if run.state.lower() == "completed" and not run.effect_receipt_cids:
            raise SupervisorUnavailableError(
                "refusing simulated completion without effect receipts"
            )
        return SupervisorObservation(
            run_id=run.run_id,
            state=run.state,
            health=run.health,
            event_cursor=run.event_cursor,
            composition_cid=self.composition_cid,
            summary=f"start authorized run_id={run.run_id}",
            values={
                "effect_applied": True,
                "identities": dict(run.identities),
                "effect_receipt_cids": list(run.effect_receipt_cids),
            },
        )

    def _run_from_prompt(
        self, prompt: str, *, start_workers: bool = True
    ) -> SupervisorRun:
        from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
            plan_create_request_from_workflow,
        )
        from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
            PromptWorkflowRequest,
            RecordStatus,
            construct_prompt_workflow_request,
        )
        from .service_factory import (
            ConfigurationUnavailableError,
            observe_production_bindings,
        )

        try:
            observation = observe_production_bindings(self._composition)
        except ConfigurationUnavailableError as exc:
            raise SupervisorConfigurationError(str(exc)) from exc
        extras = self._composition.extras
        mutation = extras.get("mutation_bindings")
        if mutation is not None and not isinstance(mutation, Mapping):
            raise SupervisorError("mutation_bindings must be a mapping")
        mutation_map = dict(mutation or {})
        materialize = bool(mutation_map)
        if materialize:
            required = (
                "authority_cid",
                "idempotency_key",
                "lease_id",
                "fencing_epoch",
            )
            missing = [key for key in required if mutation_map.get(key) in (None, "")]
            if missing:
                raise SupervisorUnavailableError(
                    "mutation requires authority, idempotency, lease, and fence"
                )
        output_root = str(self._composition.state_root or observation.state_root)
        if not output_root:
            raise SupervisorConfigurationError(
                "state_root is required to construct workflow outputs"
            )
        if self._composition.state_root is None:
            from pathlib import Path as _Path

            self._composition.state_root = _Path(observation.state_root)
        prepared = extras.get("workflow_request")
        if prepared is not None:
            if not isinstance(prepared, PromptWorkflowRequest):
                raise SupervisorError(
                    "workflow_request must be PromptWorkflowRequest"
                )
            body = prepared.prompt_source.transient_body
            if body is None or body.decode("utf-8") != prompt:
                raise SupervisorError(
                    "workflow_request is not bound to this prompt"
                )
            workflow_request = prepared
        else:
            # The workflow request is a prompt/scanner contract: its allowlist
            # and repository-root identities are the scanner's canonical
            # identities, which the scanner and the preview's scan validation
            # recompute and compare. Observation digests stay in the
            # binding/status domain.
            from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
                bind_prompt_request_to_scanner,
                repository_allowlist_cid,
                repository_root_cid,
            )

            workflow_request = construct_prompt_workflow_request(
                prompt,
                repository_root=observation.repository_root,
                repository_root_cid=repository_root_cid(observation.repository_root),
                allowlist_cid=repository_allowlist_cid(
                    (observation.repository_root,)
                ),
                program_root=observation.program_root,
                intent_ir_root=observation.intent_ir_root,
                legal_ir_root=observation.legal_ir_root,
                security_ir_root=observation.security_ir_root,
                policy_root=observation.policy_root,
                caller=observation.caller,
                output_root=output_root,
                directory=_scan_directory_for_repository(observation.repository_root),
                state_root=str(self._composition.state_root),
                supervisor_profile=observation.supervisor_profile,
                board_namespace=observation.board_namespace,
                dry_run=not materialize,
                materialize=materialize,
                start_after_materialize=False,
                authority_cid=str(mutation_map.get("authority_cid") or ""),
                idempotency_key=str(mutation_map.get("idempotency_key") or ""),
                lease_id=str(mutation_map.get("lease_id") or ""),
                fencing_epoch=(
                    int(mutation_map["fencing_epoch"])
                    if mutation_map.get("fencing_epoch") is not None
                    else None
                ),
                duckdb_available=bool(observation.duckdb_available),
            )
            if extras.get("scanner") is None:
                workflow_request = bind_prompt_request_to_scanner(
                    workflow_request,
                    (observation.repository_root,),
                )
        plan_request = plan_create_request_from_workflow(
            workflow_request,
            repository_id=observation.repository_id,
            board_namespace=observation.board_namespace,
            capability_catalog_root=observation.capability_catalog_root,
            provider_catalog_root=observation.provider_catalog_root,
            usage_policy_root=observation.usage_policy_root,
            configuration_root=observation.configuration_root,
            dirty_worktree_root=observation.dirty_worktree_root,
            task_source_id="task-source:database:production",
            task_source_revision=observation.tree_id,
        )
        # The model router is an isolated child with a multi-minute timeout.
        # A prompt must start workers before that child runs. An injected
        # planner is left untouched.
        if (
            extras.get("planner") is None
            and extras.get("prompt_supervisor_service") is None
        ):
            extras["planner"] = _DeterministicPromptPlanner()
        prompt_service = self._composition.prompt_supervisor_service()
        preview = prompt_service.preview(workflow_request)
        if preview.status is not RecordStatus.ADMITTED:
            raise SupervisorUnavailableError(
                "prompt workflow preview was not admitted: "
                + ",".join(preview.rejection_reasons or ("rejected",))
            )
        plan_service = self._composition.plan_supervisor_service()
        plan_envelope = plan_service.preview_create_admitted(plan_request)
        plan_status = str(plan_envelope.get("status") or "")
        if plan_status != "admitted" and not _plan_create_ir_absence_only(
            plan_envelope
        ):
            reasons = ",".join(
                str(item)
                for item in (plan_envelope.get("rejection_reasons") or ())
                if str(item)
            )
            raise SupervisorUnavailableError(
                "plan create preview was not admitted"
                + (f": {reasons}" if reasons else "")
            )
        preview_state = prompt_service._preview_state(preview)
        hard_receipt = getattr(preview_state.admission, "receipt", None)
        preview_graph = preview_state.graph
        root_provenance = dict(getattr(preview_graph.root_goal, "provenance", {}) or {})
        projection_ids = root_provenance.get("projection_ids") or ()
        materialization_inputs = dict(
            prompt_service.materialization_inputs(preview)
        )
        materialization_inputs["plan_create_request_cid"] = plan_request.request_cid
        materialization_inputs.update(
            dict(plan_envelope.get("materialization_inputs") or {})
        )
        identities: dict[str, Any] = {
            "objective_cid": workflow_request.prompt_cid,
            "prompt_cid": workflow_request.prompt_cid,
            "workflow_request_cid": workflow_request.request_cid,
            "plan_create_request_cid": plan_request.request_cid,
            "preview_receipt_cid": preview.receipt_cid,
            "admission_receipt_cid": preview.admission_receipt_cid,
            "plan_root_cid": preview.plan_root_cid,
            "scan_cid": preview.scan_cid,
            "plan_preview_receipt_cid": str(
                plan_envelope.get("receipt_cid") or ""
            ),
            "materialization_inputs": materialization_inputs,
            "observation": observation.to_dict(),
            "admitted_task_count": len(preview.admitted_task_cids),
            "admitted_goal_count": len(preview.admitted_goal_cids),
            "intent_autoencoder_status": str(
                root_provenance.get("autoencoder_status") or ""
            ),
            "intent_projection_ids": (
                [str(item) for item in projection_ids]
                if isinstance(projection_ids, (list, tuple))
                else []
            ),
            "intent_subgoal_count": sum(
                1 for goal in preview_graph.goals if goal.parent_goal_cid
            ),
            "hard_domain_admitted": bool(
                getattr(preview_state.admission, "admitted", False)
            ),
            "authorizes_execution": bool(
                getattr(hard_receipt, "authorizes_execution", False)
            ),
            "plan_create_status": plan_status,
            "completion_authority": False,
            "model_assertion_cannot_complete": True,
            "empty_queue_cannot_complete": True,
        }
        effect_receipts: list[str] = [preview.receipt_cid]
        state = "admitted"
        event_cursor = preview.scan_cid
        if materialize:
            apply_receipt = self._materialize_admitted(
                prompt_service=prompt_service,
                plan_service=plan_service,
                preview=preview,
                plan_request=plan_request,
                observation=observation,
                mutation_map=mutation_map,
                materialization_inputs=materialization_inputs,
            )
            apply_payload = (
                apply_receipt.to_dict()
                if hasattr(apply_receipt, "to_dict")
                else dict(apply_receipt)
            )
            identities["plan_revision_cid"] = str(
                apply_payload.get("revision_cid") or ""
            )
            identities["plan_revision_receipt_cid"] = str(
                apply_payload.get("receipt_cid") or ""
            )
            identities["task_source_cid"] = str(
                apply_payload.get("duckdb_projection_cid")
                or apply_payload.get("markdown_projection_cid")
                or ""
            )
            identities["task_source_revision_cid"] = str(
                apply_payload.get("revision_cid")
                or apply_payload.get("plan_root_cid")
                or ""
            )
            identities["markdown_projection_cid"] = str(
                apply_payload.get("markdown_projection_cid") or ""
            )
            identities["duckdb_projection_cid"] = str(
                apply_payload.get("duckdb_projection_cid") or ""
            )
            identities["lease_id"] = str(mutation_map.get("lease_id") or "")
            identities["fencing_epoch"] = int(mutation_map["fencing_epoch"])
            identities["idempotency_key"] = str(
                mutation_map.get("idempotency_key") or ""
            )
            identities["event_cursor"] = str(
                apply_payload.get("event_cursor") or event_cursor
            )
            identities["expected_effects"] = list(
                apply_payload.get("expected_effects")
                or materialization_inputs.get("expected_effects")
                or ()
            )
            identities["observed_effects"] = list(
                apply_payload.get("observed_effects") or ()
            )
            if not identities["task_source_cid"] or not identities[
                "task_source_revision_cid"
            ]:
                raise SupervisorUnavailableError(
                    "PlanRevisionStore did not publish projection identities"
                )
            effect_receipts.append(str(apply_payload.get("receipt_cid") or ""))
            state = "materialized"
            event_cursor = identities["event_cursor"]
        run = SupervisorRun(
            run_id=workflow_request.request_cid,
            run_revision=1,
            composition_cid=self.composition_cid,
            state=state,
            health="unknown",
            event_cursor=event_cursor,
            invocation_cid=workflow_request.request_cid,
            supervisor=self,
            effect_receipt_cids=tuple(item for item in effect_receipts if item),
            identities=identities,
        )
        if run.state.lower() in {"completed", "complete"}:
            raise SupervisorUnavailableError(
                "refusing simulated completion from prompt admission"
            )
        if start_workers:
            prompt_body = workflow_request.prompt_source.transient_body
            prompt_text = (
                prompt_body.decode("utf-8", errors="replace")
                if isinstance(prompt_body, bytes)
                else ""
            )
            self._start_prompt_workers(
                run=run,
                prompt_service=prompt_service,
                preview=preview,
                state_root=Path(str(self._composition.state_root)),
                repository_root=Path(observation.repository_root),
                width=extras.get("prompt_worker_width", 4),
                prompt_text=prompt_text,
            )
        self._runs[run.run_id] = run
        extras_prompts = self._composition.extras.setdefault("prompt_cids", {})
        extras_prompts[run.run_id] = workflow_request.prompt_cid
        return run

    def _start_prompt_workers(
        self,
        *,
        run: SupervisorRun,
        prompt_service: Any,
        preview: Any,
        state_root: Path,
        repository_root: Path,
        width: Any,
        prompt_text: str = "",
    ) -> None:
        """Start live workers for ready tasks without granting authority."""

        from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
            PromptWorkerError,
            launch_prompt_workers,
        )

        graph = prompt_service._preview_state(preview).graph
        try:
            launch = launch_prompt_workers(
                tasks=tuple(graph.tasks),
                admitted_task_cids=tuple(preview.admitted_task_cids),
                run_id=run.run_id,
                state_root=state_root,
                repository_root=repository_root,
                width=width,
                prompt_text=prompt_text,
            )
        except PromptWorkerError as exc:
            raise SupervisorUnavailableError(str(exc)) from exc
        run.state = "running"
        run.health = "healthy"
        run.event_cursor = launch.run_dir
        run.identities["workers_started"] = True
        run.identities["supervisor_pid"] = launch.supervisor_pid
        run.identities["worker_pids"] = list(launch.pids)
        run.identities["active_worker_count"] = launch.active_worker_count
        run.identities["worker_task_cids"] = list(launch.task_cids)
        run.identities["worker_run_dir"] = launch.run_dir
        run.identities["worker_repository_edits"] = False
        run.identities["authorizes_execution"] = bool(
            run.identities.get("authorizes_execution", False)
        )
        run.identities["completion_authority"] = False
        if run.identities["authorizes_execution"]:
            from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
                stop_prompt_workers,
            )

            stop_prompt_workers(
                launch.run_dir,
                (launch.supervisor_pid, *launch.pids),
            )
            raise SupervisorUnavailableError(
                "prompt workers refuse an execution grant"
            )
        if not launch.pids:
            raise SupervisorUnavailableError("prompt workers did not start")

    def _materialize_admitted(
        self,
        *,
        prompt_service: Any,
        plan_service: Any,
        preview: Any,
        plan_request: Any,
        observation: Any,
        mutation_map: Mapping[str, Any],
        materialization_inputs: Mapping[str, Any],
    ) -> Any:
        from pathlib import Path

        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
            MarkdownTaskSource,
        )

        state = prompt_service._preview_state(preview)
        output_root = Path(state.request.output_policy.output_root)
        markdown_source = None
        duckdb_source = None
        markdown_path = state.request.output_policy.markdown_path
        duckdb_path = state.request.output_policy.duckdb_path
        if markdown_path:
            from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_store import (
                MAX_TASKBOARD_MATERIALIZATION_ENTRIES,
            )

            absolute = output_root / markdown_path
            absolute.parent.mkdir(parents=True, exist_ok=True)
            admitted_count = len(tuple(preview.admitted_task_cids))
            if admitted_count <= MAX_TASKBOARD_MATERIALIZATION_ENTRIES:
                markdown_source = MarkdownTaskSource(
                    absolute,
                    root=str(output_root),
                    task_prefix=state.request.output_policy.task_prefix,
                    board_namespace=state.request.output_policy.board_namespace,
                    max_bytes=min(
                        state.request.budget.max_serialized_bytes,
                        1_048_576,
                    ),
                    max_tasks=MAX_TASKBOARD_MATERIALIZATION_ENTRIES,
                )
        if duckdb_path:
            if not DatabaseTaskSource.available():
                raise SupervisorUnavailableError(
                    "optional DuckDB capability is unavailable"
                )
            absolute = output_root / duckdb_path
            absolute.parent.mkdir(parents=True, exist_ok=True)
            duckdb_source = DatabaseTaskSource(
                absolute,
                owner_id=str(mutation_map.get("authority_cid") or observation.caller),
                repository_tree_id=observation.dirty_worktree_root,
                plan_root_cid=preview.plan_root_cid,
            )
        expected = tuple(
            materialization_inputs.get("expected_effects")
            or preview.expected_materialization_effects
        )
        apply_request = plan_service.build_apply_request_from_admitted_preview(
            preview=preview,
            plan_request=plan_request,
            admission=state.admission,
            goal_graph=state.graph,
            observed_roots=plan_request.roots,
            idempotency_key=str(mutation_map["idempotency_key"]),
            lease_id=str(mutation_map["lease_id"]),
            fencing_token=int(mutation_map["fencing_epoch"]),
            expected_effects=expected,
            markdown_source=markdown_source,
            duckdb_source=duckdb_source,
            repository_tree_id=observation.dirty_worktree_root,
            event_cursor=preview.scan_cid,
            goal_cids=preview.admitted_goal_cids,
            task_cids=preview.admitted_task_cids,
        )
        try:
            return plan_service.apply_revision(apply_request, authorized=True)
        except Exception as exc:
            raise SupervisorUnavailableError(str(exc)) from exc

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
        extras = self._composition.extras
        if extras.get("scanner") is None and extras.get("prompt_supervisor_service") is None:
            return SupervisorObservation(
                run_id="",
                state="preview",
                health="unknown",
                event_cursor="",
                composition_cid=self.composition_cid,
                summary="preview-only; no durable effect authorized",
                values={"prompt_cid": prompt_cid, "effect_applied": False},
            )
        run = self._run_from_prompt(prompt, start_workers=False)
        payload = {
            "prompt_cid": prompt_cid,
            "effect_applied": False,
            "identities": dict(run.identities),
            "workflow_request_cid": run.identities.get("workflow_request_cid"),
            "plan_create_request_cid": run.identities.get(
                "plan_create_request_cid"
            ),
        }
        if prompt in json.dumps(payload):
            raise SupervisorError("prompt body leak denied")
        return SupervisorObservation(
            run_id=run.run_id,
            state="preview",
            health="unknown",
            event_cursor=run.event_cursor,
            composition_cid=self.composition_cid,
            summary="admitted preview; no durable effect authorized",
            values=payload,
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
            values={
                "run_revision": run.run_revision,
                "identities": dict(run.identities),
            },
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
        if run_id is not None and run_id in self._runs:
            return self._runs[run_id]
        if run_id is None and len(self._runs) == 1:
            return next(iter(self._runs.values()))
        if run_id is None and len(self._runs) > 1:
            raise SupervisorAmbiguityError(
                "multiple active runs; supply an exact run_id",
                candidates=tuple(sorted(self._runs)),
            )
        adopted = self._live_prompt_runs()
        if run_id is not None:
            for run in adopted:
                if run.run_id == run_id:
                    self._runs[run.run_id] = run
                    return run
            raise SupervisorAmbiguityError(
                f"unknown run_id {run_id!r}",
                candidates=tuple(
                    sorted({*self._runs, *(run.run_id for run in adopted)})
                ),
            )
        if len(adopted) == 1:
            run = adopted[0]
            self._runs[run.run_id] = run
            return run
        if not adopted:
            raise SupervisorAmbiguityError(
                "no active run; supply run_id or call run(prompt) first",
                candidates=(),
            )
        raise SupervisorAmbiguityError(
            "multiple active runs; supply an exact run_id",
            candidates=tuple(run.run_id for run in adopted),
        )

    def _live_prompt_runs(self) -> list[SupervisorRun]:
        """Adopt supervisors that are still alive under this state root."""

        root = self._composition.repository_root
        state = self._composition.state_root
        if root is None or state is None:
            try:
                from .service_factory import observe_production_bindings

                observation = observe_production_bindings(self._composition)
            except Exception:
                return []
            root = observation.repository_root
            state = observation.state_root
        from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
            live_prompt_supervisors,
        )

        rows = live_prompt_supervisors(state, root)
        adopted: list[SupervisorRun] = []
        for row in rows:
            adopted.append(
                SupervisorRun(
                    run_id=str(row["run_id"]),
                    run_revision=1,
                    composition_cid=self.composition_cid,
                    state="running",
                    health="healthy",
                    event_cursor=str(row["run_dir"]),
                    invocation_cid=str(row["run_id"]),
                    supervisor=self,
                    identities={
                        "supervisor_pid": row["supervisor_pid"],
                        "worker_pids": list(row["worker_pids"]),
                        "worker_task_cids": list(row["worker_task_cids"]),
                        "worker_task_keys": list(row["worker_task_keys"]),
                        "active_worker_count": len(row["worker_pids"]),
                        "workers_started": bool(row["worker_pids"]),
                        "worker_run_dir": str(row["run_dir"]),
                        "authorizes_execution": False,
                        "completion_authority": False,
                    },
                )
            )
        return adopted


class _DeterministicPromptPlanner:
    """Plan goals and subgoals from the intent autoencoder, without the model router."""

    def plan(self, request: Any, scan: Any, **_kwargs: Any) -> Any:
        from types import SimpleNamespace

        from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
            plan_with_intent_autoencoder,
        )

        body = getattr(getattr(request, "prompt_source", None), "transient_body", None)
        focus = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else ""
        graph, advice = plan_with_intent_autoencoder(request, scan, focus_text=focus)
        receipt = SimpleNamespace(
            to_dict=lambda: {
                "request_cid": request.request_cid,
                "scan_cid": scan.scan_cid,
                "plan_root_cid": graph.plan_root_cid,
                "intent_autoencoder_status": advice.get("status") or "",
                "intent_projection_ids": list(advice.get("projection_ids") or ()),
                "intent_subgoal_count": int(advice.get("subgoal_count") or 0),
                "authorizes_execution": False,
                "completion_authority": False,
            },
            fallback=SimpleNamespace(used=True),
        )
        return SimpleNamespace(graph=graph, receipt=receipt, used_fallback=True)


def prompt_repository_root(explicit: Path | str | None = None) -> Path | None:
    """Resolve the repository a prompt addresses.

    An explicit path is that repository. Otherwise use the nearest Git root
    when it carries the activated scheduler, and the package checkout when
    the caller is in a parent tree that does not.
    """

    from .service_factory import SCHEDULER_CONFIG_RELATIVE

    if explicit:
        return Path(explicit)
    nearest = _nearest_git_root(Path.cwd())
    if nearest is not None and (nearest / SCHEDULER_CONFIG_RELATIVE).is_file():
        return nearest
    package_root = Path(__file__).resolve().parents[3]
    if (package_root / SCHEDULER_CONFIG_RELATIVE).is_file():
        return package_root
    return nearest


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
