from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.prompt_goal_planner import (
    parse_prompt_goal_graph,
)
from ipfs_accelerate_py.agent_supervisor.prompt_workflow import (
    OutputMode,
    PromptSupervisorService,
    PromptWorkflowPreviewReceipt,
    RecordStatus,
    WorkflowOutcome,
    prompt_workflow_cid,
)
from test.api.test_agent_supervisor_prompt_goal_planner import (
    _encoded_proposal,
    _request,
    _scan,
)


def _cid(name: str) -> str:
    return prompt_workflow_cid({"service-fixture": name})


class _Scanner:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, request):
        self.calls += 1
        return _scan(request)


class _Planner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, request, scan, **_kwargs):
        self.calls += 1
        graph = parse_prompt_goal_graph(
            _encoded_proposal(scan), request, scan
        )
        receipt = SimpleNamespace(
            to_dict=lambda: {
                "request_cid": request.request_cid,
                "scan_cid": scan.scan_cid,
                "plan_root_cid": graph.plan_root_cid,
                "provider": {"response_sha256": "sha256:" + "a" * 64},
            },
            fallback=SimpleNamespace(used=False),
        )
        return SimpleNamespace(
            graph=graph,
            receipt=receipt,
            used_fallback=False,
        )


class _Admission:
    def __init__(self, *, admitted: bool = True) -> None:
        self.admitted = admitted
        self.calls = 0

    def admit(self, request, scan, graph, _planning):
        self.calls += 1
        if not self.admitted:
            finding = SimpleNamespace(
                finding_id=_cid("finding:security"),
                code="security.denied",
                to_dict=lambda: {
                    "finding_id": _cid("finding:security"),
                    "code": "security.denied",
                },
            )
            receipt = SimpleNamespace(
                findings=(finding,),
                to_dict=lambda: {
                    "candidate_plan_cid": graph.plan_root_cid,
                    "verdict": "rejected",
                    "findings": [finding.to_dict()],
                },
            )
            return SimpleNamespace(
                admitted=False,
                admitted_graph=None,
                plan_root_cid="",
                reason_codes=("security.denied",),
                receipt=receipt,
            )
        final_root = prompt_workflow_cid(
            {
                "schema": "service-admitted-plan@1",
                "candidate_plan_cid": graph.plan_root_cid,
                "repository_tree_id": scan.dirty_worktree_root,
                "policy_root": request.policy_root,
            }
        )
        receipt = SimpleNamespace(
            findings=(),
            to_dict=lambda: {
                "candidate_plan_cid": graph.plan_root_cid,
                "final_plan_cid": final_root,
                "repository_tree_id": scan.dirty_worktree_root,
                "verdict": "admitted",
            },
        )
        return SimpleNamespace(
            admitted=True,
            admitted_graph=graph,
            plan_root_cid=final_root,
            task_cids=tuple(sorted(task.task_cid for task in graph.tasks)),
            receipt=receipt,
        )


class _Materializer:
    def __init__(self, kind: str, *, fail_calls: int = 0) -> None:
        self.kind = kind
        self.fail_calls = fail_calls
        self.calls = 0
        self.writes = 0

    def materialize(self, admission, *, request, preview, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_calls:
            raise RuntimeError(f"{self.kind} projection unavailable")
        changed = self.writes == 0
        if changed:
            self.writes += 1
        return {
            "committed": True,
            "changed": changed,
            "replayed": not changed,
            "projection_cid": _cid(f"{self.kind}:projection"),
            "source_schema": f"fixture/{self.kind}@1",
            "plan_root_cid": preview.plan_root_cid,
            "revision": 1,
            "task_cids": admission.task_cids,
            "event_cursor": f"{self.kind}:cursor:0",
        }


class _RootObserver:
    def __init__(self) -> None:
        self.override: dict[str, object] = {}

    def __call__(self, request, scan, preview):
        return {
            "request_cid": request.request_cid,
            "repository_root_cid": request.repository_root_cid,
            "scan_cid": scan.scan_cid,
            "dirty_worktree_root": scan.dirty_worktree_root,
            "plan_root_cid": preview.plan_root_cid,
            "program_root": request.program_root,
            "policy_root": request.policy_root,
            "intent_ir_root": request.intent_ir_root,
            "legal_ir_root": request.legal_ir_root,
            "security_ir_root": request.security_ir_root,
            "output_policy_cid": request.output_policy.content_id,
            "catalog_root": preview.catalog_root,
            "output_mode": request.output_policy.mode.value,
            "markdown_path": request.output_policy.markdown_path,
            "duckdb_path": request.output_policy.duckdb_path,
            **self.override,
        }


class _ControlService:
    def __init__(self, *, partial_starts: int = 0) -> None:
        self.materialize_calls = 0
        self.start_calls = 0
        self.partial_starts = partial_starts

    def workflow_materialize(self, _request):
        self.materialize_calls += 1
        return SimpleNamespace(
            succeeded=True,
            audit_receipt_id=_cid("control:materialize"),
            result_id=_cid("control:materialize-result"),
        )

    def start(self, request):
        self.start_calls += 1
        partial = self.start_calls <= self.partial_starts
        expected = tuple(request.expected_effects)
        return SimpleNamespace(
            succeeded=not partial,
            audit_receipt_id=_cid(f"control:start:{self.start_calls}"),
            result_id=_cid(f"control:start-result:{self.start_calls}"),
            effects=(
                ()
                if partial
                else tuple(
                    SimpleNamespace(effect_id=item.effect_id, applied=True)
                    for item in expected
                )
            ),
            data=(
                {}
                if partial
                else {
                    "process_identity": "process:fixture:1",
                    "event_cursor": "supervisor:cursor:1",
                }
            ),
        )


def _service(
    *,
    admitted: bool = True,
    mode: OutputMode = OutputMode.BOTH,
    duckdb_fail_calls: int = 0,
    control_service=None,
):
    request = _request()
    if mode is OutputMode.MARKDOWN:
        request = replace(
            request,
            output_policy=replace(
                request.output_policy,
                mode=mode,
                duckdb_path="",
            ),
        )
    elif mode is OutputMode.DUCKDB:
        request = replace(
            request,
            output_policy=replace(
                request.output_policy,
                mode=mode,
                markdown_path="",
                duckdb_path="state/prompt.duckdb",
            ),
        )
    else:
        request = replace(
            request,
            output_policy=replace(
                request.output_policy,
                mode=mode,
                duckdb_path="state/prompt.duckdb",
            ),
        )
    scanner = _Scanner()
    planner = _Planner()
    admission = _Admission(admitted=admitted)
    markdown = _Materializer("markdown")
    duckdb = _Materializer("duckdb", fail_calls=duckdb_fail_calls)
    roots = _RootObserver()
    receipts: dict[str, dict] = {}
    service = PromptSupervisorService(
        control_service=control_service,
        scanner=scanner,
        planner=planner,
        admission=admission,
        markdown_materializer=markdown,
        duckdb_materializer=duckdb,
        root_observer=roots,
        receipt_store=receipts,
        catalog_root=_cid("catalog"),
        clock_ms=lambda: 100,
    )
    return (
        request,
        service,
        scanner,
        planner,
        admission,
        markdown,
        duckdb,
        roots,
        receipts,
    )


def _materialize_control(request, preview, *, key="materialize:one", path=None):
    return SimpleNamespace(
        operation=Operation.WORKFLOW_MATERIALIZE,
        request_id=_cid(f"request:{key}:{path or 'same'}"),
        repository_root=request.repository_root,
        state_root="/workspace/repository/state",
        parameters={
            "preview_ref": preview.receipt_cid,
            "preview_root": preview.plan_root_cid,
            "output_mode": request.output_policy.mode.value,
            "markdown_path": (
                request.output_policy.markdown_path if path is None else path
            ),
            "duckdb_path": request.output_policy.duckdb_path,
            "catalog_root": preview.catalog_root,
        },
        idempotency_key=key,
        lease_id="lease:materialize",
        fencing_epoch=3,
    )


def _start_control(request, materialization, *, key="start:one"):
    effect = SimpleNamespace(effect_id="start:process")
    return SimpleNamespace(
        operation=Operation.START,
        request_id=_cid(f"request:{key}"),
        repository_root=request.repository_root,
        state_root=request.state_root,
        parameters={
            "materialization_ref": materialization.materialization_cid,
            "plan_root_cid": materialization.plan_root_cid,
            "supervisor_profile": request.supervisor_profile,
        },
        idempotency_key=key,
        expected_effects=(effect,),
    )


def test_preview_is_body_free_root_bound_rejected_or_exactly_replayed() -> None:
    (
        request,
        service,
        scanner,
        planner,
        admission,
        _markdown,
        _duckdb,
        _roots,
        receipts,
    ) = _service()

    first = service.preview(request)
    replay = service.preview(request)

    assert isinstance(first, PromptWorkflowPreviewReceipt)
    assert replay is first
    assert first.status is RecordStatus.ADMITTED
    assert first.request_cid == request.request_cid
    assert first.program_root == request.program_root
    assert first.intent_ir_root == request.intent_ir_root
    assert first.legal_ir_root == request.legal_ir_root
    assert first.security_ir_root == request.security_ir_root
    assert first.output_policy_cid == request.output_policy.content_id
    assert first.catalog_root == _cid("catalog")
    assert first.admitted_goal_cids and first.admitted_task_cids
    assert planner.calls == admission.calls == scanner.calls == 1
    assert request.prompt_source.transient_body.decode() not in str(receipts)
    assert all("response_sha256" not in str(record) for record in receipts.values())

    rejected_request, rejected_service, *_ = _service(admitted=False)
    rejected = rejected_service.preview(rejected_request)
    assert rejected.status is RecordStatus.REJECTED
    assert not rejected.admitted_task_cids
    assert rejected.rejected_branch_cids
    assert rejected.rejection_reasons == ("security.denied",)


def test_materialize_requires_separate_authority_and_exact_replay_writes_once() -> None:
    (
        request,
        service,
        _scanner,
        planner,
        _admission,
        markdown,
        duckdb,
        roots,
        _receipts,
    ) = _service()
    preview = service.preview(request)

    denied = service.materialize(preview)
    assert denied.outcome is WorkflowOutcome.FAILED
    assert denied.failure_codes == ("missing_authority",)
    assert markdown.writes == duckdb.writes == 0

    first = service.materialize(
        preview,
        authorization=_cid("authority:materialize"),
        idempotency_key="materialize:one",
        lease_id="lease:one",
        fencing_epoch=1,
    )
    replay = service.materialize(
        preview,
        authorization=_cid("authority:materialize"),
        idempotency_key="materialize:one",
        lease_id="lease:one",
        fencing_epoch=1,
    )

    assert first.outcome is WorkflowOutcome.MATERIALIZED
    assert replay.receipt_cid == first.receipt_cid
    assert markdown.writes == duckdb.writes == 1
    assert markdown.calls == duckdb.calls == 1
    assert planner.calls == 1
    assert len(first.task_source_identities) == 2
    assert set(first.event_cursors) == {"duckdb", "markdown"}
    assert set(first.observed_effects) == set(first.expected_effects)

    conflict = service.materialize(
        preview,
        authorization=_cid("authority:other"),
        idempotency_key="materialize:one",
        lease_id="lease:one",
        fencing_epoch=1,
    )
    assert conflict.failure_codes == ("idempotency_conflict",)
    assert markdown.writes == duckdb.writes == 1

    roots.override["scan_cid"] = _cid("scan:stale")
    stale = service.materialize(
        preview,
        authorization=_cid("authority:new"),
        idempotency_key="materialize:two",
        lease_id="lease:two",
        fencing_epoch=2,
    )
    assert stale.failure_codes == ("stale_roots",)
    assert markdown.writes == duckdb.writes == 1


def test_partial_dual_projection_has_exact_resume_and_no_duplicate_write() -> None:
    (
        request,
        service,
        _scanner,
        _planner,
        _admission,
        markdown,
        duckdb,
        _roots,
        _receipts,
    ) = _service(duckdb_fail_calls=1)
    preview = service.preview(request)
    mutation = {
        "authorization": _cid("authority:dual"),
        "idempotency_key": "materialize:dual",
        "lease_id": "lease:dual",
        "fencing_epoch": 5,
    }

    partial = service.materialize(preview, **mutation)
    resumed = service.materialize(preview, **mutation)
    replay = service.materialize(preview, **mutation)

    assert partial.outcome is WorkflowOutcome.PARTIAL
    assert partial.safe_continuation == f"materialize:{preview.receipt_cid}"
    assert len(partial.task_source_identities) == 1
    assert partial.observed_effects == (
        f"write_markdown:{request.output_policy.markdown_path}",
    )
    assert resumed.outcome is WorkflowOutcome.MATERIALIZED
    assert replay.receipt_cid == resumed.receipt_cid
    assert markdown.calls == 2
    assert markdown.writes == 1
    assert duckdb.calls == 2
    assert duckdb.writes == 1


def test_control_bound_bootstrap_resumes_partial_start_without_reprojection() -> None:
    control = _ControlService(partial_starts=1)
    (
        base,
        service,
        _scanner,
        planner,
        _admission,
        markdown,
        duckdb,
        _roots,
        _receipts,
    ) = _service(control_service=control)
    request = replace(
        base,
        dry_run=False,
        materialize=True,
        start_after_materialize=True,
        supervisor_profile="local-parallel",
        state_root="/workspace/repository/state",
        authority_cid=_cid("workflow:authority"),
        idempotency_key="workflow:bootstrap",
        lease_id="lease:bootstrap",
        fencing_epoch=7,
    )
    preview = service.preview(request)
    materialize_request = _materialize_control(request, preview)

    materialized = service.materialize(
        preview, control_request=materialize_request
    )
    assert materialized.materialization is not None
    start_request = _start_control(
        request, materialized.materialization
    )
    partial = service.start(
        materialized.materialization,
        control_request=start_request,
    )
    resumed = service.start(
        materialized.materialization,
        control_request=start_request,
    )
    replay = service.bootstrap(
        request,
        materialize_control_request=materialize_request,
        start_control_request=start_request,
    )

    assert partial.outcome is WorkflowOutcome.PARTIAL
    assert partial.failure_codes == ("partial_start",)
    assert partial.materialization == materialized.materialization
    assert resumed.outcome is WorkflowOutcome.STARTED
    assert resumed.run is not None
    assert resumed.run.process_identity_cid
    assert replay.receipt_cid == resumed.receipt_cid
    assert planner.calls == 1
    assert markdown.writes == duckdb.writes == 1
    assert control.materialize_calls == 1
    assert control.start_calls == 2


def test_changed_control_output_binding_fails_closed_before_projection() -> None:
    control = _ControlService()
    (
        request,
        service,
        _scanner,
        _planner,
        _admission,
        markdown,
        duckdb,
        _roots,
        _receipts,
    ) = _service(control_service=control)
    preview = service.preview(request)
    changed = _materialize_control(
        request,
        preview,
        path="plans/other.todo.md",
    )

    result = service.materialize(preview, control_request=changed)

    assert result.outcome is WorkflowOutcome.FAILED
    assert result.failure_codes == ("materialization_control_rejected",)
    assert markdown.writes == duckdb.writes == 0
    assert control.materialize_calls == 0
