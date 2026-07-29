"""SCA-176: exact cross-component MCP++ mediation proofs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.runtime_mcp_invocation_trace import (
    REQUIRED_MEDIATION_STAGES,
    RUNTIME_MCP_INVOCATION_TRACE_INTERFACE,
    DispatchPipeline,
    DispatchPipelineStage,
    InterfaceDescriptor,
    MediationPath,
    MediationPathClass,
    MediationPathSegment,
    MediationTerminalState,
    MediationTraceRequest,
    RuntimeMcpInvocationTrace,
    RuntimeMcpInvocationTraceError,
    RuntimeMcpInvocationTracer,
    RuntimePackageTarget,
    build_mediation_path,
    compute_runtime_mcp_invocation_trace,
    compute_runtime_mcp_invocation_traces,
)
from ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph import (
    GRAPH_VERSION,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    ContractProvenance,
    SymbolicContractGraph,
)


SNAPSHOT = "repository-snapshot:sha256:runtime-mcp-mediation-fixture"

_REQUIRED = tuple(
    DispatchPipelineStage(name) for name in REQUIRED_MEDIATION_STAGES
)


def _descriptor(
    *,
    package_id: str = "ipfs_kit_py",
    route_id: str = "route:ipfs.add",
    schema_id: str = "schema:ipfs.add@1",
    function_id: str = "ipfs_kit_py.ipfs.add",
    display_name: str = "ipfs.add",
    behavior_id: str = "behavior:add",
    event_id: str = "event:add.receipt",
    receipt_id: str = "receipt:add",
    descriptor_id: str = "descriptor:ipfs.add",
) -> InterfaceDescriptor:
    return InterfaceDescriptor(
        route_id=route_id,
        schema_id=schema_id,
        function_id=function_id,
        behavior_id=behavior_id,
        event_id=event_id,
        receipt_id=receipt_id,
        package_id=package_id,
        descriptor_id=descriptor_id,
        display_name=display_name,
    )


def _mediated_path(
    *,
    path_class: MediationPathClass = MediationPathClass.TOOLS_DISPATCH,
    expected: InterfaceDescriptor | None = None,
    observed: InterfaceDescriptor | None = None,
    source_prefix: str = "td",
    start_line: int = 10,
) -> MediationPath:
    expected = expected or _descriptor()
    return build_mediation_path(
        path_class=path_class,
        stages=_REQUIRED,
        expected=expected,
        observed=observed or expected,
        source_prefix=source_prefix,
        start_line=start_line,
    )


def _request(
    *,
    operation_id: str = "ipfs.add",
    package_id: str = "ipfs_kit_py",
    paths: tuple[MediationPath, ...] = (),
    expected: InterfaceDescriptor | None = None,
    observed: InterfaceDescriptor | None = None,
    supported: bool = True,
    measured: bool = True,
) -> MediationTraceRequest:
    expected = expected or _descriptor(package_id=package_id)
    return MediationTraceRequest(
        operation_id=operation_id,
        package_id=package_id,
        expected_descriptor=expected,
        observed_descriptor=observed,
        paths=paths,
        supported=supported,
        measured=measured,
    )


def _span(path: str, line: int) -> dict[str, object]:
    return {
        "path": path,
        "source_sha256": "sha256:" + f"{line:064x}",
        "start_line": line,
        "start_column": 0,
        "end_line": line,
        "end_column": 20,
    }


def _node(
    key: str,
    kind: ContractNodeKind,
    *,
    payload: dict[str, object] | None = None,
) -> ContractGraphNode:
    return ContractGraphNode(
        kind=kind,
        stable_key=key,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        payload=payload or {"label": key},
        source_refs=(f"source:{key}",),
    )


def _edge(
    source: ContractGraphNode,
    target: ContractGraphNode,
    kind: ContractEdgeKind,
    *,
    line: int,
    stage: str | None = None,
    compatibility: bool = False,
    path_class: str | None = None,
    payload: dict[str, object] | None = None,
) -> ContractGraphEdge:
    values: dict[str, object] = {
        "source_span": _span("fixture/runtime_mcp.py", line),
    }
    if stage is not None:
        values["pipeline_stage"] = stage
    if compatibility:
        values["compatibility"] = True
        values["route_kind"] = "compatibility_route"
    if path_class is not None:
        values["path_class"] = path_class
    if payload:
        values.update(payload)
    return ContractGraphEdge(
        kind=kind,
        source=source.node_id,
        target=target.node_id,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        mandatory=True,
        payload=values,
        source_refs=(f"ast-source:{line}",),
    )


def test_interface_and_reviewed_pipeline_constants() -> None:
    assert (
        RUNTIME_MCP_INVOCATION_TRACE_INTERFACE
        == "RuntimeMcpInvocationTrace@1"
    )
    pipeline = DispatchPipeline.reviewed_default()
    assert pipeline.pipeline_id == "mcp-plus-plus-reviewed@1"
    assert set(pipeline.required_stage_values) == set(REQUIRED_MEDIATION_STAGES)
    assert {
        "health",
        "discovery",
        "call",
        "policy",
        "transport",
        "handler",
        "implementation",
    }.issubset({stage.value for stage in pipeline.stages})
    assert {item.value for item in RuntimePackageTarget} == {
        "ipfs_kit_py",
        "ipfs_datasets_py",
        "ipfs_accelerate_py",
        "agent_supervisor",
    }


def test_tools_dispatch_uses_reviewed_pipeline_or_refutes() -> None:
    expected = _descriptor()
    mediated = _mediated_path(
        path_class=MediationPathClass.TOOLS_DISPATCH,
        expected=expected,
        source_prefix="dispatch",
    )
    ok = compute_runtime_mcp_invocation_trace(
        _request(paths=(mediated,), expected=expected)
    )
    assert ok.terminal_state is MediationTerminalState.MEDIATED
    assert ok.reason_code == "exact_reviewed_pipeline_path"
    assert ok.proof_paths
    assert ok.identities_match
    assert ok.mediated_paths[0].path_class is MediationPathClass.TOOLS_DISPATCH

    # tools_dispatch that skips policy cannot prove mediation.
    incomplete_stages = (
        DispatchPipelineStage.CAPABILITY,
        DispatchPipelineStage.CONNECTOR,
        DispatchPipelineStage.CALL,
        DispatchPipelineStage.TRANSPORT,
        DispatchPipelineStage.HANDLER,
        DispatchPipelineStage.IMPLEMENTATION,
    )
    incomplete = build_mediation_path(
        path_class=MediationPathClass.TOOLS_DISPATCH,
        stages=incomplete_stages,
        expected=expected,
        source_prefix="skip-policy",
        start_line=100,
    )
    refuted = compute_runtime_mcp_invocation_trace(
        _request(paths=(incomplete,), expected=expected)
    )
    assert refuted.terminal_state is MediationTerminalState.REFUTED
    assert refuted.reason_code == "primary_path_missing_reviewed_pipeline_stage"
    assert not refuted.proof_paths
    assert refuted.mediated_paths  # primary path remains visible


def test_http_primary_path_requires_pipeline_and_identities() -> None:
    expected = _descriptor(
        package_id="ipfs_datasets_py",
        route_id="route:dataset.save",
        schema_id="schema:dataset.save@1",
        function_id="ipfs_datasets_py.dataset.save",
        display_name="dataset.save",
    )
    http_path = _mediated_path(
        path_class=MediationPathClass.HTTP,
        expected=expected,
        source_prefix="http",
        start_line=30,
    )
    ok = compute_runtime_mcp_invocation_trace(
        _request(
            operation_id="dataset.save",
            package_id="ipfs_datasets_py",
            paths=(http_path,),
            expected=expected,
        )
    )
    assert ok.terminal_state is MediationTerminalState.MEDIATED
    assert ok.mediated_paths[0].path_class is MediationPathClass.HTTP

    mismatched = _mediated_path(
        path_class=MediationPathClass.HTTP,
        expected=expected,
        observed=_descriptor(
            package_id="ipfs_datasets_py",
            route_id="route:dataset.save",
            schema_id="schema:dataset.save@1",
            function_id="ipfs_datasets_py.dataset.save_v2",
            display_name="dataset.save.v2",
        ),
        source_prefix="http-mismatch",
        start_line=50,
    )
    bad = compute_runtime_mcp_invocation_trace(
        _request(
            operation_id="dataset.save",
            package_id="ipfs_datasets_py",
            paths=(mismatched,),
            expected=expected,
        )
    )
    assert bad.terminal_state is MediationTerminalState.REFUTED
    assert bad.reason_code == "route_schema_function_identity_mismatch"


def test_name_only_descriptor_match_cannot_prove_mediation() -> None:
    expected = _descriptor(
        route_id="route:true",
        schema_id="schema:true@1",
        function_id="pkg.true_fn",
        display_name="same-name",
    )
    observed = _descriptor(
        route_id="route:other",
        schema_id="schema:other@1",
        function_id="pkg.other_fn",
        display_name="same-name",
    )
    assert expected.name_only_match(observed)
    assert not expected.matches(observed)
    path = _mediated_path(
        expected=expected,
        observed=observed,
        source_prefix="name-only",
        start_line=70,
    )
    trace = compute_runtime_mcp_invocation_trace(
        _request(paths=(path,), expected=expected, observed=observed)
    )
    assert trace.terminal_state is MediationTerminalState.REFUTED
    assert trace.reason_code == "descriptor_name_match_not_identity"
    assert path.name_only_match
    assert not path.proof_eligible


def test_direct_fetch_import_and_compatibility_bypasses_are_visible() -> None:
    expected = _descriptor(package_id="ipfs_accelerate_py")
    direct_fetch = build_mediation_path(
        path_class=MediationPathClass.DIRECT_FETCH,
        stages=(
            DispatchPipelineStage.CALL,
            DispatchPipelineStage.HANDLER,
            DispatchPipelineStage.IMPLEMENTATION,
        ),
        expected=expected,
        source_prefix="fetch",
        start_line=1,
        bypass=True,
    )
    direct_import = build_mediation_path(
        path_class=MediationPathClass.DIRECT_IMPORT,
        stages=(
            DispatchPipelineStage.HANDLER,
            DispatchPipelineStage.IMPLEMENTATION,
        ),
        expected=expected,
        source_prefix="import",
        start_line=20,
        bypass=True,
    )
    compatibility = build_mediation_path(
        path_class=MediationPathClass.COMPATIBILITY,
        stages=(
            DispatchPipelineStage.TRANSPORT,
            DispatchPipelineStage.HANDLER,
            DispatchPipelineStage.IMPLEMENTATION,
        ),
        expected=expected,
        source_prefix="compat",
        start_line=40,
        bypass=True,
    )
    # Even alongside a mediated primary path, bypasses remain visible.
    primary = _mediated_path(
        expected=expected,
        source_prefix="primary",
        start_line=80,
    )
    trace = compute_runtime_mcp_invocation_trace(
        _request(
            operation_id="model.infer",
            package_id="ipfs_accelerate_py",
            expected=expected,
            paths=(primary, direct_fetch, direct_import, compatibility),
        )
    )
    assert trace.terminal_state is MediationTerminalState.MEDIATED
    classes = {item.path_class for item in trace.bypass_paths}
    assert classes == {
        MediationPathClass.DIRECT_FETCH,
        MediationPathClass.DIRECT_IMPORT,
        MediationPathClass.COMPATIBILITY,
    }
    assert set(trace.to_dict()["visible_bypass_classes"]) == {
        "direct_fetch",
        "direct_import",
        "compatibility",
    }

    only_bypass = compute_runtime_mcp_invocation_trace(
        _request(
            operation_id="model.infer",
            package_id="ipfs_accelerate_py",
            expected=expected,
            paths=(direct_fetch, direct_import, compatibility),
        )
    )
    assert only_bypass.terminal_state is MediationTerminalState.REFUTED
    assert only_bypass.reason_code == "only_bypass_paths_visible"
    assert only_bypass.visible_bypasses
    assert not only_bypass.proof_paths


def test_native_supervisor_and_all_three_packages_receive_exact_states() -> None:
    packages = (
        RuntimePackageTarget.IPFS_KIT_PY,
        RuntimePackageTarget.IPFS_DATASETS_PY,
        RuntimePackageTarget.IPFS_ACCELERATE_PY,
        RuntimePackageTarget.AGENT_SUPERVISOR,
    )
    requests: list[MediationTraceRequest] = []
    for index, package in enumerate(packages):
        expected = _descriptor(
            package_id=package.value,
            route_id=f"route:{package.value}.op",
            schema_id=f"schema:{package.value}.op@1",
            function_id=f"{package.value}.native.op",
            display_name=f"{package.value}.op",
            descriptor_id=f"descriptor:{package.value}.op",
        )
        if package is RuntimePackageTarget.IPFS_DATASETS_PY:
            # Refuted: only compatibility bypass.
            path = build_mediation_path(
                path_class=MediationPathClass.COMPATIBILITY,
                stages=(
                    DispatchPipelineStage.TRANSPORT,
                    DispatchPipelineStage.HANDLER,
                ),
                expected=expected,
                source_prefix=f"pkg-{index}",
                start_line=10 * (index + 1),
                bypass=True,
            )
        elif package is RuntimePackageTarget.IPFS_ACCELERATE_PY:
            # Ambiguous: dynamic unresolved segment on a primary path.
            path = build_mediation_path(
                path_class=MediationPathClass.TOOLS_DISPATCH,
                stages=_REQUIRED,
                expected=expected,
                source_prefix=f"pkg-{index}",
                start_line=10 * (index + 1),
                dynamic_stages=("call",),
            )
        elif package is RuntimePackageTarget.AGENT_SUPERVISOR:
            # Unsupported family.
            requests.append(
                _request(
                    operation_id=f"{package.value}.op",
                    package_id=package.value,
                    expected=expected,
                    paths=(),
                    supported=False,
                )
            )
            continue
        else:
            path = _mediated_path(
                expected=expected,
                source_prefix=f"pkg-{index}",
                start_line=10 * (index + 1),
            )
        requests.append(
            _request(
                operation_id=f"{package.value}.op",
                package_id=package.value,
                expected=expected,
                paths=(path,),
            )
        )

    # Not-measured package covered as an extra operation on kit target.
    kit_expected = _descriptor(package_id="ipfs_kit_py")
    requests.append(
        _request(
            operation_id="ipfs_kit_py.unmeasured",
            package_id="ipfs_kit_py",
            expected=kit_expected,
            paths=(_mediated_path(expected=kit_expected, source_prefix="nm"),),
            measured=False,
        )
    )

    batch = compute_runtime_mcp_invocation_traces(requests)
    by_key = {
        (item.package_id, item.operation_id): item for item in batch.traces
    }

    assert (
        by_key[("ipfs_kit_py", "ipfs_kit_py.op")].terminal_state
        is MediationTerminalState.MEDIATED
    )
    assert (
        by_key[("ipfs_datasets_py", "ipfs_datasets_py.op")].terminal_state
        is MediationTerminalState.REFUTED
    )
    assert (
        by_key[("ipfs_accelerate_py", "ipfs_accelerate_py.op")].terminal_state
        is MediationTerminalState.AMBIGUOUS
    )
    assert (
        by_key[("agent_supervisor", "agent_supervisor.op")].terminal_state
        is MediationTerminalState.UNSUPPORTED
    )
    assert (
        by_key[("ipfs_kit_py", "ipfs_kit_py.unmeasured")].terminal_state
        is MediationTerminalState.NOT_MEASURED
    )

    # Closed terminal vocabulary is fully exercised.
    assert {item.terminal_state for item in batch.traces} == set(
        MediationTerminalState
    )
    # Deterministic package ordering.
    package_order = [item.package_id for item in batch.traces]
    assert package_order == sorted(package_order)
    round_trip = RuntimeMcpInvocationTrace.from_json(
        by_key[("ipfs_kit_py", "ipfs_kit_py.op")].to_json()
    )
    assert round_trip == by_key[("ipfs_kit_py", "ipfs_kit_py.op")]


def test_structural_graph_projection_and_incomplete_graph() -> None:
    capability = _node("capability:ipfs.add", ContractNodeKind.METHOD)
    connector = _node("connector:mcp++", ContractNodeKind.INTERFACE)
    call = _node("call:tools_dispatch", ContractNodeKind.CALL)
    policy = _node("policy:ucan", ContractNodeKind.POLICY)
    transport = _node("transport:http", ContractNodeKind.TRANSPORT)
    handler = _node("handler:add", ContractNodeKind.HANDLER)
    implementation = _node(
        "implementation:add", ContractNodeKind.SYMBOL
    )
    expected = _descriptor()
    identity_payload = {
        "route_id": expected.route_id,
        "schema_id": expected.schema_id,
        "function_id": expected.function_id,
        "behavior_id": expected.behavior_id,
        "event_id": expected.event_id,
        "receipt_id": expected.receipt_id,
        "package_id": expected.package_id,
        "descriptor_id": expected.descriptor_id,
        "display_name": expected.display_name,
        "path_class": "tools_dispatch",
    }
    edges = (
        _edge(
            capability,
            connector,
            ContractEdgeKind.DECLARES,
            line=1,
            stage="capability",
        ),
        _edge(
            connector,
            call,
            ContractEdgeKind.DISPATCHES_TO,
            line=2,
            stage="connector",
            path_class="tools_dispatch",
        ),
        _edge(
            call,
            policy,
            ContractEdgeKind.DISPATCHES_TO,
            line=3,
            stage="call",
            path_class="tools_dispatch",
        ),
        _edge(
            policy,
            transport,
            ContractEdgeKind.ENFORCED_BY,
            line=4,
            stage="policy",
        ),
        _edge(
            transport,
            handler,
            ContractEdgeKind.TRANSPORTED_BY,
            line=5,
            stage="transport",
        ),
        _edge(
            handler,
            implementation,
            ContractEdgeKind.HANDLED_BY,
            line=6,
            stage="handler",
        ),
        # Final hop carries identity binding + implementation stage.
        ContractGraphEdge(
            kind=ContractEdgeKind.IMPLEMENTS,
            source=implementation.node_id,
            target=implementation.node_id,
            snapshot_id=SNAPSHOT,
            provenance=ContractProvenance.AST,
            authority=ContractAuthority.SOURCE_OBSERVATION,
            version=GRAPH_VERSION,
            mandatory=True,
            payload={
                "source_span": _span("fixture/runtime_mcp.py", 7),
                "pipeline_stage": "implementation",
                **identity_payload,
            },
            source_refs=("ast-source:7",),
        ),
    )
    # Self-loop IMPLEMENTS is awkward; use a distinct impl target instead.
    impl_fn = _node(
        "function:ipfs_kit_py.ipfs.add",
        ContractNodeKind.SYMBOL,
        payload={"label": "ipfs.add", **identity_payload},
    )
    edges = (
        _edge(
            capability,
            connector,
            ContractEdgeKind.DECLARES,
            line=1,
            stage="capability",
        ),
        _edge(
            connector,
            call,
            ContractEdgeKind.DISPATCHES_TO,
            line=2,
            stage="connector",
            path_class="tools_dispatch",
        ),
        _edge(
            call,
            policy,
            ContractEdgeKind.DISPATCHES_TO,
            line=3,
            stage="call",
            path_class="tools_dispatch",
        ),
        _edge(
            policy,
            transport,
            ContractEdgeKind.ENFORCED_BY,
            line=4,
            stage="policy",
        ),
        _edge(
            transport,
            handler,
            ContractEdgeKind.TRANSPORTED_BY,
            line=5,
            stage="transport",
        ),
        _edge(
            handler,
            implementation,
            ContractEdgeKind.HANDLED_BY,
            line=6,
            stage="handler",
        ),
        _edge(
            implementation,
            impl_fn,
            ContractEdgeKind.IMPLEMENTS,
            line=7,
            stage="implementation",
            payload=identity_payload,
        ),
    )
    graph = SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=(
            capability,
            connector,
            call,
            policy,
            transport,
            handler,
            implementation,
            impl_fn,
        ),
        edges=edges,
    )
    tracer = RuntimeMcpInvocationTracer(graph=graph)
    trace = tracer.trace_from_structural_graph(
        operation_id="ipfs.add",
        package_id="ipfs_kit_py",
        source_node_id=capability.node_id,
        target_node_ids=(impl_fn.node_id,),
        expected_descriptor=expected,
        path_class_hint=MediationPathClass.TOOLS_DISPATCH,
    )
    assert trace.terminal_state is MediationTerminalState.MEDIATED
    assert trace.structural_trace_id
    assert trace.proof_paths
    assert all(
        stage in trace.proof_paths[0].stages
        for stage in REQUIRED_MEDIATION_STAGES
    )

    incomplete = replace(
        graph,
        mandatory_edge_ids=(
            *graph.mandatory_edge_ids,
            "baguqeeramissingmandatoryedge",
        ),
        graph_root_claim="",
        identity_claim=None,
    )
    incomplete_trace = RuntimeMcpInvocationTracer(
        graph=incomplete
    ).trace_from_structural_graph(
        operation_id="ipfs.add",
        package_id="ipfs_kit_py",
        source_node_id=capability.node_id,
        target_node_ids=(impl_fn.node_id,),
        expected_descriptor=expected,
    )
    assert (
        incomplete_trace.terminal_state
        is MediationTerminalState.NOT_MEASURED
    )
    assert incomplete_trace.reason_code == "incomplete_symbolic_contract_graph"
    assert incomplete_trace.complete is False


def test_content_identity_and_tampering_fail_closed() -> None:
    expected = _descriptor()
    path = _mediated_path(expected=expected, source_prefix="id")
    trace = compute_runtime_mcp_invocation_trace(
        _request(paths=(path,), expected=expected)
    )
    payload = trace.to_dict()
    payload["reason_code"] = "tampered"
    with pytest.raises(
        RuntimeMcpInvocationTraceError, match="trace identity"
    ):
        RuntimeMcpInvocationTrace.from_dict(payload)

    path_payload = path.to_dict()
    path_payload["segments"][0]["edge_id"] = "tampered-edge"
    with pytest.raises(
        RuntimeMcpInvocationTraceError, match="segment identity"
    ):
        MediationPath.from_dict(path_payload)

    descriptor_payload = expected.to_dict()
    descriptor_payload["route_id"] = "route:tampered"
    with pytest.raises(
        RuntimeMcpInvocationTraceError, match="interface descriptor identity"
    ):
        InterfaceDescriptor.from_dict(descriptor_payload)


def test_batch_rejects_duplicate_package_operations() -> None:
    expected = _descriptor()
    path = _mediated_path(expected=expected)
    request = _request(paths=(path,), expected=expected)
    with pytest.raises(
        RuntimeMcpInvocationTraceError, match="duplicate package/operation"
    ):
        compute_runtime_mcp_invocation_traces((request, request))


def test_segment_without_source_provenance_is_not_measured() -> None:
    expected = _descriptor()
    bare = MediationPathSegment(
        stage=DispatchPipelineStage.CALL,
        source_node_id="a",
        target_node_id="b",
        edge_id="edge:bare",
        source_ids=(),
        source_spans=(),
    )
    # Build a full required-stage path but with one bare segment replacing call.
    good_segments = list(_mediated_path(expected=expected).segments)
    replaced = []
    for segment in good_segments:
        if segment.stage is DispatchPipelineStage.CALL:
            replaced.append(
                MediationPathSegment(
                    stage=segment.stage,
                    source_node_id=segment.source_node_id,
                    target_node_id=segment.target_node_id,
                    edge_id=segment.edge_id,
                    edge_kind=segment.edge_kind,
                    source_ids=(),
                    source_spans=(),
                )
            )
        else:
            replaced.append(segment)
    path = MediationPath(
        path_class=MediationPathClass.TOOLS_DISPATCH,
        segments=tuple(replaced),
        expected_descriptor=expected,
        observed_descriptor=expected,
    )
    assert not path.proof_eligible
    assert bare.has_exact_source is False
    trace = compute_runtime_mcp_invocation_trace(
        _request(paths=(path,), expected=expected)
    )
    assert trace.terminal_state is MediationTerminalState.NOT_MEASURED
    assert trace.reason_code == "path_source_provenance_incomplete"


def test_pipeline_cover_helper_and_round_trip_batch() -> None:
    pipeline = DispatchPipeline.reviewed_default()
    assert pipeline.covers(REQUIRED_MEDIATION_STAGES)
    assert not pipeline.covers(
        ("capability", "call", "handler", "implementation")
    )
    expected = _descriptor(package_id="ipfs_kit_py")
    batch = compute_runtime_mcp_invocation_traces(
        (
            _request(
                operation_id="a",
                package_id="ipfs_kit_py",
                expected=expected,
                paths=(
                    _mediated_path(
                        expected=expected, source_prefix="a", start_line=1
                    ),
                ),
            ),
            _request(
                operation_id="b",
                package_id="ipfs_datasets_py",
                expected=_descriptor(package_id="ipfs_datasets_py"),
                paths=(),
                measured=False,
            ),
        )
    )
    encoded = batch.to_json()
    decoded = type(batch).from_dict(
        __import__("json").loads(encoded)
    )
    assert decoded.batch_id == batch.batch_id
    assert decoded.terminal_states_by_package()["ipfs_kit_py"] == "mediated"
    assert (
        decoded.terminal_states_by_package()["ipfs_datasets_py"]
        == "not_measured"
    )
