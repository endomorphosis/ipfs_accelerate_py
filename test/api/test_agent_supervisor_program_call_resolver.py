"""Tests for conservative program call/import resolution (VFS-009)."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.program_call_resolver import (
    RESOLVER_VERSION,
    CallResolution,
    CallResolverError,
    MissingEvidenceError,
    ProgramCallResolver,
    ReasonCode,
    ResolutionEvidence,
    ResolverCatalog,
    confidence_for,
    make_resolution,
    resolve_program_calls,
    resolve_relative_module,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    ProgramEdgeKind,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan,
    build_program_graph,
    make_edge,
    make_node,
)


FOREST_ID = "forest:test-vfs-009"
PRODUCER = "program-ast-adapter@1"
BLOB_A = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BLOB_B = "baguqeerbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _span(line: int = 1, col: int = 0) -> SourceSpan:
    return SourceSpan(
        line_start=line, column_start=col, line_end=line, column_end=col + 8
    )


def _node(
    kind: ProgramNodeKind | str,
    key: str,
    *,
    component_id: str = "",
    blob_cid: str = BLOB_A,
    forest_id: str = FOREST_ID,
    qualified_name: str = "",
    path: str = "",
    language: str = "python",
    resolver_status: ResolverStatus | str = ResolverStatus.UNRESOLVED,
    record: dict[str, Any] | None = None,
) -> Any:
    return make_node(
        kind=kind,
        record_key=key,
        producer=PRODUCER,
        blob_cid=blob_cid,
        forest_id=forest_id,
        component_id=component_id or key,
        qualified_name=qualified_name or key,
        path=path,
        language=language,
        span=_span(),
        resolver_status=resolver_status,
        record=record or {},
    )


def _edge(
    source: str,
    target: str,
    kind: ProgramEdgeKind | str,
    *,
    component_id: str = "comp",
    resolver_status: ResolverStatus | str = ResolverStatus.UNRESOLVED,
    record: dict[str, Any] | None = None,
) -> Any:
    return make_edge(
        source=source,
        target=target,
        kind=kind,
        producer=PRODUCER,
        blob_cid=BLOB_A,
        forest_id=FOREST_ID,
        component_id=component_id,
        span=_span(2),
        resolver_status=resolver_status,
        record=record or {},
    )


def _evidence(**overrides: Any) -> ResolutionEvidence:
    payload = {
        "rule_id": "rule:test",
        "producer": PRODUCER,
        "blob_cid": BLOB_A,
        "forest_id": FOREST_ID,
        "span": _span(),
        "source_record_key": "site",
        "target_record_key": "target",
        "notes": {},
    }
    payload.update(overrides)
    return ResolutionEvidence.from_dict(payload)


# ---------------------------------------------------------------------------
# Relative / package imports, aliases
# ---------------------------------------------------------------------------


def test_resolve_relative_module_helper() -> None:
    assert resolve_relative_module("pkg.sub.mod", ".sibling") == "pkg.sub.sibling"
    assert resolve_relative_module("pkg.sub.mod", "..other") == "pkg.other"
    assert resolve_relative_module("pkg.sub", ".nested", is_package=True) == "pkg.sub.nested"
    assert resolve_relative_module("pkg.mod", "absolute.mod") == "absolute.mod"
    with pytest.raises(CallResolverError):
        resolve_relative_module("pkg", "....escape")


def test_relative_import_resolves_statically() -> None:
    mod_a = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.a",
        component_id="module:pkg.a",
        qualified_name="pkg.a",
        path="pkg/a.py",
    )
    mod_b = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.b",
        component_id="module:pkg.b",
        qualified_name="pkg.b",
        path="pkg/b.py",
    )
    imp = _node(
        ProgramNodeKind.IMPORT,
        "import:pkg.a:b",
        component_id="module:pkg.a",
        path="pkg/a.py",
        qualified_name="b",
        record={"target": ".b", "relative_level": 1, "alias": "b"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod_a, mod_b, imp),
        edges=(
            _edge(mod_a.node_id, imp.node_id, ProgramEdgeKind.IMPORTS, component_id="module:pkg.a"),
        ),
        producer=PRODUCER,
    )
    result = resolve_program_calls(graph)
    res = result.resolutions_for_site(imp.node_id)[0]
    assert res.status is ResolverStatus.RESOLVED_STATIC
    assert res.reason_code is ReasonCode.ALIAS_BINDING
    assert res.targets == ("pkg.b",)
    assert res.confidence == confidence_for(res.status, res.reason_code)
    assert res.evidence


def test_package_import_and_uninstalled_dependency() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:app.main",
        component_id="module:app.main",
        qualified_name="app.main",
        path="app/main.py",
    )
    installed = _node(
        ProgramNodeKind.MODULE,
        "module:local_util",
        component_id="module:local_util",
        qualified_name="local_util",
        path="local_util/__init__.py",
    )
    imp_ok = _node(
        ProgramNodeKind.IMPORT,
        "import:app.main:local_util",
        component_id="module:app.main",
        qualified_name="local_util",
        record={"target": "local_util"},
    )
    imp_missing = _node(
        ProgramNodeKind.IMPORT,
        "import:app.main:not_installed_pkg",
        component_id="module:app.main",
        qualified_name="not_installed_pkg",
        record={"target": "not_installed_pkg.api"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod, installed, imp_ok, imp_missing),
        producer=PRODUCER,
    )
    catalog = ResolverCatalog(installed_packages=frozenset({"local_util", "app"}))
    result = resolve_program_calls(graph, catalog=catalog)
    ok = result.resolutions_for_site(imp_ok.node_id)[0]
    missing = result.resolutions_for_site(imp_missing.node_id)[0]
    assert ok.status is ResolverStatus.RESOLVED_STATIC
    assert ok.reason_code is ReasonCode.PACKAGE_IMPORT
    assert missing.status is ResolverStatus.EXTERNAL
    assert missing.reason_code is ReasonCode.UNINSTALLED_DEPENDENCY
    assert missing.confidence == confidence_for(
        ResolverStatus.EXTERNAL, ReasonCode.UNINSTALLED_DEPENDENCY
    )


def test_optional_import_is_candidate() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:app.opt",
        component_id="module:app.opt",
        qualified_name="app.opt",
    )
    imp = _node(
        ProgramNodeKind.IMPORT,
        "import:app.opt:maybe",
        component_id="module:app.opt",
        qualified_name="maybe",
        record={"target": "maybe_dep", "optional": True},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, imp), producer=PRODUCER)
    res = resolve_program_calls(graph).resolutions_for_site(imp.node_id)[0]
    assert res.status is ResolverStatus.CANDIDATE
    assert res.reason_code is ReasonCode.OPTIONAL_IMPORT
    assert res.is_frontier


def test_namespace_package_is_ambiguous_without_concrete_module() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:app.ns",
        component_id="module:app.ns",
        qualified_name="app.ns",
    )
    imp = _node(
        ProgramNodeKind.IMPORT,
        "import:app.ns:ns_pkg",
        component_id="module:app.ns",
        qualified_name="plugin",
        record={"target": "ns_pkg.plugin"},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, imp), producer=PRODUCER)
    catalog = ResolverCatalog(namespace_packages=frozenset({"ns_pkg"}))
    res = resolve_program_calls(graph, catalog=catalog).resolutions_for_site(imp.node_id)[0]
    assert res.status is ResolverStatus.AMBIGUOUS
    assert res.reason_code is ReasonCode.NAMESPACE_PACKAGE


# ---------------------------------------------------------------------------
# Re-exports and loops
# ---------------------------------------------------------------------------


def test_reexport_resolves_through_chain() -> None:
    mod_impl = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.impl",
        component_id="module:pkg.impl",
        qualified_name="pkg.impl",
    )
    symbol = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.impl.helper",
        component_id="module:pkg.impl",
        qualified_name="pkg.impl.helper",
    )
    mod_mid = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.mid",
        component_id="module:pkg.mid",
        qualified_name="pkg.mid",
    )
    reexport_mid = _node(
        ProgramNodeKind.EXPORT,
        "export:pkg.mid.helper",
        component_id="module:pkg.mid",
        qualified_name="pkg.mid.helper",
        record={
            "kind": "re_export",
            "from_module": "pkg.impl",
            "export_name": "helper",
            "re_export": True,
        },
    )
    mod_api = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.api",
        component_id="module:pkg.api",
        qualified_name="pkg.api",
    )
    reexport_api = _node(
        ProgramNodeKind.EXPORT,
        "export:pkg.api.helper",
        component_id="module:pkg.api",
        qualified_name="pkg.api.helper",
        record={
            "kind": "re_export",
            "from_module": "pkg.mid",
            "export_name": "helper",
            "re_export": True,
        },
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod_impl, symbol, mod_mid, reexport_mid, mod_api, reexport_api),
        producer=PRODUCER,
    )
    res = resolve_program_calls(graph).resolutions_for_site(reexport_api.node_id)[0]
    assert res.status is ResolverStatus.RESOLVED_STATIC
    assert res.reason_code is ReasonCode.REEXPORT
    assert res.targets == ("pkg.impl.helper",)


def test_reexport_loop_is_ambiguous() -> None:
    mod_a = _node(
        ProgramNodeKind.MODULE,
        "module:loop.a",
        component_id="module:loop.a",
        qualified_name="loop.a",
    )
    mod_b = _node(
        ProgramNodeKind.MODULE,
        "module:loop.b",
        component_id="module:loop.b",
        qualified_name="loop.b",
    )
    exp_a = _node(
        ProgramNodeKind.EXPORT,
        "export:loop.a.f",
        component_id="module:loop.a",
        qualified_name="loop.a.f",
        record={
            "kind": "re_export",
            "from_module": "loop.b",
            "export_name": "f",
            "re_export": True,
        },
    )
    exp_b = _node(
        ProgramNodeKind.EXPORT,
        "export:loop.b.f",
        component_id="module:loop.b",
        qualified_name="loop.b.f",
        record={
            "kind": "re_export",
            "from_module": "loop.a",
            "export_name": "f",
            "re_export": True,
        },
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod_a, mod_b, exp_a, exp_b),
        producer=PRODUCER,
    )
    res = resolve_program_calls(graph).resolutions_for_site(exp_a.node_id)[0]
    assert res.status is ResolverStatus.AMBIGUOUS
    assert res.reason_code is ReasonCode.REEXPORT_LOOP
    assert res.is_frontier
    assert len(res.targets) >= 2


# ---------------------------------------------------------------------------
# Class/member calls and same-name collisions
# ---------------------------------------------------------------------------


def test_class_member_call_resolves() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.svc",
        component_id="module:pkg.svc",
        qualified_name="pkg.svc",
    )
    method = _node(
        ProgramNodeKind.DEFINITION,
        "def:pkg.svc.Worker.run",
        component_id="module:pkg.svc",
        qualified_name="pkg.svc.Worker.run",
        record={"owner": "pkg.svc.Worker", "member": "run"},
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:pkg.svc:Worker.run",
        component_id="module:pkg.svc",
        qualified_name="pkg.svc.Worker.run",
        record={"callee": "pkg.svc.Worker.run"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=(mod, method, call), producer=PRODUCER
    )
    res = resolve_program_calls(graph).resolutions_for_site(call.node_id)[0]
    assert res.status is ResolverStatus.RESOLVED_STATIC
    assert res.reason_code is ReasonCode.CLASS_MEMBER
    assert res.edge_kind is ProgramEdgeKind.CALLS
    assert res.targets == ("pkg.svc.Worker.run",)


def test_adversarial_same_name_functions_never_pick_one() -> None:
    mod_a = _node(
        ProgramNodeKind.MODULE,
        "module:alpha",
        component_id="module:alpha",
        qualified_name="alpha",
    )
    mod_b = _node(
        ProgramNodeKind.MODULE,
        "module:beta",
        component_id="module:beta",
        qualified_name="beta",
    )
    def_a = _node(
        ProgramNodeKind.DEFINITION,
        "def:alpha.helper",
        component_id="module:alpha",
        qualified_name="alpha.helper",
    )
    def_b = _node(
        ProgramNodeKind.DEFINITION,
        "def:beta.helper",
        component_id="module:beta",
        qualified_name="beta.helper",
    )
    caller = _node(
        ProgramNodeKind.MODULE,
        "module:gamma",
        component_id="module:gamma",
        qualified_name="gamma",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:gamma:helper",
        component_id="module:gamma",
        qualified_name="helper",
        record={"callee": "helper"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod_a, mod_b, def_a, def_b, caller, call),
        producer=PRODUCER,
    )
    res = resolve_program_calls(graph).resolutions_for_site(call.node_id)[0]
    assert res.status is ResolverStatus.AMBIGUOUS
    assert res.reason_code is ReasonCode.SAME_NAME_COLLISION
    assert set(res.targets) == {"alpha.helper", "beta.helper"}
    # Must not fabricate a single direct resolved edge.
    assert not res.is_direct_edge_allowed
    edges = resolve_program_calls(graph).resolution_edges(graph)
    # Materialized edges for ambiguous multi-target still keep non-terminal status
    # when targets exist as nodes; none may be resolved_static.
    for edge in edges:
        if edge.source == call.node_id:
            assert edge.binding.resolver_status is not ResolverStatus.RESOLVED_STATIC


def test_same_module_definition_resolves() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.local",
        component_id="module:pkg.local",
        qualified_name="pkg.local",
    )
    helper = _node(
        ProgramNodeKind.DEFINITION,
        "def:pkg.local.helper",
        component_id="module:pkg.local",
        qualified_name="pkg.local.helper",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:pkg.local:helper",
        component_id="module:pkg.local",
        qualified_name="helper",
        record={"callee": "helper"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=(mod, helper, call), producer=PRODUCER
    )
    res = resolve_program_calls(graph).resolutions_for_site(call.node_id)[0]
    assert res.status is ResolverStatus.RESOLVED_STATIC
    assert res.reason_code is ReasonCode.SAME_MODULE_DEFINITION
    assert res.targets == ("pkg.local.helper",)


# ---------------------------------------------------------------------------
# Known registrations, generated SDK, cross-package interfaces
# ---------------------------------------------------------------------------


def test_known_registration_is_candidate_not_static() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:svc",
        component_id="module:svc",
        qualified_name="svc",
    )
    impl = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:svc.real_entry",
        component_id="module:svc",
        qualified_name="svc.real_entry",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:svc:entry",
        component_id="module:svc",
        qualified_name="entry",
        record={"callee": "entry"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=(mod, impl, call), producer=PRODUCER
    )
    catalog = ResolverCatalog(known_registrations={"entry": "svc.real_entry"})
    res = resolve_program_calls(graph, catalog=catalog).resolutions_for_site(
        call.node_id
    )[0]
    assert res.status is ResolverStatus.CANDIDATE
    assert res.reason_code is ReasonCode.KNOWN_REGISTRATION
    assert res.targets == ("svc.real_entry",)
    assert not res.is_direct_edge_allowed


def test_generated_sdk_method_resolves() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:client",
        component_id="module:client",
        qualified_name="client",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:client:sdk_read",
        component_id="module:client",
        qualified_name="Client.read",
        record={"callee": "Client.read"},
    )
    target = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:sdk.Client.read",
        component_id="module:sdk",
        qualified_name="sdk.Client.read",
        path="generated/sdk.py",
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=(mod, call, target), producer=PRODUCER
    )
    catalog = ResolverCatalog(
        generated_sdk_methods={"Client.read": "sdk.Client.read"}
    )
    res = resolve_program_calls(graph, catalog=catalog).resolutions_for_site(
        call.node_id
    )[0]
    assert res.status is ResolverStatus.RESOLVED_STATIC
    assert res.reason_code is ReasonCode.GENERATED_SDK_METHOD


def test_generated_client_stays_candidate() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:ui",
        component_id="module:ui",
        qualified_name="ui",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:ui:gen_client",
        component_id="module:ui",
        qualified_name="GeneratedClient.invoke",
        record={"callee": "GeneratedClient.invoke", "generated_client": True},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, call), producer=PRODUCER)
    catalog = ResolverCatalog(
        generated_sdk_methods={"GeneratedClient.invoke": "remote.api.invoke"}
    )
    res = resolve_program_calls(graph, catalog=catalog).resolutions_for_site(
        call.node_id
    )[0]
    assert res.status is ResolverStatus.CANDIDATE
    assert res.reason_code is ReasonCode.GENERATED_CLIENT


def test_cross_package_interface_resolves() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:kit.facade",
        component_id="module:kit.facade",
        qualified_name="kit.facade",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:kit.facade:open",
        component_id="module:kit.facade",
        qualified_name="open_fs",
        record={"callee": "open_fs"},
    )
    impl = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:ipfs_kit_py.vfs.open_fs",
        component_id="module:ipfs_kit_py.vfs",
        qualified_name="ipfs_kit_py.vfs.open_fs",
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=(mod, call, impl), producer=PRODUCER
    )
    catalog = ResolverCatalog(
        cross_package_interfaces={"open_fs": "ipfs_kit_py.vfs.open_fs"}
    )
    res = resolve_program_calls(graph, catalog=catalog).resolutions_for_site(
        call.node_id
    )[0]
    assert res.status is ResolverStatus.RESOLVED_STATIC
    assert res.reason_code is ReasonCode.CROSS_PACKAGE_INTERFACE


# ---------------------------------------------------------------------------
# Dynamic / transport mechanisms require evidence and stay non-static
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("callee", "mechanism", "status", "reason"),
    [
        ("subprocess.run", "subprocess", ResolverStatus.EXTERNAL, ReasonCode.SUBPROCESS),
        ("requests.get", "http", ResolverStatus.EXTERNAL, ReasonCode.HTTP),
        ("grpc.unary", "rpc", ResolverStatus.EXTERNAL, ReasonCode.RPC),
        ("libp2p.dial", "libp2p", ResolverStatus.EXTERNAL, ReasonCode.LIBP2P),
        ("session.call_tool", "mcp", ResolverStatus.EXTERNAL, ReasonCode.MCP),
        (
            "importlib.import_module",
            "dynamic_import",
            ResolverStatus.CANDIDATE,
            ReasonCode.DYNAMIC_IMPORT,
        ),
        ("setattr", "monkey_patch", ResolverStatus.AMBIGUOUS, ReasonCode.MONKEY_PATCH),
    ],
)
def test_dynamic_mechanisms_are_typed_frontiers(
    callee: str,
    mechanism: str,
    status: ResolverStatus,
    reason: ReasonCode,
) -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:dyn",
        component_id="module:dyn",
        qualified_name="dyn",
    )
    call = _node(
        ProgramNodeKind.CALL,
        f"call:dyn:{callee}",
        component_id="module:dyn",
        qualified_name=callee,
        record={"callee": callee},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, call), producer=PRODUCER)
    res = resolve_program_calls(graph).resolutions_for_site(call.node_id)[0]
    assert res.mechanism == mechanism
    assert res.status is status
    assert res.reason_code is reason
    assert res.status is not ResolverStatus.RESOLVED_STATIC
    assert res.evidence
    assert res.is_frontier


def test_dependency_injection_and_callback_via_record_flags() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:di",
        component_id="module:di",
        qualified_name="di",
    )
    di_call = _node(
        ProgramNodeKind.CALL,
        "call:di:inject",
        component_id="module:di",
        qualified_name="container.resolve",
        record={"callee": "container.resolve", "dependency_injection": True},
    )
    cb_call = _node(
        ProgramNodeKind.CALL,
        "call:di:cb",
        component_id="module:di",
        qualified_name="on_ready",
        record={"callee": "on_ready", "callback": True},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=(mod, di_call, cb_call), producer=PRODUCER
    )
    result = resolve_program_calls(graph)
    di = result.resolutions_for_site(di_call.node_id)[0]
    cb = result.resolutions_for_site(cb_call.node_id)[0]
    assert di.reason_code is ReasonCode.DEPENDENCY_INJECTION
    assert di.status is ResolverStatus.AMBIGUOUS
    assert cb.reason_code is ReasonCode.CALLBACK
    assert cb.status is ResolverStatus.AMBIGUOUS
    assert di.evidence and cb.evidence


def test_mcp_known_registration_on_dynamic_site_is_candidate() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:mcp_client",
        component_id="module:mcp_client",
        qualified_name="mcp_client",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:mcp_client:tools_call",
        component_id="module:mcp_client",
        qualified_name="session.call_tool",
        record={"callee": "session.call_tool", "tool_name": "vfs_read"},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, call), producer=PRODUCER)
    catalog = ResolverCatalog(
        known_registrations={"session.call_tool": "ipfs_kit_py.mcp.vfs_read"}
    )
    res = resolve_program_calls(graph, catalog=catalog).resolutions_for_site(
        call.node_id
    )[0]
    assert res.mechanism == "mcp"
    assert res.status is ResolverStatus.CANDIDATE
    assert res.reason_code is ReasonCode.MCP
    assert res.targets == ("ipfs_kit_py.mcp.vfs_read",)


# ---------------------------------------------------------------------------
# Evidence, confidence, identity, graph application
# ---------------------------------------------------------------------------


def test_confidence_is_deterministic_and_status_bounded() -> None:
    assert confidence_for(ResolverStatus.RESOLVED_STATIC, ReasonCode.PACKAGE_IMPORT) == 100
    assert confidence_for(ResolverStatus.CANDIDATE, ReasonCode.KNOWN_REGISTRATION) == 50
    assert confidence_for(ResolverStatus.AMBIGUOUS, ReasonCode.SAME_NAME_COLLISION) == 25
    assert confidence_for(ResolverStatus.EXTERNAL, ReasonCode.SUBPROCESS) == 40
    # Reason cannot raise confidence above status.
    assert confidence_for(ResolverStatus.CANDIDATE, ReasonCode.PACKAGE_IMPORT) == 50
    assert confidence_for(ResolverStatus.UNRESOLVED, ReasonCode.NO_TARGET) == 0


def test_missing_evidence_is_rejected() -> None:
    with pytest.raises(MissingEvidenceError):
        CallResolution(
            site_id="call:x",
            site_kind="call",
            status=ResolverStatus.CANDIDATE,
            reason_code=ReasonCode.DYNAMIC_IMPORT,
            confidence=confidence_for(
                ResolverStatus.CANDIDATE, ReasonCode.DYNAMIC_IMPORT
            ),
            targets=("mod",),
            evidence=(),
        )


def test_dynamic_mechanism_cannot_be_resolved_static() -> None:
    with pytest.raises(CallResolverError):
        CallResolution(
            site_id="call:x",
            site_kind="call",
            status=ResolverStatus.RESOLVED_STATIC,
            reason_code=ReasonCode.SUBPROCESS,
            confidence=40,
            targets=("subprocess.run",),
            evidence=(_evidence(),),
            mechanism="subprocess",
        )


def test_wrong_confidence_is_rejected() -> None:
    with pytest.raises(CallResolverError):
        CallResolution(
            site_id="call:x",
            site_kind="call",
            status=ResolverStatus.RESOLVED_STATIC,
            reason_code=ReasonCode.PACKAGE_IMPORT,
            confidence=99,
            targets=("pkg.mod",),
            evidence=(_evidence(),),
        )


def test_resolution_identity_is_content_addressed() -> None:
    left = make_resolution(
        site_id="call:1",
        site_kind="call",
        status=ResolverStatus.CANDIDATE,
        reason_code=ReasonCode.DYNAMIC_IMPORT,
        targets=("x",),
        evidence=(_evidence(rule_id="rule:a"),),
    )
    right = make_resolution(
        site_id="call:1",
        site_kind="call",
        status=ResolverStatus.CANDIDATE,
        reason_code=ReasonCode.DYNAMIC_IMPORT,
        targets=("x",),
        evidence=(_evidence(rule_id="rule:a"),),
    )
    assert left.resolution_id == right.resolution_id
    assert left.to_dict() == right.to_dict()


def test_apply_to_graph_preserves_existing_nodes_and_adds_resolution_edges() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.core",
        component_id="module:pkg.core",
        qualified_name="pkg.core",
    )
    helper = _node(
        ProgramNodeKind.DEFINITION,
        "def:pkg.core.work",
        component_id="module:pkg.core",
        qualified_name="pkg.core.work",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:pkg.core:work",
        component_id="module:pkg.core",
        qualified_name="work",
        record={"callee": "work"},
    )
    original_edge = _edge(
        mod.node_id,
        helper.node_id,
        ProgramEdgeKind.DEFINES,
        component_id="module:pkg.core",
        resolver_status=ResolverStatus.RESOLVED_STATIC,
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod, helper, call),
        edges=(original_edge,),
        producer=PRODUCER,
    )
    result = resolve_program_calls(graph)
    applied = result.apply_to_graph(graph)
    assert {node.node_id for node in applied.nodes} == {
        node.node_id for node in graph.nodes
    }
    assert original_edge.edge_id in {edge.edge_id for edge in applied.edges}
    assert len(applied.edges) > len(graph.edges)
    # Existing edge payloads are unchanged (no AST mutation).
    kept = next(edge for edge in applied.edges if edge.edge_id == original_edge.edge_id)
    assert kept.to_dict() == original_edge.to_dict()
    new_edges = [edge for edge in applied.edges if edge.edge_id != original_edge.edge_id]
    assert any(
        edge.source == call.node_id
        and edge.target == helper.node_id
        and edge.kind is ProgramEdgeKind.CALLS
        and edge.binding.resolver_status is ResolverStatus.RESOLVED_STATIC
        for edge in new_edges
    )


def test_never_manufactures_edge_for_missing_target_node() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:solo",
        component_id="module:solo",
        qualified_name="solo",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:solo:missing",
        component_id="module:solo",
        qualified_name="ghost",
        record={"callee": "totally.missing.fn"},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, call), producer=PRODUCER)
    result = resolve_program_calls(graph)
    res = result.resolutions_for_site(call.node_id)[0]
    assert res.status in {
        ResolverStatus.UNRESOLVED,
        ResolverStatus.CANDIDATE,
        ResolverStatus.UNKNOWN,
    }
    edges = result.resolution_edges(graph)
    assert not any(edge.source == call.node_id for edge in edges)


def test_alias_call_via_import_binding() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:app.use",
        component_id="module:app.use",
        qualified_name="app.use",
    )
    util = _node(
        ProgramNodeKind.MODULE,
        "module:util",
        component_id="module:util",
        qualified_name="util",
    )
    fn = _node(
        ProgramNodeKind.DEFINITION,
        "def:util.compute",
        component_id="module:util",
        qualified_name="util.compute",
    )
    imp = _node(
        ProgramNodeKind.IMPORT,
        "import:app.use:util",
        component_id="module:app.use",
        qualified_name="u",
        record={"target": "util", "alias": "u", "local_name": "u"},
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:app.use:u.compute",
        component_id="module:app.use",
        qualified_name="u.compute",
        record={"callee": "u.compute"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod, util, fn, imp, call),
        edges=(
            _edge(mod.node_id, imp.node_id, ProgramEdgeKind.IMPORTS, component_id="module:app.use"),
        ),
        producer=PRODUCER,
    )
    result = resolve_program_calls(graph)
    call_res = result.resolutions_for_site(call.node_id)[0]
    assert call_res.status is ResolverStatus.RESOLVED_STATIC
    assert call_res.reason_code is ReasonCode.ALIAS_BINDING
    assert call_res.targets == ("util.compute",)


def test_result_frontier_and_stats_are_deterministic() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:stats",
        component_id="module:stats",
        qualified_name="stats",
    )
    call_a = _node(
        ProgramNodeKind.CALL,
        "call:stats:sub",
        component_id="module:stats",
        qualified_name="subprocess.run",
        record={"callee": "subprocess.run"},
    )
    call_b = _node(
        ProgramNodeKind.CALL,
        "call:stats:http",
        component_id="module:stats",
        qualified_name="requests.get",
        record={"callee": "requests.get"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=(mod, call_a, call_b), producer=PRODUCER
    )
    left = resolve_program_calls(graph)
    right = resolve_program_calls(graph)
    assert left.result_id == right.result_id
    assert left.to_dict() == right.to_dict()
    assert left.resolver_version == RESOLVER_VERSION
    frontier = left.frontier()
    assert len(frontier) == 2
    assert all(item.resolver_status.frontier for item in frontier)
    stats = left.stats()
    assert stats["resolution_count"] == 2
    assert stats["frontier_count"] == 2
    assert stats["direct_edge_count"] == 0
    assert stats["by_mechanism"]["subprocess"] == 1
    assert stats["by_mechanism"]["http"] == 1


def test_external_package_catalog() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:app.ext",
        component_id="module:app.ext",
        qualified_name="app.ext",
    )
    imp = _node(
        ProgramNodeKind.IMPORT,
        "import:app.ext:os",
        component_id="module:app.ext",
        qualified_name="os",
        record={"target": "os.path"},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, imp), producer=PRODUCER)
    catalog = ResolverCatalog(external_packages=frozenset({"os"}))
    res = resolve_program_calls(graph, catalog=catalog).resolutions_for_site(imp.node_id)[0]
    assert res.status is ResolverStatus.EXTERNAL
    assert res.reason_code is ReasonCode.EXTERNAL_MODULE


def test_resolver_consumes_graph_without_mutation() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:imm",
        component_id="module:imm",
        qualified_name="imm",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:imm:f",
        component_id="module:imm",
        qualified_name="f",
        record={"callee": "f"},
    )
    graph = build_program_graph(forest_id=FOREST_ID, nodes=(mod, call), producer=PRODUCER)
    before = graph.to_dict()
    ProgramCallResolver(graph).resolve()
    assert graph.to_dict() == before


def test_same_name_collision_requires_two_targets() -> None:
    with pytest.raises(CallResolverError):
        make_resolution(
            site_id="call:x",
            site_kind="call",
            status=ResolverStatus.AMBIGUOUS,
            reason_code=ReasonCode.SAME_NAME_COLLISION,
            targets=("only.one",),
            evidence=(_evidence(),),
        )


def test_manufactured_edge_guard_on_direct_flag() -> None:
    res = make_resolution(
        site_id="call:x",
        site_kind="call",
        status=ResolverStatus.CANDIDATE,
        reason_code=ReasonCode.KNOWN_REGISTRATION,
        targets=("svc.impl",),
        evidence=(_evidence(),),
        edge_kind=ProgramEdgeKind.CALLS,
        mechanism="registration",
    )
    assert not res.is_direct_edge_allowed


def test_batch_resolve_includes_imports_reexports_and_calls() -> None:
    mod = _node(
        ProgramNodeKind.MODULE,
        "module:batch",
        component_id="module:batch",
        qualified_name="batch",
    )
    other = _node(
        ProgramNodeKind.MODULE,
        "module:other",
        component_id="module:other",
        qualified_name="other",
    )
    imp = _node(
        ProgramNodeKind.IMPORT,
        "import:batch:other",
        component_id="module:batch",
        qualified_name="other",
        record={"target": "other"},
    )
    exp = _node(
        ProgramNodeKind.EXPORT,
        "export:batch.x",
        component_id="module:batch",
        qualified_name="batch.x",
        record={
            "kind": "re_export",
            "from_module": "other",
            "export_name": "x",
            "re_export": True,
        },
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:batch:http",
        component_id="module:batch",
        qualified_name="requests.get",
        record={"callee": "requests.get"},
    )
    graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=(mod, other, imp, exp, call),
        producer=PRODUCER,
    )
    result = resolve_program_calls(graph)
    kinds = {item.site_kind for item in result.resolutions}
    assert "import" in kinds
    assert "reexport" in kinds or "export" in kinds
    assert "call" in kinds
    assert result.source_graph_id == graph.graph_id
