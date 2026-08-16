#!/usr/bin/env python3
"""Runtime tests for dual MCP bindings (RuntimeBindingAdapter@1).

Task: MCPP-023
Acceptance:
  - Runtime tests cover legacy client, current client, and dual peer.
  - No silent initialize on the current path.
  - Accelerate and datasets advertise only bindings they implement and
    reject the others fail-closed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root / package roots are importable when pytest is launched
# from the monorepo workspace root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ACCEL_ROOT = _REPO_ROOT / "ipfs_accelerate_py"
_DATASETS_PKG_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for _p in (_REPO_ROOT, _ACCEL_ROOT, _DATASETS_PKG_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ipfs_accelerate_py.mcp_server.mcplusplus import bindings as accel_bindings
from ipfs_accelerate_py.mcp_server.mcplusplus.bindings import (
    CURRENT_BINDING_ID,
    CURRENT_PROTOCOL_VERSION,
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    ERR_NOT_INITIALIZED,
    ERR_UNSUPPORTED_PROTOCOL_VERSION,
    INTERFACE_LABEL,
    LEGACY_BINDING_ID,
    LEGACY_PROTOCOL_VERSION,
    META_BINDING_ID,
    META_PROTOCOL_VERSION,
    PeerMode,
    REASON_BINDING_MISMATCH,
    REASON_BINDING_NOT_OFFERED,
    REASON_FORGED_VERSION,
    REASON_INIT_AS_CURRENT,
    REASON_SILENT_DOWNGRADE,
    REASON_VERSION_BINDING_MISMATCH,
    RuntimeBindingAdapter,
    create_runtime_binding_adapter,
    current_request_meta,
    legacy_initialize_params,
    make_current_request,
    make_legacy_request,
    open_legacy_session,
)

datasets_bindings = pytest.importorskip(
    "ipfs_datasets_py.mcp_server.mcplusplus.bindings",
    reason="datasets mcplusplus bindings not importable",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def accelerate_dual() -> RuntimeBindingAdapter:
    return create_runtime_binding_adapter(
        mode=PeerMode.DUAL,
        runtime="accelerate",
        server_name="ipfs-accelerate-mcp++",
    )


@pytest.fixture
def accelerate_legacy() -> RuntimeBindingAdapter:
    return create_runtime_binding_adapter(
        mode=PeerMode.LEGACY_ONLY,
        runtime="accelerate",
    )


@pytest.fixture
def accelerate_current() -> RuntimeBindingAdapter:
    return create_runtime_binding_adapter(
        mode=PeerMode.CURRENT_ONLY,
        runtime="accelerate",
    )


@pytest.fixture
def datasets_dual() -> RuntimeBindingAdapter:
    return datasets_bindings.create_runtime_binding_adapter(
        mode=PeerMode.DUAL,
        runtime="datasets",
        server_name="ipfs-datasets-mcp++",
    )


@pytest.fixture
def datasets_current() -> RuntimeBindingAdapter:
    return datasets_bindings.create_runtime_binding_adapter(
        mode=PeerMode.CURRENT_ONLY,
        runtime="datasets",
    )


# ---------------------------------------------------------------------------
# Interface / advertisement
# ---------------------------------------------------------------------------


class TestInterfaceAndAdvertisement:
    def test_interface_label_exported(self):
        assert INTERFACE_LABEL == "RuntimeBindingAdapter@1"
        assert datasets_bindings.INTERFACE_LABEL == INTERFACE_LABEL

    def test_accelerate_dual_advertises_both(self, accelerate_dual: RuntimeBindingAdapter):
        ad = accelerate_dual.advertisement()
        assert ad["interface"] == INTERFACE_LABEL
        assert ad["runtime"] == "accelerate"
        assert ad["mode"] == "dual"
        assert LEGACY_BINDING_ID in ad["supportedBindings"]
        assert CURRENT_BINDING_ID in ad["supportedBindings"]
        assert LEGACY_PROTOCOL_VERSION in ad["supportedVersions"]
        assert CURRENT_PROTOCOL_VERSION in ad["supportedVersions"]
        assert ad["capabilities"]["mcp++"]["bindingIds"] == ad["supportedBindings"]

    def test_accelerate_legacy_only_does_not_claim_current(
        self, accelerate_legacy: RuntimeBindingAdapter
    ):
        ad = accelerate_legacy.advertisement()
        assert ad["supportedBindings"] == [LEGACY_BINDING_ID]
        assert CURRENT_BINDING_ID not in ad["supportedBindings"]
        assert accelerate_legacy.implements(LEGACY_BINDING_ID)
        assert not accelerate_legacy.implements(CURRENT_BINDING_ID)

    def test_accelerate_current_only_does_not_claim_legacy(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        ad = accelerate_current.advertisement()
        assert ad["supportedBindings"] == [CURRENT_BINDING_ID]
        assert LEGACY_BINDING_ID not in ad["supportedBindings"]

    def test_datasets_dual_advertises_datasets_runtime(
        self, datasets_dual: RuntimeBindingAdapter
    ):
        ad = datasets_dual.advertisement()
        assert ad["runtime"] == "datasets"
        assert ad["serverInfo"]["name"] == "ipfs-datasets-mcp++"
        assert set(ad["supportedBindings"]) == {
            LEGACY_BINDING_ID,
            CURRENT_BINDING_ID,
        }


# ---------------------------------------------------------------------------
# Legacy client
# ---------------------------------------------------------------------------


class TestLegacyClient:
    def test_legacy_client_initialize_and_tools(
        self, accelerate_legacy: RuntimeBindingAdapter
    ):
        init = open_legacy_session(accelerate_legacy)
        assert init.ok, init.error
        assert init.path == "legacy"
        assert init.result is not None
        assert init.result["protocolVersion"] == LEGACY_PROTOCOL_VERSION
        assert (
            init.result["capabilities"]["mcp++"]["bindingId"] == LEGACY_BINDING_ID
        )
        assert accelerate_legacy.initialize_calls == 1
        assert accelerate_legacy.phase.name == "READY"

        listed = accelerate_legacy.handle(
            make_legacy_request("tools/list", req_id=2)
        )
        assert listed.ok, listed.error
        assert listed.path == "legacy"
        names = {t["name"] for t in listed.result["tools"]}
        assert "echo" in names
        assert listed.result["bindingId"] == LEGACY_BINDING_ID

        called = accelerate_legacy.handle(
            make_legacy_request(
                "tools/call",
                req_id=3,
                params={"name": "echo", "arguments": {"text": "legacy-ok"}},
            )
        )
        assert called.ok, called.error
        assert called.result["content"][0]["text"] == "legacy-ok"
        assert called.result["bindingId"] == LEGACY_BINDING_ID

    def test_legacy_client_before_initialize_rejected(
        self, accelerate_legacy: RuntimeBindingAdapter
    ):
        resp = accelerate_legacy.handle(make_legacy_request("tools/list", req_id=1))
        assert not resp.ok
        assert resp.error["code"] == ERR_NOT_INITIALIZED
        assert resp.error["data"]["reason"] == "not_initialized"

    def test_legacy_client_on_dual_peer_advertises_both(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        init = open_legacy_session(accelerate_dual)
        assert init.ok, init.error
        assert LEGACY_BINDING_ID in init.result["supportedBindings"]
        assert CURRENT_BINDING_ID in init.result["supportedBindings"]
        listed = accelerate_dual.handle(make_legacy_request("tools/list", req_id=5))
        assert listed.ok
        assert listed.path == "legacy"

    def test_datasets_legacy_client(self, datasets_dual: RuntimeBindingAdapter):
        init = open_legacy_session(datasets_dual)
        assert init.ok, init.error
        assert init.result["serverInfo"]["name"] == "ipfs-datasets-mcp++"
        listed = datasets_dual.handle(make_legacy_request("tools/list", req_id=2))
        assert listed.ok
        assert listed.result["bindingId"] == LEGACY_BINDING_ID


# ---------------------------------------------------------------------------
# Current client (no initialize)
# ---------------------------------------------------------------------------


class TestCurrentClient:
    def test_current_client_tools_without_initialize(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        resp = accelerate_current.handle(
            make_current_request("tools/list", req_id=10, meta=meta)
        )
        assert resp.ok, resp.error
        assert resp.path == "current"
        assert accelerate_current.initialize_calls == 0
        assert resp.result["_meta"][META_BINDING_ID] == CURRENT_BINDING_ID
        names = {t["name"] for t in resp.result["tools"]}
        assert "echo" in names

        call = accelerate_current.handle(
            make_current_request(
                "tools/call",
                req_id=11,
                params={"name": "echo", "arguments": {"text": "current-ok"}},
                meta=meta,
            )
        )
        assert call.ok, call.error
        assert call.result["content"][0]["text"] == "current-ok"
        assert accelerate_current.initialize_calls == 0

    def test_current_client_discover_without_initialize(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        resp = accelerate_current.handle(
            make_current_request("server/discover", req_id=12, meta=meta)
        )
        assert resp.ok, resp.error
        assert resp.result["supportedVersions"] == [CURRENT_PROTOCOL_VERSION]
        assert (
            resp.result["capabilities"]["mcp++"]["bindingId"] == CURRENT_BINDING_ID
        )
        assert accelerate_current.initialize_calls == 0

    def test_current_client_on_dual_peer(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        resp = accelerate_dual.handle(
            make_current_request("tools/list", req_id=20, meta=meta)
        )
        assert resp.ok, resp.error
        assert resp.path == "current"
        assert accelerate_dual.initialize_calls == 0
        assert accelerate_dual.active_binding == CURRENT_BINDING_ID

    def test_datasets_current_client(
        self, datasets_current: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        resp = datasets_current.handle(
            make_current_request("tools/list", req_id=10, meta=meta)
        )
        assert resp.ok, resp.error
        assert datasets_current.initialize_calls == 0
        assert resp.result["_meta"][META_BINDING_ID] == CURRENT_BINDING_ID


# ---------------------------------------------------------------------------
# No silent initialize on the current path
# ---------------------------------------------------------------------------


class TestNoSilentInitializeOnCurrentPath:
    def test_current_only_rejects_initialize(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        resp = accelerate_current.handle(
            make_legacy_request(
                "initialize",
                req_id=1,
                params=legacy_initialize_params(),
            )
        )
        assert not resp.ok
        assert resp.error["code"] == ERR_METHOD_NOT_FOUND
        assert resp.error["data"]["reason"] == REASON_INIT_AS_CURRENT
        assert resp.error["data"]["bindingId"] == CURRENT_BINDING_ID
        assert accelerate_current.initialize_calls == 1
        assert accelerate_current.phase.name == "UNINITIALIZED"
        assert accelerate_current.active_binding is None

    def test_current_only_rejects_initialized_notification(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        resp = accelerate_current.handle(
            make_legacy_request(
                "notifications/initialized",
                notification=True,
                params={},
            )
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_INIT_AS_CURRENT

    def test_current_path_never_auto_initializes(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        """tools/list on current path must not open a legacy session."""
        meta = current_request_meta()
        resp = accelerate_current.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        )
        assert resp.ok
        assert accelerate_current.initialize_calls == 0
        assert accelerate_current.phase.name == "UNINITIALIZED"

    def test_modern_version_on_initialize_not_promoted_to_current(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        resp = accelerate_dual.handle(
            make_legacy_request(
                "initialize",
                req_id=1,
                params=legacy_initialize_params(
                    protocol_version=CURRENT_PROTOCOL_VERSION
                ),
            )
        )
        assert not resp.ok
        assert resp.error["code"] == ERR_UNSUPPORTED_PROTOCOL_VERSION
        assert resp.error["data"]["reason"] == REASON_VERSION_BINDING_MISMATCH
        assert accelerate_dual.active_binding is None
        assert accelerate_dual.phase.name == "UNINITIALIZED"

    def test_datasets_current_only_rejects_initialize(
        self, datasets_current: RuntimeBindingAdapter
    ):
        resp = datasets_current.handle(
            make_legacy_request(
                "initialize",
                req_id=1,
                params=legacy_initialize_params(),
            )
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_INIT_AS_CURRENT
        assert datasets_current.initialize_calls == 1


# ---------------------------------------------------------------------------
# Dual peer
# ---------------------------------------------------------------------------


class TestDualPeer:
    def test_dual_serves_legacy_and_current_clients(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        # Legacy client on dual peer
        init = open_legacy_session(accelerate_dual)
        assert init.ok, init.error
        legacy_list = accelerate_dual.handle(
            make_legacy_request("tools/list", req_id=2)
        )
        assert legacy_list.ok
        assert legacy_list.path == "legacy"

        # Explicit upgrade / independent current path after legacy
        meta = current_request_meta()
        current_list = accelerate_dual.handle(
            make_current_request("tools/list", req_id=3, meta=meta)
        )
        assert current_list.ok, current_list.error
        assert current_list.path == "current"
        assert accelerate_dual.active_binding == CURRENT_BINDING_ID

    def test_dual_discover_lists_both_bindings(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        resp = accelerate_dual.handle(
            make_current_request("server/discover", req_id=1, meta=meta)
        )
        assert resp.ok, resp.error
        assert set(resp.result["supportedBindings"]) == {
            LEGACY_BINDING_ID,
            CURRENT_BINDING_ID,
        }
        assert CURRENT_PROTOCOL_VERSION in resp.result["supportedVersions"]
        assert LEGACY_PROTOCOL_VERSION in resp.result["supportedVersions"]

    def test_dual_independent_fresh_legacy_after_reset(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        assert accelerate_dual.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        ).ok
        accelerate_dual.reset()
        init = open_legacy_session(accelerate_dual)
        assert init.ok, init.error
        assert accelerate_dual.active_binding == LEGACY_BINDING_ID

    def test_datasets_dual_peer(self, datasets_dual: RuntimeBindingAdapter):
        init = open_legacy_session(datasets_dual)
        assert init.ok
        meta = current_request_meta()
        # Fresh dual connection for current client semantics: reset active path
        datasets_dual.reset()
        current = datasets_dual.handle(
            make_current_request("tools/list", req_id=10, meta=meta)
        )
        assert current.ok, current.error
        assert current.path == "current"
        assert datasets_dual.initialize_calls == 0


# ---------------------------------------------------------------------------
# Fail-closed: forgery, silent downgrade, unoffered bindings
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_silent_downgrade_after_current_rejected(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        assert accelerate_dual.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        ).ok
        assert accelerate_dual.active_binding == CURRENT_BINDING_ID

        resp = accelerate_dual.handle(
            make_legacy_request(
                "initialize",
                req_id=2,
                params=legacy_initialize_params(),
            )
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_SILENT_DOWNGRADE
        assert accelerate_dual.rejected_downgrades >= 1

    def test_legacy_only_rejects_current_meta(
        self, accelerate_legacy: RuntimeBindingAdapter
    ):
        meta = current_request_meta()
        resp = accelerate_legacy.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        )
        assert not resp.ok
        assert resp.error["code"] == ERR_UNSUPPORTED_PROTOCOL_VERSION
        assert resp.error["data"]["reason"] == REASON_BINDING_NOT_OFFERED

    def test_current_path_rejects_legacy_version_meta(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        meta = current_request_meta(protocol_version=LEGACY_PROTOCOL_VERSION)
        resp = accelerate_current.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_FORGED_VERSION

    def test_current_path_rejects_legacy_binding_id(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        meta = current_request_meta(binding_id=LEGACY_BINDING_ID)
        resp = accelerate_current.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_BINDING_MISMATCH

    def test_unknown_binding_id_rejected(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        meta = current_request_meta(binding_id="mcp-binding/unknown-future")
        resp = accelerate_dual.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_BINDING_NOT_OFFERED

    def test_legacy_client_current_binding_id_on_initialize(
        self, accelerate_dual: RuntimeBindingAdapter
    ):
        resp = accelerate_dual.handle(
            make_legacy_request(
                "initialize",
                req_id=1,
                params=legacy_initialize_params(binding_id=CURRENT_BINDING_ID),
            )
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_BINDING_MISMATCH

    def test_missing_current_meta_rejected_on_current_only(
        self, accelerate_current: RuntimeBindingAdapter
    ):
        resp = accelerate_current.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
        )
        assert not resp.ok
        assert resp.error["code"] == ERR_INVALID_PARAMS

    def test_datasets_rejects_unoffered_current_when_legacy_only(self):
        peer = datasets_bindings.create_runtime_binding_adapter(
            mode=PeerMode.LEGACY_ONLY,
            runtime="datasets",
        )
        meta = current_request_meta()
        resp = peer.handle(
            make_current_request("tools/list", req_id=1, meta=meta)
        )
        assert not resp.ok
        assert resp.error["data"]["reason"] == REASON_BINDING_NOT_OFFERED
        assert peer.advertisement()["supportedBindings"] == [LEGACY_BINDING_ID]


# ---------------------------------------------------------------------------
# Module surface parity accelerate ↔ datasets
# ---------------------------------------------------------------------------


class TestPackageSurface:
    def test_datasets_exports_match_accelerate(self):
        required = {
            "RuntimeBindingAdapter",
            "PeerMode",
            "create_runtime_binding_adapter",
            "LEGACY_BINDING_ID",
            "CURRENT_BINDING_ID",
            "INTERFACE_LABEL",
        }
        for name in required:
            assert hasattr(accel_bindings, name), name
            assert hasattr(datasets_bindings, name), name

    def test_datasets_factory_defaults(self):
        peer = datasets_bindings.create_datasets_binding_adapter()
        assert peer.runtime == "datasets"
        assert peer.server_name == "ipfs-datasets-mcp++"
        assert peer.mode is PeerMode.DUAL
