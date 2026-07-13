"""Profile G fleet transport contract tests."""

from ipfs_accelerate_py.mcp_server.fastapi_service import _profile_g_rest_binding
from ipfs_accelerate_py.mcp_server.mcplusplus.profile_g_transport import (
    PROFILE_G_METHODS,
    ProfileGDispatcher,
    ProfileGTransportError,
    jsonrpc_error,
)
from ipfs_accelerate_py.mcp_server.server import get_unified_supported_profiles


def test_all_profile_g_methods_are_advertised_and_dispatchable():
    calls = []
    dispatcher = ProfileGDispatcher(lambda method, params: calls.append((method, params)) or {"ok": True})
    for method in PROFILE_G_METHODS:
        assert dispatcher.dispatch(method, {}) == {"ok": True}
    assert [call[0] for call in calls] == list(PROFILE_G_METHODS)
    assert "mcp++/risk-scheduling" in get_unified_supported_profiles()


def test_profile_g_rest_path_parameters_are_authoritative():
    assert _profile_g_rest_binding("GET", "/mcp/tasks/ready") == ("mcp++/tasks/ready", {})
    assert _profile_g_rest_binding("GET", "/mcp/tasks/bafy-test") == (
        "mcp++/tasks/get", {"task_cid": "bafy-test"}
    )
    assert _profile_g_rest_binding("POST", "/mcp/schedule/claims/bafy-claim/renew") == (
        "mcp++/schedule/renew", {"claim_cid": "bafy-claim"}
    )


def test_profile_g_errors_preserve_stable_wire_code():
    def fail(_method, _params):
        error = RuntimeError("stale token")
        error.code = "G_CLAIM_CONFLICT"
        raise error

    try:
        ProfileGDispatcher(fail).dispatch("mcp++/schedule/renew", {})
    except ProfileGTransportError as error:
        response = jsonrpc_error(7, error)
    assert response["error"]["code"] == -32046
    assert response["error"]["data"]["code"] == "G_CLAIM_CONFLICT"

