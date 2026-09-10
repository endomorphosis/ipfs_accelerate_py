"""Pins and sealed descriptors cross the router's compatibility API boundary."""

import os
import sys

import pytest

from ipfs_accelerate_py import agent_implementation_route as route
from ipfs_accelerate_py import llm_router as router
from test.api.test_agent_supervisor_native_dependency_pin import (
    AUTHORIZATION_ID,
    _FakeExtensionLoader,
    _install_fake_loader,
    _write_source,
)


@pytest.mark.parametrize("inspector", [route, router], ids=["route", "router"])
@pytest.mark.parametrize("sealer", [route, router], ids=["route", "router"])
def test_pin_from_either_api_can_seal_and_roundtrip_through_both(
    tmp_path, inspector, sealer,
):
    source = _write_source(tmp_path)
    pin = inspector.inspect_agent_supervisor_native_dependency_source(
        source, distribution_version="1.5.5", engine_version="v1.5.5",
    )
    launch = sealer.seal_agent_supervisor_native_dependency(
        source, expected_pin=pin, accepted_authorization_id=AUTHORIZATION_ID,
    )
    try:
        for api in (route, router):
            assert api.parse_agent_supervisor_native_dependency_pin(pin.as_dict()) == pin
            parsed = api.parse_agent_supervisor_native_dependency_launch(launch.as_dict())
            assert parsed == launch
            assert api.verify_agent_supervisor_native_dependency_sealed_fd(parsed) == (
                f"/proc/self/fd/{launch.descriptor.descriptor}"
            )
    finally:
        os.close(launch.descriptor.descriptor)


@pytest.mark.parametrize("sealer", [route, router], ids=["route", "router"])
def test_cross_api_pin_does_not_admit_changed_native_bytes(tmp_path, sealer):
    source = _write_source(tmp_path)
    inspector = router if sealer is route else route
    pin = inspector.inspect_agent_supervisor_native_dependency_source(
        source, distribution_version="1.5.5", engine_version="v1.5.5",
    )
    raw = bytearray(source.read_bytes())
    raw[-1] ^= 1
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="does not match the accepted pin"):
        sealer.seal_agent_supervisor_native_dependency(
            source, expected_pin=pin, accepted_authorization_id=AUTHORIZATION_ID,
        )


@pytest.mark.parametrize("loader_api", [route, router], ids=["route", "router"])
def test_preload_shares_active_launch_and_single_initialization_state(
    tmp_path, monkeypatch, loader_api,
):
    _install_fake_loader(monkeypatch)
    source = _write_source(tmp_path)
    pin = router.inspect_agent_supervisor_native_dependency_source(
        source, distribution_version="9.9.9", engine_version="v9.9.9",
    )
    launch = router.seal_agent_supervisor_native_dependency(
        source, expected_pin=pin, accepted_authorization_id=AUTHORIZATION_ID,
    )
    try:
        module = loader_api.preload_agent_supervisor_native_dependency(launch)
        for api in (route, router):
            assert api.active_agent_supervisor_native_dependency_launch() == launch
        assert sys.modules["duckdb"] is sys.modules["_duckdb"] is module
        calls = list(_FakeExtensionLoader.calls)
        # The loader is a Python double; removing aliases must not erase the
        # shared irreversible initialization barrier for either public API.
        sys.modules.pop("duckdb")
        sys.modules.pop("_duckdb")
        for api in (route, router):
            with pytest.raises(ValueError, match="preload process is terminal"):
                api.preload_agent_supervisor_native_dependency(launch)
        assert _FakeExtensionLoader.calls == calls
    finally:
        sys.modules.pop("duckdb", None)
        sys.modules.pop("_duckdb", None)
        os.close(launch.descriptor.descriptor)
