from ipfs_accelerate_py.agent_supervisor.analysis.runtime_service_identity import (
    RuntimeServiceObservation,
    ServiceIdentityStatus,
    observe_runtime_service,
)


def obs(**overrides):
    value = dict(
        role="accelerate",
        interpreter="/py",
        module_origin="/checkout/mod.py",
        module_digest="sha256:m",
        checkout_commit="c",
        checkout_tree="t",
        overlay_id="o",
        argv=("/py",),
        environment={"MODE": "safe"},
        config_cid="cfg",
        state_cid="state",
        transport="mcp",
        endpoint="http://127.0.0.1:9000",
        pid=1,
        started_at="now",
        process_identity="pid:1:start",
        observed_port=9000,
    )
    value.update(overrides)
    return RuntimeServiceObservation(**value)


def test_observation_requires_a_matching_live_witness_and_rejects_drift_or_forgery():
    kwargs = dict(
        expected_role="accelerate",
        expected_interpreter="/py",
        expected_module_digest="sha256:m",
        expected_commit="c",
        expected_tree="t",
        expected_config_cid="cfg",
        expected_state_cid="state",
        expected_transport="mcp",
        expected_port=9000,
        environment_allowlist=frozenset({"MODE"}),
    )
    assert (
        observe_runtime_service(obs(), **kwargs).status is ServiceIdentityStatus.INTEGRATION_PENDING
    )
    assert (
        observe_runtime_service(
            obs(endpoint="http://remote:9000"), identity_vectors_ready=True, **kwargs
        ).status
        is ServiceIdentityStatus.INVALID
    )
    claim = observe_runtime_service(obs(), **kwargs).process_witness_cid
    assert (
        observe_runtime_service(
            obs(),
            claimed_process_witness_cid=claim,
            identity_vectors_ready=True,
            **kwargs,
        ).status
        is ServiceIdentityStatus.VALID
    )
    assert (
        observe_runtime_service(
            obs(),
            claimed_process_witness_cid="bafyforged-process-witness",
            identity_vectors_ready=True,
            **kwargs,
        ).status
        is ServiceIdentityStatus.INVALID
    )
    assert (
        observe_runtime_service(
            obs(started_at="reused"),
            claimed_process_witness_cid=claim,
            identity_vectors_ready=True,
            **kwargs,
        ).status
        is ServiceIdentityStatus.INVALID
    )


def test_loopback_endpoint_rejects_userinfo_and_hostname_tricks():
    kwargs = dict(
        expected_role="accelerate",
        expected_interpreter="/py",
        expected_module_digest="sha256:m",
        expected_commit="c",
        expected_tree="t",
        expected_config_cid="cfg",
        expected_state_cid="state",
        expected_transport="mcp",
        expected_port=9000,
        environment_allowlist=frozenset({"MODE"}),
    )
    for endpoint in (
        "http://127.0.0.1:9000@remote.invalid",
        "http://operator@127.0.0.1:9000",
        "http://localhost:9000?relay=remote",
    ):
        result = observe_runtime_service(obs(endpoint=endpoint), **kwargs)
        assert result.status is ServiceIdentityStatus.INVALID
        assert "remote_endpoint" in result.reasons
