"""Synthetic, source-safe tests for the Leanstral topology collector."""

from copy import deepcopy
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import trio

from ipfs_accelerate_py.mcplusplus_module import (
    leanstral_topology_collector as topology_collector,
)
from ipfs_accelerate_py.mcplusplus_module.leanstral_topology import (
    IndependentDialObservation,
    InterfaceAddress,
    LEANSTRAL_P2P_LISTEN_ADDR,
    ProbeObservation,
    canonical_json_cid,
    normalize_served_model_record,
)
from ipfs_accelerate_py.mcplusplus_module.leanstral_topology_collector import (
    CLIENT_PROBE_SCHEMA,
    COLLECTOR_FAILURE_SCHEMA,
    LeanstralCollectorConfig,
    TopologyCollectionError,
    _run_independent_client_subprocess,
    _with_cid,
    assemble_topology_receipt,
    canonical_identity_json,
    collect_leanstral_topology,
    collector_failure_receipt,
)
from ipfs_accelerate_py.utils.llama_cpp import DEFAULT_LEANSTRAL_MODEL_REF


SERVER_PEER_ID = "12D3KooWLeanstralCollectorServer"
CLIENT_PEER_ID = "12D3KooWLeanstralCollectorClient"
SOURCE_COMMIT = "a" * 40
ENDPOINT = "http://127.0.0.1:8080/v1"
BOOTSTRAP = "/dnsaddr/bootstrap.libp2p.io/p2p/QmExactBootstrap"


def _config() -> LeanstralCollectorConfig:
    return LeanstralCollectorConfig(
        endpoint_url=ENDPOINT,
        allowed_interfaces=("wlP9s9", "tun0", "tun1"),
        bootstrap_peers=(BOOTSTRAP,),
        probe_timeout_s=2.0,
        model_timeout_s=1.0,
        client_process_timeout_s=9.0,
    )


def _interfaces():
    return (
        InterfaceAddress("wlP9s9", "172.30.4.2", True, "lan"),
        InterfaceAddress("tun0", "10.8.0.99", True, "lan"),
        InterfaceAddress("tun1", "10.10.0.14", True, "lan"),
        InterfaceAddress("docker0", "172.17.0.1", False, "container"),
        InterfaceAddress("lo", "127.0.0.1", True, "unrelated"),
    )


def _model():
    return normalize_served_model_record(
        transport_model_id="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4",
        endpoint=ENDPOINT,
        owned_by="llamacpp",
        metadata={"model_path": "/must/not/enter/identity"},
    )


def _multiaddrs():
    return [
        f"/ip4/{address}/tcp/19001/p2p/{SERVER_PEER_ID}"
        for address in ("10.10.0.14", "10.8.0.99", "172.30.4.2")
    ]


def _server_status():
    bootstrap_attempt = asdict(
        ProbeObservation(
            mechanism="bootstrap",
            target=BOOTSTRAP,
            attempted=True,
            success=True,
            timeout_s=2.0,
            duration_ms=11.0,
            observer_peer_id=SERVER_PEER_ID,
        )
    )
    return {
        "p2p_requested": True,
        "p2p_enabled": True,
        "peer_id": SERVER_PEER_ID,
        "multiaddrs": _multiaddrs(),
        "protocol": "/mcp+p2p/1.0.0",
        "listen_addrs": [LEANSTRAL_P2P_LISTEN_ADDR],
        "listen_port": 19001,
        "operational": True,
        "bootstrap": {
            "configured_peers": [BOOTSTRAP],
            "attempts": [bootstrap_attempt],
        },
        "rendezvous": {
            "service": {
                "mode": "same_as_service_peer",
                "configured": True,
                "implemented": True,
                "peer_id": SERVER_PEER_ID,
            },
            "namespace": "leanstral-local",
        },
        "capabilities": {
            "mcp_stream": {
                "configured": True,
                "implemented": True,
                "advertised": True,
            },
            "bootstrap": {
                "configured": True,
                "implemented": True,
                "advertised": True,
            },
            "rendezvous": {
                "configured": True,
                "implemented": True,
                "advertised": True,
            },
            "pubsub": {
                "configured": False,
                "implemented": False,
                "advertised": False,
                "policy": "disabled_until_implemented",
            },
            "floodsub": {
                "configured": False,
                "implemented": False,
                "advertised": False,
                "policy": "disabled_until_implemented",
            },
        },
    }


def _client_receipt(models=None):
    projected_models = []
    for model in models or [_model()]:
        projected = deepcopy(model)
        projected["metadata"] = {}
        projected_models.append(projected)
    target = _multiaddrs()[0]
    bootstrap = asdict(
        ProbeObservation(
            mechanism="bootstrap",
            target=target,
            attempted=True,
            success=True,
            timeout_s=2.0,
            duration_ms=12.0,
            observer_peer_id=CLIENT_PEER_ID,
        )
    )
    rendezvous = asdict(
        ProbeObservation(
            mechanism="rendezvous",
            target=target,
            attempted=True,
            success=True,
            timeout_s=2.0,
            duration_ms=18.0,
            observer_peer_id=CLIENT_PEER_ID,
            namespace="leanstral-local",
            details={
                "registered": True,
                "discovered_peers": [f"/ip4/127.0.0.1/tcp/32000/p2p/{CLIENT_PEER_ID}"],
            },
        )
    )
    dial = asdict(
        IndependentDialObservation(
            dialer_peer_id=CLIENT_PEER_ID,
            target_peer_id=SERVER_PEER_ID,
            target_multiaddr=target,
            attempted=True,
            success=True,
            timeout_s=2.0,
            duration_ms=9.0,
        )
    )
    identity = {
        "schema": CLIENT_PROBE_SCHEMA,
        "process_role": "independent_client_subprocess",
        "client_peer_id": CLIENT_PEER_ID,
        "target_multiaddr": target,
        "bootstrap_exercises": [bootstrap],
        "rendezvous_exercises": [rendezvous],
        "independent_dial": dial,
        "model_listing": {
            "status": "success",
            "count": len(projected_models),
            "models": projected_models,
        },
        "inference_attempted": False,
    }
    return _with_cid(identity, field_name="client_receipt_cid")


def _assemble(client=None, status=None, raw_models=None):
    return assemble_topology_receipt(
        config=_config(),
        interfaces=_interfaces(),
        server_status=status or _server_status(),
        raw_http_models=raw_models or [_model()],
        client_receipt=client or _client_receipt(),
        source_commit=SOURCE_COMMIT,
    )


def test_assemble_emits_valid_path_free_cidv1_receipt():
    from multiformats import CID

    receipt = _assemble()

    assert receipt["validation"] == {"valid": True, "errors": []}
    assert receipt["contract"]["p2p_port"] == 19001
    assert receipt["observation"]["http_port"] == 8080
    assert receipt["observation"]["server_instance_count"] == 1
    assert {probe["target"] for probe in receipt["observation"]["bootstrap_exercises"]} == {
        BOOTSTRAP,
        _multiaddrs()[0],
    }
    assert receipt["observation"]["served_models"][0]["id"] == "leanstral_local"
    assert receipt["observation"]["served_models"][0]["provider"] == "llamacpp"
    assert "Leanstral-1.5" in receipt["observation"]["served_models"][0]["transport_model_id"]
    assert receipt["observation"]["served_models"][0]["metadata"] == {}
    rendered = canonical_identity_json(receipt)
    assert "/must/not/enter/identity" not in rendered
    decoded = CID.decode(receipt["receipt_cid"])
    assert decoded.version == 1
    assert decoded.codec.name == "raw"
    assert decoded.hashfun.name == "sha2-256"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda value: value["bootstrap_exercises"][0].update(success=False),
            "bootstrap_exercise_incomplete",
        ),
        (
            lambda value: value["rendezvous_exercises"][0]["details"].update(discovered_peers=[]),
            "rendezvous_exercise_incomplete",
        ),
        (
            lambda value: value["independent_dial"].update(success=False),
            "independent_dial_incomplete",
        ),
        (
            lambda value: value["model_listing"].update(count=2),
            "mcp_model_listing_invalid",
        ),
    ),
)
def test_client_evidence_failures_are_rejected_even_with_recomputed_cid(
    mutation,
    expected,
):
    receipt = _client_receipt()
    identity = deepcopy(receipt)
    identity.pop("client_receipt_cid")
    mutation(identity)
    receipt = _with_cid(identity, field_name="client_receipt_cid")

    with pytest.raises(TopologyCollectionError, match=expected):
        _assemble(client=receipt)


def test_client_receipt_cid_tampering_fails_closed():
    receipt = _client_receipt()
    receipt["client_receipt_cid"] = canonical_json_cid({"different": True})

    with pytest.raises(
        TopologyCollectionError,
        match="independent_client_receipt_cid_mismatch",
    ):
        _assemble(client=receipt)


def test_pubsub_or_floodsub_overclaim_is_rejected():
    status = _server_status()
    status["capabilities"]["pubsub"]["implemented"] = True
    status["capabilities"]["pubsub"]["advertised"] = True

    with pytest.raises(TopologyCollectionError, match="topology_validation_failed"):
        _assemble(status=status)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda status: status["bootstrap"].update(attempts=[]),
            "configured_bootstrap_exercises_missing",
        ),
        (
            lambda status: status["bootstrap"]["attempts"][0].update(
                success=False,
                error="connect_failed",
            ),
            "configured_bootstrap_exercise_no_success",
        ),
        (
            lambda status: status["rendezvous"].update(namespace="other"),
            "rendezvous_namespace_mismatch",
        ),
    ),
)
def test_configured_bootstrap_and_rendezvous_evidence_is_required(
    mutation,
    expected,
):
    status = _server_status()
    mutation(status)

    with pytest.raises(TopologyCollectionError, match=expected):
        _assemble(status=status)


def test_p2p_listing_must_match_independent_http_model_manager_probe():
    client = _client_receipt()
    identity = deepcopy(client)
    identity.pop("client_receipt_cid")
    identity["model_listing"]["models"][0]["transport_model_id"] = "Other/Leanstral-transport"
    client = _with_cid(identity, field_name="client_receipt_cid")

    with pytest.raises(
        TopologyCollectionError,
        match="served_model_transport_identity_mismatch",
    ):
        _assemble(client=client)


@pytest.mark.parametrize(
    "local_path",
    ("/srv/models/Leanstral.gguf", r"C:\models\Leanstral.gguf"),
)
def test_absolute_transport_model_path_is_never_emitted(local_path):
    model = _model()
    model["transport_model_id"] = local_path

    with pytest.raises(
        TopologyCollectionError,
        match="served_model_identity_contains_local_path",
    ):
        _assemble(raw_models=[model])


def test_failure_output_is_canonical_path_free_and_cid_bound():
    receipt = collector_failure_receipt("source_tree_dirty")
    rendered = canonical_identity_json(receipt)

    assert json.loads(rendered) == receipt
    assert receipt["schema"] == COLLECTOR_FAILURE_SCHEMA
    assert receipt["operational_completion"] is False
    identity = dict(receipt)
    supplied = identity.pop("receipt_cid")
    assert supplied == canonical_json_cid(identity)
    assert "/" not in receipt["error_code"]


def test_client_success_receipt_cannot_embed_local_diagnostics():
    client = _client_receipt()
    identity = deepcopy(client)
    identity.pop("client_receipt_cid")
    identity["rendezvous_exercises"][0]["details"]["source_path"] = "/srv/private/runtime.json"
    client = _with_cid(identity, field_name="client_receipt_cid")

    with pytest.raises(
        TopologyCollectionError,
        match="identity_contains_local_path_field",
    ):
        _assemble(client=client)


def test_independent_client_executes_from_the_same_source_repository(monkeypatch):
    observed = {}

    async def fake_run_process(command, **options):
        observed["command"] = command
        observed["cwd"] = options["cwd"]
        assert options["capture_stdout"] is True
        assert "stdout" not in options
        return SimpleNamespace(
            returncode=0,
            stdout=canonical_identity_json(_client_receipt()).encode("utf-8"),
        )

    monkeypatch.setattr(trio, "run_process", fake_run_process)

    async def run():
        return await _run_independent_client_subprocess(
            target_multiaddr=_multiaddrs()[0],
            endpoint_url=ENDPOINT,
            expected_transport_model_id=DEFAULT_LEANSTRAL_MODEL_REF,
            rendezvous_namespace="leanstral-local",
            probe_timeout_s=2.0,
            model_timeout_s=1.0,
            process_timeout_s=9.0,
        )

    result = trio.run(run)

    assert result["client_receipt_cid"] == _client_receipt()["client_receipt_cid"]
    assert observed["command"][1:3] == [
        "-m",
        "ipfs_accelerate_py.mcplusplus_module.leanstral_topology_collector",
    ]
    assert DEFAULT_LEANSTRAL_MODEL_REF in observed["command"]
    source_repo = Path(observed["cwd"])
    assert source_repo == Path(__file__).resolve().parents[3]
    assert (source_repo / ".git").exists()


def test_main_keeps_lazy_dependency_logs_out_of_json_stdout(monkeypatch, capsys):
    """A lazy libp2p-style ``basicConfig`` call cannot corrupt the receipt."""

    def fake_trio_run(async_callable):
        del async_callable
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logging.getLogger("synthetic.lazy.libp2p").warning("synthetic dependency diagnostic")
        return _client_receipt()

    monkeypatch.setattr(topology_collector.trio, "run", fake_trio_run)

    return_code = topology_collector.main(
        [
            "_client",
            "--target-multiaddr",
            _multiaddrs()[0],
            "--endpoint-url",
            ENDPOINT,
            "--expected-transport-model-id",
            DEFAULT_LEANSTRAL_MODEL_REF,
            "--rendezvous-namespace",
            "leanstral-local",
            "--probe-timeout-s",
            "2.0",
            "--model-timeout-s",
            "1.0",
        ]
    )
    captured = capsys.readouterr()

    assert return_code == 0
    assert json.loads(captured.out) == _client_receipt()
    assert captured.out.count("\n") == 1
    assert "synthetic dependency diagnostic" not in captured.out
    assert "synthetic dependency diagnostic" in captured.err


def test_collector_uses_one_service_node_and_existing_model_endpoint(monkeypatch):
    created = []
    raw_model = _model()

    class FakeNode:
        def __init__(self, *, listen_addrs, bootstrap_peers, advertise_addrs):
            created.append(self)
            self.listen_addrs = listen_addrs
            self.bootstrap_peers = bootstrap_peers
            self.advertise_addrs = advertise_addrs
            self.handler = None
            self.stopped = False

        def set_tool_handler(self, handler):
            self.handler = handler

        async def start(self, nursery):
            del nursery

        async def stop(self):
            self.stopped = True

        def to_dict(self):
            return _server_status()

    model_provider_calls = []

    async def model_provider(endpoint_url, timeout_s):
        model_provider_calls.append((endpoint_url, timeout_s))
        return [raw_model]

    async def native_model_list_served(*, endpoint_url, timeout):
        assert endpoint_url == ENDPOINT
        assert timeout == 1.0
        return {"status": "success", "models": [raw_model], "count": 1}

    from ipfs_accelerate_py.mcp_server.tools.model_tools import native_model_tools

    monkeypatch.setattr(
        native_model_tools,
        "model_list_served",
        native_model_list_served,
    )

    async def client_runner(**kwargs):
        assert kwargs["target_multiaddr"] == _multiaddrs()[0]
        listing = await created[0].handler(
            "model_list_served",
            {"_sender_peer_id": CLIENT_PEER_ID},
        )
        return _client_receipt(models=listing["models"])

    async def run():
        return await collect_leanstral_topology(
            _config(),
            _require_clean_source=False,
            _source_commit=SOURCE_COMMIT,
            _node_factory=FakeNode,
            _client_runner=client_runner,
            _interface_observer=lambda _allowlist: _interfaces(),
            _model_provider=model_provider,
        )

    receipt = trio.run(run)

    assert receipt["validation"]["valid"] is True
    assert len(created) == 1
    assert created[0].listen_addrs == ["/ip4/0.0.0.0/tcp/19001"]
    assert created[0].bootstrap_peers == [BOOTSTRAP]
    assert created[0].advertise_addrs == [
        "/ip4/10.10.0.14/tcp/19001",
        "/ip4/10.8.0.99/tcp/19001",
        "/ip4/172.30.4.2/tcp/19001",
    ]
    assert created[0].stopped is True
    assert model_provider_calls == [(ENDPOINT, 1.0)]


def test_config_rejects_http_on_p2p_port_and_missing_interface_policy():
    with pytest.raises(ValueError, match="must not use"):
        LeanstralCollectorConfig(
            endpoint_url="http://127.0.0.1:19001/v1",
            allowed_interfaces=("tun0",),
        )
    with pytest.raises(ValueError, match="advertise interface"):
        LeanstralCollectorConfig(endpoint_url=ENDPOINT)
    with pytest.raises(ValueError, match="must not contain credentials"):
        LeanstralCollectorConfig(
            endpoint_url="http://user:redacted@127.0.0.1:8080/v1",
            allowed_interfaces=("tun0",),
        )
    with pytest.raises(ValueError, match="path-free interface"):
        LeanstralCollectorConfig(
            endpoint_url=ENDPOINT,
            allowed_interfaces=("/srv/private",),
        )
    with pytest.raises(ValueError, match="path-free identifier"):
        LeanstralCollectorConfig(
            endpoint_url=ENDPOINT,
            allowed_interfaces=("tun0",),
            rendezvous_namespace="../../private",
        )
    with pytest.raises(ValueError, match="path-free exact peer"):
        LeanstralCollectorConfig(
            endpoint_url=ENDPOINT,
            allowed_interfaces=("tun0",),
            bootstrap_peers=("file:///srv/private/p2p/secret",),
        )
    with pytest.raises(ValueError, match="path-free Leanstral transport"):
        LeanstralCollectorConfig(
            endpoint_url=ENDPOINT,
            expected_transport_model_id="../../Leanstral.gguf",
            allowed_interfaces=("tun0",),
        )
