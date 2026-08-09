"""DCR-022: bind launched MCP services to exact runtime identities."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.runtime_service_identity import (
    DEFAULT_SERVICES_RELATIVE,
    InvalidationReason,
    ObservationDisposition,
    ProcessIdentity,
    REQUIRED_SERVICE_ROLES,
    RUNTIME_SERVICE_MANIFEST_INTERFACE,
    RUNTIME_SERVICE_MANIFEST_SCHEMA,
    RUNTIME_SERVICE_WITNESS_INTERFACE,
    RUNTIME_WITNESS_EVIDENCE_TERM,
    RuntimeServiceIdentityError,
    RuntimeServiceManifest,
    ServiceEndpoint,
    ServiceRuntimeObservation,
    build_runtime_service_witness,
    content_cid,
    current_interpreter,
    is_pseudo_cid,
    load_runtime_service_manifest,
    synthesize_bound_observations,
    validate_observation_against_witness,
    write_runtime_witness,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / DEFAULT_SERVICES_RELATIVE).is_file():
            return candidate
    return here.parents[4]


def _observation_replace(
    observation: ServiceRuntimeObservation, **overrides: object
) -> ServiceRuntimeObservation:
    payload = observation.to_dict()
    payload.update(overrides)
    return ServiceRuntimeObservation.from_dict(payload)


def test_manifest_is_present_and_removes_port_disagreements() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    payload = manifest.to_dict()

    assert payload["schema"] == RUNTIME_SERVICE_MANIFEST_SCHEMA
    assert payload["interface"] == RUNTIME_SERVICE_MANIFEST_INTERFACE
    assert payload["evidence_term"] == RUNTIME_WITNESS_EVIDENCE_TERM
    assert manifest.service_id == "deterministic-contract-repair-mcp-runtime-v1"
    assert {item.role for item in manifest.services} == set(REQUIRED_SERVICE_ROLES)

    ports = [item.endpoint.port for item in manifest.services]
    assert len(ports) == len(set(ports)), "reviewed ports must be unique"
    urls = [item.endpoint.url for item in manifest.services]
    assert len(urls) == len(set(urls))

    assert manifest.policies["one_endpoint_per_service_role"] is True
    assert (
        manifest.policies["endpoint_availability_without_process_identity_insufficient"]
        is True
    )
    assert manifest.policies["port_disagreements_allowed"] is False
    assert "One reviewed endpoint per service role" in manifest.conflict_policy

    for service in manifest.services:
        assert not is_pseudo_cid(service.configuration_cid)
        assert not is_pseudo_cid(service.state_cid)
        assert service.configuration_cid == content_cid(dict(service.configuration))
        assert service.state_cid == content_cid(dict(service.state))
        assert service.endpoint.host == "127.0.0.1"
        assert service.endpoint.port >= 1024


def test_manifest_rejects_duplicate_ports() -> None:
    root = _repo_root()
    raw = json.loads((root / DEFAULT_SERVICES_RELATIVE).read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["services"][1]["endpoint"]["port"] = raw["services"][0]["endpoint"]["port"]
    raw["services"][1]["endpoint"]["url"] = (
        f"http://127.0.0.1:{raw['services'][1]['endpoint']['port']}/mcp"
    )
    with pytest.raises(RuntimeServiceIdentityError) as exc:
        RuntimeServiceManifest.from_dict(raw)
    assert exc.value.reason_code == InvalidationReason.PORT_DISAGREEMENT.value


def test_build_witness_binds_interpreter_modules_commit_args_env_config_endpoint() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(
        manifest,
        repo_root=root,
        environment={
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
            "HOME": "/tmp/ipfs-accelerate-validation-home-test",
            "SECRET_TOKEN": "must-not-bind",
        },
    )
    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    payload = witness.to_dict()

    assert witness.passed is True
    assert payload["interface"] == RUNTIME_SERVICE_WITNESS_INTERFACE
    assert payload["evidence_term"] == RUNTIME_WITNESS_EVIDENCE_TERM
    assert not is_pseudo_cid(payload["witness_cid"])
    assert not is_pseudo_cid(payload["manifest_cid"])
    assert len(witness.commit) == 40
    assert len(witness.tree) == 40
    assert "port_disagreements_removed" in witness.reason_codes
    assert "pid_start_time_bound" in witness.reason_codes
    assert payload["policies"]["process_replacement_invalidates"] is True

    for role in REQUIRED_SERVICE_ROLES:
        role_witness = witness.role_witness(role)
        assert role_witness.process.interpreter == current_interpreter()
        assert role_witness.process.pid > 0
        assert role_witness.process.start_time
        assert role_witness.endpoint_bound is True
        assert role_witness.modules
        for module in role_witness.modules:
            assert str(module["digest"]).startswith("sha256:")
            assert module["path"]
        assert "SECRET_TOKEN" not in role_witness.environment
        assert role_witness.environment.get("PATH")
        assert not is_pseudo_cid(role_witness.configuration_cid)
        assert not is_pseudo_cid(role_witness.state_cid)


def test_matching_observation_remains_valid() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    for observation in observations:
        verdict = validate_observation_against_witness(witness, observation)
        assert verdict.valid is True
        assert verdict.disposition is ObservationDisposition.VALID
        assert "observation_matches_witness" in verdict.reason_codes


def test_process_replacement_invalidates_observation() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    original = observations[0]
    assert original.process is not None
    replaced = _observation_replace(
        original,
        process=ProcessIdentity(
            pid=original.process.pid,
            start_time="boot0:ticks:replaced",
            interpreter=original.process.interpreter,
        ).to_dict(),
    )
    verdict = validate_observation_against_witness(witness, replaced)
    assert verdict.valid is False
    assert verdict.disposition is ObservationDisposition.INVALIDATED
    assert InvalidationReason.PROCESS_REPLACEMENT.value in verdict.reason_codes


def test_changed_config_or_state_invalidates_observation() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    original = observations[1]
    altered_config = content_cid({"mutated": True, "role": original.role})
    verdict = validate_observation_against_witness(
        witness,
        _observation_replace(original, configuration_cid=altered_config),
    )
    assert verdict.valid is False
    assert InvalidationReason.CONFIG_STATE_CHANGED.value in verdict.reason_codes

    altered_state = content_cid({"mutated_state": True, "role": original.role})
    verdict_state = validate_observation_against_witness(
        witness,
        _observation_replace(original, state_cid=altered_state),
    )
    assert verdict_state.valid is False
    assert InvalidationReason.CONFIG_STATE_CHANGED.value in verdict_state.reason_codes


def test_wrong_checkout_invalidates_observation() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    original = observations[2]
    verdict = validate_observation_against_witness(
        witness,
        _observation_replace(original, commit="0" * 40, tree="f" * 40),
    )
    assert verdict.valid is False
    assert InvalidationReason.WRONG_CHECKOUT.value in verdict.reason_codes


def test_unbound_endpoint_invalidates_observation() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    original = observations[0]
    unbound = _observation_replace(
        original,
        endpoint_bound=False,
        endpoint_available=False,
        endpoint=None,
        process=None,
    )
    verdict = validate_observation_against_witness(witness, unbound)
    assert verdict.valid is False
    assert verdict.disposition is ObservationDisposition.UNBOUND
    assert InvalidationReason.UNBOUND_ENDPOINT.value in verdict.reason_codes


def test_endpoint_availability_without_process_identity_is_insufficient() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    declaration = manifest.service_for_role("accelerate")
    git_obs = observations[0]
    incomplete = ServiceRuntimeObservation(
        role="accelerate",
        process=None,
        arguments=declaration.arguments,
        environment=git_obs.environment,
        endpoint=declaration.endpoint,
        endpoint_bound=False,
        endpoint_available=True,
        commit=git_obs.commit,
        tree=git_obs.tree,
        modules=git_obs.modules,
        configuration_cid=declaration.configuration_cid,
        state_cid=declaration.state_cid,
        transport=declaration.transport,
    )
    with pytest.raises(RuntimeServiceIdentityError) as exc:
        build_runtime_service_witness(
            manifest=manifest,
            observations=[
                incomplete,
                observations[1],
                observations[2],
            ],
            repo_root=root,
        )
    assert (
        exc.value.reason_code
        == InvalidationReason.ENDPOINT_WITHOUT_PROCESS_IDENTITY.value
    )

    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    verdict = validate_observation_against_witness(witness, incomplete)
    assert verdict.valid is False
    assert (
        InvalidationReason.ENDPOINT_WITHOUT_PROCESS_IDENTITY.value in verdict.reason_codes
    )


def test_port_disagreement_on_observation_fails_closed() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    bad_endpoint = ServiceEndpoint(
        kind="loopback_http",
        host="127.0.0.1",
        port=19999,
        path="/mcp",
        url="http://127.0.0.1:19999/mcp",
    )
    observations[0] = _observation_replace(
        observations[0],
        endpoint=bad_endpoint.to_dict(),
    )
    with pytest.raises(RuntimeServiceIdentityError) as exc:
        build_runtime_service_witness(
            manifest=manifest,
            observations=observations,
            repo_root=root,
        )
    assert exc.value.reason_code == InvalidationReason.PORT_DISAGREEMENT.value


def test_wrong_checkout_at_bind_time_fails_closed() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    with pytest.raises(RuntimeServiceIdentityError) as exc:
        build_runtime_service_witness(
            manifest=manifest,
            observations=observations,
            repo_root=root,
            expected_commit="0" * 40,
        )
    assert exc.value.reason_code == InvalidationReason.WRONG_CHECKOUT.value


def test_write_runtime_witness_is_atomic_and_content_addressed(tmp_path: Path) -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    witness = build_runtime_service_witness(
        manifest=manifest,
        observations=observations,
        repo_root=root,
    )
    out = tmp_path / "runtime-witness.json"
    written = write_runtime_witness(witness, path=out, repo_root=root)
    assert written == out
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["passed"] is True
    assert loaded["service_id"] == witness.service_id
    assert loaded["commit"] == witness.commit
    assert loaded["tree"] == witness.tree
    assert loaded["witness_cid"] == witness.witness_cid
    assert {role["role"] for role in loaded["roles"]} == set(REQUIRED_SERVICE_ROLES)
    assert not is_pseudo_cid(loaded["witness_cid"])


def test_pseudo_cid_is_rejected_for_config_and_state() -> None:
    root = _repo_root()
    manifest = load_runtime_service_manifest(repo_root=root)
    observations = synthesize_bound_observations(manifest, repo_root=root)
    with pytest.raises(RuntimeServiceIdentityError) as exc:
        _observation_replace(
            observations[0],
            configuration_cid="sha256:" + ("ab" * 32),
        )
    assert exc.value.reason_code == "pseudo_cid_rejected"
