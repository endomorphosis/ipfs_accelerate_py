"""DCR-091 typed-input fail-closed tests."""

from ipfs_accelerate_py.agent_supervisor.analysis.hermetic_conformance import (
    HermeticConformanceDisposition,
    HermeticConformanceReport,
)
from ipfs_accelerate_py.agent_supervisor.analysis.live_service_conformance import (
    assess_live_services,
)
from ipfs_accelerate_py.agent_supervisor.analysis.runtime_service_identity import (
    RuntimeServiceIdentity,
    RuntimeServiceObservation,
    ServiceIdentityStatus,
)


def test_plain_mappings_and_missing_live_evidence_are_pending() -> None:
    result = assess_live_services(
        None,
        (),
        observation_epoch=None,
        graph_cid="graph",
        semantic_roots={"s": "1"},
        snapshot_roots={"p": "1"},
        comparison=None,
    )
    assert result.disposition == "integration_pending"
    assert result.reason_codes == ("typed_dcr090_report_required",)


def test_forged_report_and_role_process_shapes_do_not_fabricate_success() -> None:
    result = assess_live_services(
        {"structural_fixture": True},  # type: ignore[arg-type]
        (),
        observation_epoch=None,
        graph_cid="graph",
        semantic_roots={"s": "1"},
        snapshot_roots={"p": "1"},
        comparison=None,
    )
    assert result.disposition == "integration_pending"
    assert result.execution_authorized is False


def test_wrong_role_and_noncurrent_process_witness_are_pending() -> None:
    report = HermeticConformanceReport(
        HermeticConformanceDisposition.INTEGRATION_PENDING,
        ("structural_fixture_non_live",),
        (),
        (),
    )
    identity = RuntimeServiceIdentity(
        RuntimeServiceObservation(
            role="forged",
            interpreter="python",
            module_origin="module.py",
            module_digest="sha256:module",
            checkout_commit="commit",
            checkout_tree="tree",
            overlay_id="",
            argv=("python",),
            environment={},
            config_cid="config",
            state_cid="state",
            transport="mcp",
            endpoint="http://127.0.0.1:1",
            pid=1,
            started_at="now",
            process_identity="untrusted",
            observed_port=1,
        ),
        ServiceIdentityStatus.INVALID,
        ("process_replaced_or_reused",),
    )
    result = assess_live_services(
        report,
        (identity, identity, identity),
        observation_epoch=None,
        graph_cid="graph",
        semantic_roots={"s": "1"},
        snapshot_roots={"p": "1"},
        comparison=None,
    )
    assert result.reason_codes == ("exact_accelerate_datasets_kit_roles_required",)
