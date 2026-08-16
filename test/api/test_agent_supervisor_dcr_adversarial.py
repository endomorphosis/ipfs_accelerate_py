"""Focused DCR-093 adversarial mutation framework tests."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_adversarial import (
    DEFAULT_MUTATION_CORPUS,
    MutationCase,
    MutationDisposition,
    evaluate_dcr_adversarial,
)


def test_representative_actual_boundary_attacks_are_killed() -> None:
    report = evaluate_dcr_adversarial()

    assert len(report.results) == len(DEFAULT_MUTATION_CORPUS)
    assert {item.observed_disposition for item in report.results} == {MutationDisposition.KILLED}
    assert report.disposition == "integration_pending"
    assert report.reason_codes == ("dcr092_positive_end_to_end_control_absent",)
    assert report.report_cid


def test_weakened_injected_validator_is_recorded_as_survivor() -> None:
    report = evaluate_dcr_adversarial(
        (MutationCase("remote_endpoint", "endpoint"),),
        validators={"endpoint": lambda _: False},
        positive_control_present=True,
    )

    assert report.disposition == "integration_pending"
    assert report.results[0].observed_disposition is MutationDisposition.SURVIVED
    assert report.reason_codes == ("surviving_or_error_mutation_blocks_readiness",)
