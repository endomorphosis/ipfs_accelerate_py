"""Project pinned durable observations into a supervisor usage population.

The returned report contains PRIVATE complete receipts, as an ordinary retained
supervisor report does. It is an accounting input, not a new agent dispatch,
Harbor trial, official reward, complete campaign census, or billing record.
"""
from __future__ import annotations

from . import terminal_world_packet_attempt_collection as collection_owner

SCHEMA = "terminal-collected-supervisor-usage-report@1"


def project_collected_supervisor_report(declaration, slot_census, verifier_results=None, *, arm,
        expected_declaration_sha256, expected_slot_census_sha256,
        expected_verifier_results_sha256=None):
    """Return one arm's actual retained receipt population for usage consumers.

    Validate all original inputs and component joins before selecting an arm.
    Missing receipts are represented in the marker and slot census; they never
    become a successful empty population. Global unknown-call information is
    carried into both arms because it cannot safely be attributed away.
    Existing durability/quality gates are retained separately from token cost.
    """
    if type(arm) is not str or arm not in ("baseline", "candidate"):
        raise ValueError("explicit_collected_supervisor_arm_required")
    compared = collection_owner.compare_collected_paired_quality_usage(
        declaration, slot_census, verifier_results,
        expected_declaration_sha256=expected_declaration_sha256,
        expected_slot_census_sha256=expected_slot_census_sha256,
        expected_verifier_results_sha256=expected_verifier_results_sha256)
    collected = compared["collection"]
    attempts = [row for row in collected["attempts"] if row["arm"] == arm]
    receipts = [row["receipt"] for row in attempts if row["receipt"] is not None]
    coverage = collected["collection_coverage"]
    marker = coverage["unreceipted_provider_attempt"]
    missing = sum(row["receipt"] is None for row in attempts)
    if marker is False and (missing or coverage["supplied_attempt_census_complete"] is not True):
        marker = None
    return dict(schema=SCHEMA, provider_invocations=receipts,
        unreceipted_provider_attempt=marker,
        world_packet_attempt_collection=dict(
            arm=arm, collection_sha256=compared["collection_sha256"],
            comparison_sha256=collection_owner.attempt_collection_sha256(compared),
            accounting_report_sha256=compared["accounting_report_sha256"],
            input_pins=dict(collected["input_pins"]),
            declared_arm_slot_count=len(attempts), retained_arm_receipt_count=len(receipts),
            missing_arm_receipt_count=missing,
            arm_slot_states=[dict(attempt_id=row["attempt_id"], dispatch_state=row["dispatch_state"],
                outcome=row["outcome"], receipt_available=row["receipt"] is not None)
                for row in attempts],
            supplied_attempt_census_complete=coverage["supplied_attempt_census_complete"],
            durable_population_gate=dict(compared["durable_population_gate"]),
            comparison_gate=dict(compared["comparison_gate"]),
            token_cost_scope="supplied arm receipts; missing or unknown calls withhold complete totals",
            physical_provider_calls_verified=False, campaign_call_population_complete=None,
            new_provider_dispatch=False, official_reward_created=False,
            source_currentness_verified=False, provider_ingestion_verified=False,
            billing_total_verified=False, default_adoption_authorized=False,
            completion_authority=False))
