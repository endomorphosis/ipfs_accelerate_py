"""Semantic domain-layout export names and board-prefix aliases."""

from __future__ import annotations

import importlib


def test_semantic_layout_exports_and_deprecated_aliases_match() -> None:
    api = importlib.import_module("ipfs_accelerate_py.agent_supervisor")

    # Preferred semantic names
    assert api.AGENT_SUPERVISOR_CORE_PACKAGES == ("core",)
    assert api.AGENT_SUPERVISOR_CONTROL_PACKAGES == ("control",)
    assert api.AGENT_SUPERVISOR_TASK_SOURCES_PACKAGES == ("task_sources",)
    assert api.AGENT_SUPERVISOR_CONTEXT_PROMPT_PACKAGES == ("context", "prompt")
    assert api.AGENT_SUPERVISOR_ANALYSIS_PROOF_PACKAGES == ("analysis", "proof")
    assert "todo_daemon" in api.AGENT_SUPERVISOR_INTEGRATIONS_DAEMON_PACKAGES
    assert "objectives" in api.AGENT_SUPERVISOR_OPERATIONS_PACKAGES

    assert api.AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS == (
        "ASREF-G020",
        "ASREF-G030",
        "ASREF-G040",
        "ASREF-G050",
    )
    assert api.AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS == (
        "ASREF-G060",
        "ASREF-G070",
        "ASREF-G080",
    )
    assert api.AGENT_SUPERVISOR_DOMAIN_LAYOUT_GOAL_IDS == (
        *api.AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS,
        *api.AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS,
    )

    # Board-prefix aliases remain and point at the same objects/values
    assert api.AGENT_SUPERVISOR_G020_PACKAGES is api.AGENT_SUPERVISOR_CORE_PACKAGES
    assert api.AGENT_SUPERVISOR_G030_PACKAGES is api.AGENT_SUPERVISOR_CONTROL_PACKAGES
    assert (
        api.AGENT_SUPERVISOR_G040_PACKAGES
        is api.AGENT_SUPERVISOR_TASK_SOURCES_PACKAGES
    )
    assert (
        api.AGENT_SUPERVISOR_G050_PACKAGES
        is api.AGENT_SUPERVISOR_CONTEXT_PROMPT_PACKAGES
    )
    assert (
        api.AGENT_SUPERVISOR_G060_PACKAGES
        is api.AGENT_SUPERVISOR_ANALYSIS_PROOF_PACKAGES
    )
    assert (
        api.AGENT_SUPERVISOR_G070_PACKAGES is api.AGENT_SUPERVISOR_OPERATIONS_PACKAGES
    )
    assert (
        api.AGENT_SUPERVISOR_G080_PACKAGES
        is api.AGENT_SUPERVISOR_INTEGRATIONS_DAEMON_PACKAGES
    )
    assert (
        api.AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G020_G050
        is api.AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS
    )
    assert (
        api.AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G060_G080
        is api.AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS
    )
    assert (
        api.AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE
        is api.AGENT_SUPERVISOR_DOMAIN_LAYOUT_GOAL_IDS
    )
    assert (
        api.AGENT_SUPERVISOR_PACKAGE_GOAL_TO_PACKAGES
        is api.AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES
    )
    assert (
        api.AGENT_SUPERVISOR_CUTOVER_GOAL_ID
        is api.AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_GOAL_ID
    )
    assert api.AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_GOAL_ID == "ASREF-G090"
    assert api.AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_TASK_ID == "ASREF-013"

    # Stem inventories
    assert "conflict_graph" in api.AGENT_SUPERVISOR_CORE_STEMS
    assert api.AGENT_SUPERVISOR_G020_CORE_STEMS is api.AGENT_SUPERVISOR_CORE_STEMS
    assert (
        api.AGENT_SUPERVISOR_G030_CONTROL_STEMS is api.AGENT_SUPERVISOR_CONTROL_STEMS
    )
    assert (
        api.AGENT_SUPERVISOR_G040_TASK_SOURCES_STEMS
        is api.AGENT_SUPERVISOR_TASK_SOURCES_STEMS
    )
    assert (
        api.AGENT_SUPERVISOR_G050_PLANNED_FLAT_MODULES
        is api.AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES
    )

    # Layout map still keys on board goal-id strings
    assert api.AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES["ASREF-G020"] == ("core",)
    assert api.AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES["ASREF-G080"] == (
        "todo_daemon",
        "integrations",
    )

    # Public __all__ includes preferred semantic names
    for name in (
        "AGENT_SUPERVISOR_CORE_PACKAGES",
        "AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS",
        "AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_GOAL_ID",
        "AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES",
    ):
        assert name in api.__all__
        assert hasattr(api, name)
