"""EAAEF-160: pinned wheels without editable/sibling authority."""

from __future__ import annotations

import pytest

from scripts.release.build_external_agent_stack import (
    PRIMARY_PACKAGES,
    StackBuildError,
    default_primary_plan,
    plan_wheel_build,
)


def test_default_plan_is_three_primary_packages() -> None:
    plan = default_primary_plan()
    assert tuple(plan["packages"]) == PRIMARY_PACKAGES
    assert plan["editable"] is False
    assert plan["sibling_checkout"] is False
    assert plan["include_mcp"] is False


def test_editable_and_sibling_are_rejected() -> None:
    with pytest.raises(StackBuildError, match="editable"):
        plan_wheel_build(
            packages=PRIMARY_PACKAGES,
            include_mcp=False,
            editable=True,
            sibling_checkout=False,
        )
    with pytest.raises(StackBuildError, match="sibling"):
        plan_wheel_build(
            packages=PRIMARY_PACKAGES,
            include_mcp=False,
            editable=False,
            sibling_checkout=True,
        )
