"""Intelligence Index v4.3 cost-efficient routing for supervisor llm_router calls."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LlmRouterInvocation,
    apply_efficient_llm_route,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm_defaults import DEFAULT_CODEX_MODEL
from ipfs_accelerate_py.llm_allocation.intelligence_index import (
    INDEX_VERSION,
    intelligence_floor_for_task,
    select_efficient_model,
    select_efficient_route,
)


def test_index_snapshot_is_v4_3() -> None:
    assert INDEX_VERSION == "4.3"


def test_cheapest_model_meets_difficulty_floor() -> None:
    easy = select_efficient_model(min_intelligence=26)
    assert easy.slug == "mimo-v2-5-pro"
    assert easy.cost_usd_per_task < 0.06

    standard = select_efficient_model(min_intelligence=38)
    assert standard.slug == "glm-5-3-flash"
    assert standard.intelligence >= 41
    assert standard.cost_usd_per_task < 0.30

    coding = select_efficient_model(min_intelligence=41)
    assert coding.slug == "glm-5-3-flash"

    agent = select_efficient_model(min_intelligence=45)
    assert agent.slug == "gpt-6-astra-low"
    assert agent.model_name == "gpt-6-astra"
    assert agent.reasoning_effort == "low"

    frontier = select_efficient_model(min_intelligence=52)
    assert frontier.slug == "gpt-6-astra-xhigh"
    assert frontier.cost_usd_per_task < 3.0


def test_provider_filter_stays_on_that_backend() -> None:
    grok = select_efficient_model(min_intelligence=38, provider="grok_cli")
    assert grok.model_name == "grok-4.6"
    assert grok.providers == ("grok_cli",)
    assert grok.reasoning_effort == "medium"
    assert grok.intelligence >= 43

    muse = select_efficient_model(min_intelligence=45, provider="muse_code")
    assert muse.model_name == "muse-spark-1.3"


def test_astra_beats_fable_at_same_intelligence() -> None:
    astra = select_efficient_model(min_intelligence=52)
    assert "astra" in astra.slug
    assert astra.cost_usd_per_task < 6.0


def test_task_kind_floors() -> None:
    assert intelligence_floor_for_task("easy") == 26
    assert intelligence_floor_for_task("implementation") == 41
    assert intelligence_floor_for_task("frontier") == 52
    assert intelligence_floor_for_task("", min_intelligence=50) == 50


def test_unpinned_supervisor_invocation_selects_efficient_model(tmp_path: Path) -> None:
    config = LlmRouterInvocation(repo_root=tmp_path, task_kind="implementation")
    assert config.model_name == DEFAULT_CODEX_MODEL
    resolved = apply_efficient_llm_route(config)
    assert resolved.model_name == "glm-5.3-flash"
    assert resolved.provider == "openrouter"
    assert resolved.catalog_revision == "aa-intelligence-index-v4.3"
    assert resolved.min_intelligence == 41


def test_pinned_provider_and_model_are_not_replaced(tmp_path: Path) -> None:
    config = LlmRouterInvocation(
        repo_root=tmp_path,
        model_name=DEFAULT_CODEX_MODEL,
        provider="codex_cli",
    )
    resolved = apply_efficient_llm_route(config)
    assert resolved.model_name == DEFAULT_CODEX_MODEL
    assert resolved.provider == "codex_cli"


def test_auto_model_name_selects_route() -> None:
    route = select_efficient_route(
        model_name="auto",
        task_kind="frontier",
    )
    assert route.auto_selected is True
    assert route.slug == "gpt-6-astra-xhigh"
    assert route.provider == "openai"
