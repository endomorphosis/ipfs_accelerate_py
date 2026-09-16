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
    assert grok.providers == ("grok_cli",)
    assert grok.model_name.startswith("grok-")
    assert grok.intelligence >= 38

    grok_hard = select_efficient_model(min_intelligence=43, provider="grok_cli")
    assert grok_hard.model_name == "grok-4.6"
    assert grok_hard.intelligence >= 43

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
    resolved = apply_efficient_llm_route(config, discover=False)
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


def test_full_snapshot_has_cost_intelligence_matrix() -> None:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import (
        intelligence_cost_matrix,
        load_intelligence_index_models,
    )

    loaded = load_intelligence_index_models()
    assert len(loaded) >= 90
    assert all(item.cost_usd_per_task > 0 and item.intelligence > 0 for item in loaded)
    openai_only = intelligence_cost_matrix(available_providers=["openai"])
    assert openai_only
    assert all("openai" in row["providers"] for row in openai_only)
    grok_only = intelligence_cost_matrix(available_providers=["grok_cli"])
    assert grok_only
    assert all("grok_cli" in row["providers"] for row in grok_only)


def test_available_openai_does_not_select_openrouter_glm() -> None:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import select_efficient_model

    picked = select_efficient_model(
        min_intelligence=41,
        available_providers=["openai", "codex_cli"],
    )
    assert picked.slug != "glm-5-3-flash"
    assert picked.providers[0] in {"openai", "codex_cli"}


def test_populate_intelligence_index_into_model_manager(tmp_path: Path) -> None:
    from ipfs_accelerate_py.llm_allocation.model_manager_sync import (
        populate_intelligence_index,
    )
    from ipfs_accelerate_py.model_manager import ModelManager

    manager = ModelManager(
        storage_path=str(tmp_path / "models.json"),
        use_database=False,
        enable_ipfs=False,
        project_legacy_models=False,
    )
    result = populate_intelligence_index(manager)
    assert result["added"] >= 90
    sample = manager.get_model("aa:4.3:gpt-6-astra")
    assert sample is not None
    metrics = sample.performance_metrics or {}
    assert metrics["intelligence_index"] > 50
    assert metrics["cost_usd_per_task"] > 0
    assert "openai" in (sample.supported_backends or [])


def test_populate_cli_models_uses_installed_tools(tmp_path: Path, monkeypatch) -> None:
    from ipfs_accelerate_py.llm_allocation import model_manager_sync
    from ipfs_accelerate_py.model_manager import ModelManager

    monkeypatch.setattr(
        "ipfs_accelerate_py.llm_allocation.cli_status.cli_tools_status",
        lambda: {
            "tools": {
                "grok_cli": {
                    "installed": True,
                    "executable": "/tmp/grok",
                    "ready": True,
                    "authenticated": True,
                }
            }
        },
    )
    monkeypatch.setattr(
        model_manager_sync,
        "list_cli_provider_models",
        lambda provider, executable="", timeout_seconds=4.0: ("grok-4.6",),
    )
    manager = ModelManager(
        storage_path=str(tmp_path / "cli-models.json"),
        use_database=False,
        enable_ipfs=False,
        project_legacy_models=False,
    )
    result = model_manager_sync.populate_cli_models(manager)
    assert result["providers"]["grok_cli"]["models"] == ["grok-4.6"]
    stored = manager.get_model("cli:grok_cli:grok-4.6")
    assert stored is not None
    assert stored.supported_backends == ["grok_cli"]
