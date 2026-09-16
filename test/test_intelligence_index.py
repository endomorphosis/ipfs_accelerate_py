"""Intelligence Index v4.3 cost-efficient routing for supervisor llm_router calls."""

from __future__ import annotations

import json
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
    assert len(loaded) == 97
    assert all(item.intelligence > 0 for item in loaded)
    assert sum(1 for item in loaded if item.cost_usd_per_task > 0) == 92
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


def test_populate_openrouter_and_hf_inference_models(tmp_path: Path) -> None:
    from ipfs_accelerate_py.llm_allocation.model_manager_sync import (
        populate_hf_inference_models,
        populate_openrouter_models,
    )
    from ipfs_accelerate_py.model_manager import ModelManager

    manager = ModelManager(
        storage_path=str(tmp_path / "api-models.json"),
        use_database=False,
        enable_ipfs=False,
        project_legacy_models=False,
    )
    openrouter = populate_openrouter_models(manager, live=False)
    hf = populate_hf_inference_models(manager, live=False)
    assert openrouter["added"] >= 4
    assert hf["added"] >= 3
    sample = manager.get_model("openrouter:openai/gpt-6-astra")
    assert sample is not None
    assert sample.supported_backends == ["openrouter"]
    hf_sample = manager.get_model("hf:HuggingFaceH4/zephyr-7b-beta")
    assert hf_sample is not None
    assert hf_sample.supported_backends == ["hf_inference_api"]


BOARD_KIND_EXPECTATIONS = {
    "inventory": ("mimo-v2-5-pro", 26),
    "extraction": ("mimo-v2-5-pro", 26),
    "legal": ("mimo-v2-5-pro", 26),
    "review": ("glm-5-3-flash", 38),
    "unblock_review": ("glm-5-3-flash", 38),
    "planning": ("glm-5-3-flash", 38),
    "proposal": ("glm-5-3-flash", 38),
    "validation": ("glm-5-3-flash", 38),
    "implementation": ("glm-5-3-flash", 41),
    "repair": ("glm-5-3-flash", 41),
    "coding": ("glm-5-3-flash", 41),
    "merge": ("gpt-6-astra-low", 45),
    "agent": ("gpt-6-astra-low", 45),
    "rescue": ("gpt-6-astra-medium", 48),
    "scientific": ("gpt-6-astra-medium", 48),
    "frontier": ("gpt-6-astra-xhigh", 52),
}


def test_every_task_kind_has_an_ideal_model() -> None:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import (
        TASK_KIND_INTELLIGENCE,
        ideal_model_for_task,
        intelligence_floor_for_task,
    )

    for kind, floor in TASK_KIND_INTELLIGENCE.items():
        route = ideal_model_for_task({"kind": kind})
        assert route.auto_selected is True, kind
        assert route.task_kind == kind
        assert route.min_intelligence == floor
        assert route.cost_usd_per_task > 0
        assert route.intelligence + 1e-9 >= floor
        assert route.model_name
        assert route.provider


def test_board_items_resolve_expected_ideal_models() -> None:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import (
        board_task_kind,
        ideal_model_for_task,
    )

    for kind, (slug, floor) in BOARD_KIND_EXPECTATIONS.items():
        item = {
            "task_id": f"TASK-{kind.upper()}",
            "metadata": {"kind": kind},
            "title": f"{kind} work on the supervisor board",
        }
        assert board_task_kind(item) == kind
        route = ideal_model_for_task(item)
        assert route.slug == slug, (kind, route.slug, slug)
        assert route.min_intelligence == floor


def test_snapshot_covers_all_labs_and_costed_rows() -> None:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import (
        CLI_DEFAULT_MODELS,
        LAB_PROVIDERS,
        load_intelligence_index_models,
    )
    from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS

    loaded = load_intelligence_index_models()
    assert len(loaded) == 97
    costed = [item for item in loaded if item.cost_usd_per_task > 0]
    assert len(costed) == 92
    assert all(item.providers for item in loaded)
    assert CLI_PROVIDERS <= set(CLI_DEFAULT_MODELS)
    snapshot = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "ipfs_accelerate_py/llm_allocation/data/intelligence_index_v4_3.json"
        ).read_text(encoding="utf-8")
    )
    labs = {str(row["lab_slug"]) for row in snapshot["records"]}
    assert labs <= set(LAB_PROVIDERS)


def test_cli_only_board_still_picks_a_completing_model() -> None:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import ideal_model_for_task
    from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS

    item = {"metadata": {"kind": "implementation"}}
    for provider in sorted(CLI_PROVIDERS):
        route = ideal_model_for_task(item, available_providers=[provider])
        assert route.provider == provider, provider
        assert route.model_name
        assert route.cost_usd_per_task > 0
        if route.intelligence + 1e-9 < route.min_intelligence:
            # Provider has no Index model at the floor; still return its best.
            assert route.intelligence > 0, provider
        else:
            assert route.intelligence >= route.min_intelligence, provider


def test_no_cost_snapshot_rows_are_not_selected() -> None:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import (
        load_intelligence_index_models,
        select_efficient_model,
    )

    missing = [item.slug for item in load_intelligence_index_models() if item.cost_usd_per_task <= 0]
    assert "devstral-2" in missing
    assert "gemma-4-31b" in missing
    picked = select_efficient_model(min_intelligence=8)
    assert picked.slug not in missing
    assert picked.cost_usd_per_task > 0
