from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.merge.protected_recovery_fence import (
    FENCE_CONTENTION_BACKOFF_SECONDS,
    generated_board_output_paths,
    is_generated_board_output_path,
    is_protected_recovery_fence_contention,
    is_supervisor_recovery_owner_script,
)


def test_generated_board_output_path_classifies_todo_and_discovery() -> None:
    assert is_generated_board_output_path("docs/architecture/board.todo.md")
    assert is_generated_board_output_path("docs/architecture/x.objectives.md")
    assert is_generated_board_output_path(
        "data/meta_glasses_display_widgets/discovery/note.md"
    )
    assert not is_generated_board_output_path(
        "scripts/run_agent_supervisor_proof_carrying_context_engine.py"
    )
    assert not is_generated_board_output_path(
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
    )
    assert not is_generated_board_output_path("/abs/tasks.todo.md")
    assert not is_generated_board_output_path("../escape.todo.md")


def test_generated_board_output_paths_preserves_order_and_skips_code() -> None:
    assert generated_board_output_paths(
        [
            "docs/a.todo.md",
            "scripts/operator.py",
            "docs/a.todo.md",
            "docs/b.objectives.md",
        ]
    ) == ("docs/a.todo.md", "docs/b.objectives.md")


def test_fence_contention_reasons_are_wait_states() -> None:
    assert is_protected_recovery_fence_contention(
        "external_protected_checkout_recovery_required"
    )
    assert is_protected_recovery_fence_contention(
        "checkout_mutation_protected_recovery_required"
    )
    assert not is_protected_recovery_fence_contention(
        "protected_generated_history_untrusted"
    )
    assert not is_protected_recovery_fence_contention("portal_provider_failed")
    assert FENCE_CONTENTION_BACKOFF_SECONDS == 30


def test_supervisor_entry_scripts_are_recovery_owners() -> None:
    assert is_supervisor_recovery_owner_script("implementation_supervisor.py")
    assert is_supervisor_recovery_owner_script(
        "/repo/scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
    )
    assert not is_supervisor_recovery_owner_script("implementation_daemon.py")
