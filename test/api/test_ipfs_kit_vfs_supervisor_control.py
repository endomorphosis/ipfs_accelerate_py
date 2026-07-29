from pathlib import Path


def test_vfs_grok_lane_enables_bounded_generated_board_repair() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    control_path = (
        repo_root
        / "scripts"
        / "ops"
        / "agent_supervisor"
        / "ipfs_kit_vfs_symbolic_assurance_control.sh"
    )
    text = control_path.read_text(encoding="utf-8")
    provider_block = text.split(
        'if [[ "${provider}" == "grok-build" ]]; then',
        1,
    )[1]
    grok_block, codex_block = provider_block.split("\n  else\n", 1)

    assert text.count('"--auto-commit-generated-dirty"') == 1
    assert grok_block.count('"--generated-dirty-path"') == 2
    assert '"--generated-dirty-path" "${OBJECTIVE_ABS}"' in grok_block
    assert '"--generated-dirty-path" "${TODO_ABS}"' in grok_block
    assert '"--generated-dirty-max-paths" "2"' in grok_block
    assert "--auto-commit-generated-dirty" not in codex_block
