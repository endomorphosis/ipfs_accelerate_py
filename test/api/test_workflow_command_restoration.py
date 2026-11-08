from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    restore_substituted_workflow_commands,
    workflow_pairs_are_non_weakening,
)

BEFORE = """\
name: pages
jobs:
  build:
    steps:
      - name: Build
        run: npm run build
      - name: Publish
        uses: actions/deploy-pages@v4
"""

AFTER = """\
name: pages
jobs:
  build:
    steps:
      - name: Build
        run: npx vite build
      - name: Publish
        uses: actions/deploy-pages@v4
"""


def test_substituted_build_command_is_restored_and_admitted():
    restored = restore_substituted_workflow_commands(BEFORE, AFTER)
    assert restored is not None
    assert "run: npm run build" in restored
    assert "run: npx vite build" in restored
    assert workflow_pairs_are_non_weakening(
        [(".github/workflows/deploy.yml", BEFORE, restored)]
    )
    assert restore_substituted_workflow_commands(BEFORE, restored) is None


def test_bypass_addition_is_not_restored():
    weakened = AFTER.replace(
        "run: npx vite build",
        "run: npx vite build || true",
    )
    assert restore_substituted_workflow_commands(BEFORE, weakened) is None


def test_block_scalar_removal_is_not_guessed():
    before = BEFORE.replace(
        "run: npm run build",
        "run: |\n          npm run build\n          npm run extra",
    )
    assert restore_substituted_workflow_commands(before, AFTER) is None
