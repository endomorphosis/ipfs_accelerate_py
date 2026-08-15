"""AAE-063: current-tree qualification of the final report and guide."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT = REPO_ROOT / "docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md"
GUIDE = REPO_ROOT / "docs/guides/adversarial_assurance_engine.md"
VALIDATOR = REPO_ROOT / "scripts/validate_adversarial_assurance_engine_board.py"

CONCLUDING_CLAIM = (
    "The system used semantically targeted counterfactual mutations to test whether "
    "declared tests, proofs, policies, semantic summaries, and incremental seals "
    "reject important incorrect behavior. Surviving mutants were classified as "
    "assurance gaps, candidate remediations were evaluated against held-out "
    "mutations, and accepted assurance-policy changes were promoted through a "
    "reproducible, content-addressed qualification process."
)

REQUIRED_TERMS = (
    "commits",
    "reuse",
    "operators",
    "counts",
    "scores",
    "survivors",
    "vacuity",
    "gaps",
    "detection",
    "cost",
    "cache",
    "remediation",
    "promotion",
    "regression",
    "seal",
    "improvement",
    "limits",
    "next steps",
    "generated",
    "admitted",
    "killed",
    "survived",
    "equivalent",
    "invalid",
    "inconclusive",
    "overconstraint",
    "non-claim",
)

FORBIDDEN_CLAIMS = (
    "high mutation score is proof of correctness",
    "proves the product is correct",
    "is production-authoritative",
)


def test_report_and_guide_exist() -> None:
    assert REPORT.is_file(), "AAE-063 report is absent"
    assert GUIDE.is_file(), "AAE-063 operator guide is absent"


def test_report_contains_required_vocabulary() -> None:
    text = REPORT.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())
    missing = [term for term in REQUIRED_TERMS if term not in normalized]
    assert missing == [], f"report omits required terms: {missing}"
    compact = " ".join(text.split())
    assert " ".join(CONCLUDING_CLAIM.split()) in compact
    for claim in FORBIDDEN_CLAIMS:
        assert claim not in normalized


def test_report_binds_current_gitlinks() -> None:
    text = REPORT.read_text(encoding="utf-8")

    def gitlink(path: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", f"HEAD:{path}"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    for relative in ("ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py/mcplusplus"):
        sha = gitlink(relative)
        assert re.fullmatch(r"[0-9a-f]{40}", sha)
        assert sha in text, f"report missing live gitlink {relative} {sha}"


def test_board_validator_accepts_current_tree() -> None:
    result = subprocess.run(
        ["python3", str(VALIDATOR), "--check-all"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"valid":true' in result.stdout.replace(" ", "")
