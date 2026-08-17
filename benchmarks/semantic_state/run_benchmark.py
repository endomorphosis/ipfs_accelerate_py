#!/usr/bin/env python3
"""CLI entry for SemanticStateBenchmark@1 (SCH-017).

Usage:
  python3.12 benchmarks/semantic_state/run_benchmark.py
  python3.12 benchmarks/semantic_state/run_benchmark.py --write
  python3.12 benchmarks/semantic_state/run_benchmark.py --check

``--check`` recomputes the full corpus and compares deterministic semantic
fields to the published JSON after stripping wall-clock observations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.semantic_state.benchmark import (
    BENCHMARK_INTERFACE,
    EXPECTED_TASK_COUNT,
    BenchmarkError,
    check_report,
    render_markdown,
    run_benchmark,
    write_report,
)

DEFAULT_JSON = (
    REPO_ROOT / "docs" / "benchmarks" / "semantic_compression_harness_results.json"
)
DEFAULT_MD = (
    REPO_ROOT / "docs" / "benchmarks" / "semantic_compression_harness_results.md"
)


def _force_stage_published_json(json_path: Path) -> None:
    """Force-stage the published JSON so root ``*.json`` ignores cannot hide it.

    SCH-017 declares ``docs/benchmarks/semantic_compression_harness_results.json``
    as an output, but the repository root ``.gitignore`` matches ``*.json``.
    Without ``git add -f``, the implementation daemon's ``git add -A`` leaves the
    file untracked and fails the declared-output handoff invariant.
    """

    try:
        resolved = json_path.resolve()
        repo_root = REPO_ROOT.resolve()
        relative = resolved.relative_to(repo_root)
    except (OSError, ValueError):
        return
    if relative.as_posix() != "docs/benchmarks/semantic_compression_harness_results.json":
        return
    try:
        subprocess.run(
            ["git", "add", "-f", "--", str(relative)],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description=(
            "Run the exactly-40-task semantic compression harness benchmark "
            f"({BENCHMARK_INTERFACE})."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write JSON and Markdown results under docs/benchmarks/.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Recompute deterministic semantic fields and compare to published "
            "results (wall-clock fields excluded)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON,
        help=f"JSON results path (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=DEFAULT_MD,
        help=f"Markdown results path (default: {DEFAULT_MD})",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary JSON to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.check:
            envelope = check_report(json_path=args.json_out)
            print(json.dumps(envelope, indent=2, sort_keys=True))
            print(
                f"OK: deterministic fields match for {EXPECTED_TASK_COUNT} tasks; "
                "all gates passed."
            )
            return 0

        report = run_benchmark()
        summary = report["summary"]
        gates = summary.get("gates", {})
        if not all(gates.values()):
            failed = [name for name, ok in gates.items() if not ok]
            raise BenchmarkError(f"benchmark gates failed: {failed}")

        if args.write:
            json_path, md_path = write_report(
                report, json_path=args.json_out, markdown_path=args.md_out
            )
            _force_stage_published_json(Path(json_path))
            print(f"Wrote {json_path}")
            print(f"Wrote {md_path}")
        elif args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            # Default: print a short human summary and full markdown to stdout.
            print(
                f"{BENCHMARK_INTERFACE}: {summary['task_count']} tasks; "
                f"median reduction {summary['median_reduction_ratio'] * 100:.2f}%; "
                f"FN={summary['total_false_negatives']}; "
                f"stale={summary['total_stale_admissions']}; "
                f"simulated={summary['total_simulated_admissions']}"
            )
            print(render_markdown(report))

        return 0
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"ERROR: unexpected failure: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
