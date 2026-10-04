"""Render measured pilot results without imputing missing native usage."""
import argparse
import json
from pathlib import Path


def render(report):
    rows = report.get("runs", [])
    lines = ["# Router coding pilot results", "",
             f"Status: {report.get('status', 'in_progress')}", "",
             "Component pilot with four seeded repairs per arm; one observation per arm.",
             "This is not a Terminal-Bench score or a full supervisor-daemon evaluation.", "",
             "| Configuration | Passed | Wall seconds | Native tokens | Solver seconds | Token change vs serial |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        tokens = row.get("provider_tokens")
        savings = row.get("token_savings")
        change = "unknown" if savings is None else f"{-100 * savings:+.2f}%"
        proof_seconds = sum(p["wall_seconds"] for p in row.get("proof_receipts", []))
        lines.append(f"| {row['mode']} | {row['passed']} | {row['wall_seconds']:.2f} | "
                     f"{tokens if tokens is not None else 'unknown'} | {proof_seconds:.3f} | {change} |")
    lines += ["", "Provider and model: `" + str(report.get("route", {}).get("provider"))
              + "` / `" + str(report.get("route", {}).get("model")) + "`.", "",
              "Token totals include reported input and output tokens for every attempted call,",
              "including failed attempts. Cached input is a subset and is not added again.",
              "CLI system prompts and context are included in native usage. Differences between",
              "single runs may reflect sampling, cache effects, or provider load; they do not",
              "establish a general token-saving or speedup claim.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.input.read_text())))
