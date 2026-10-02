"""Check the frozen 32-item ledger without converting test passes into closure.

This verifies evidence bytes and prerequisite bookkeeping. It does not evaluate
the sufficiency of experiments, confer proof authority, or award benchmark score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def dependencies(text):
    result = set()
    for start, end in re.findall(r"RPI-(\d{3})\s+through\s+RPI-(\d{3})", text):
        if int(start) > int(end):
            raise ValueError("reversed dependency range")
        result.update(f"RPI-{n:03d}" for n in range(int(start), int(end) + 1))
    for group in re.findall(r"RPI-(\d{3}(?:/\d{3})*)", text):
        result.update("RPI-" + number for number in group.split("/"))
    return sorted(result)


def audit_ledger(ledger, *, repositories):
    rows = ledger["criteria"]
    expected = {f"RPI-{n:03d}" for n in range(1, 33)}
    if len(rows) != 32 or {r["id"] for r in rows} != expected:
        raise ValueError("exact unique frozen32 criterion population required")
    by_id = {r["id"]: r for r in rows}
    errors = []
    verified = []
    for name, evidence in ledger["evidence"].items():
        repository = evidence.get("repository", "ipfs_accelerate_py")
        if repository not in repositories:
            errors.append(dict(kind="missing_repository", evidence=name, repository=repository))
            continue
        path = Path(repositories[repository]) / evidence["path"]
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            errors.append(dict(kind="missing_evidence", evidence=name))
            continue
        if actual != evidence["sha256"]:
            errors.append(dict(kind="changed_evidence", evidence=name))
        else:
            verified.append(name)
    for row in rows:
        unknown = set(row["evidence"]) - ledger["evidence"].keys()
        if unknown:
            errors.append(dict(kind="unregistered_evidence", criterion=row["id"], evidence=sorted(unknown)))
        deps = dependencies(row["dependencies"])
        if any(d not in by_id for d in deps):
            raise ValueError("dependency outside frozen criterion population")
        if row["production_acceptance"] == "closed":
            open_deps = [d for d in deps if by_id[d]["production_acceptance"] != "closed"]
            if open_deps:
                errors.append(dict(kind="open_prerequisite", criterion=row["id"], dependencies=open_deps))
            if row.get("criterion_acceptance") != "qualified_for_declared_profile":
                errors.append(dict(kind="unqualified_closure", criterion=row["id"]))
            if row.get("blocking_dependencies") or row["remaining_work"] or not row["evidence"]:
                errors.append(dict(kind="incomplete_closure", criterion=row["id"]))
    closed = sum(r["production_acceptance"] == "closed" for r in rows)
    counts = dict(criteria=32, production_closed=closed, production_open=32-closed,
        criteria_qualified_for_declared_profile=sum(
            r.get("criterion_acceptance") == "qualified_for_declared_profile" for r in rows))
    if counts != ledger["summary"]:
        errors.append(dict(kind="incorrect_summary", expected=counts))
    return dict(schema="repository-backlog-bookkeeping-audit@1", valid=not errors,
        summary=counts, verified_evidence=verified, errors=errors,
        acceptance_sufficiency_evaluated=False, proof_authority=False, benchmark_score=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerate", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads((args.accelerate / "docs/architecture/repository_proof_index_backlog_status.json").read_text())
    report = audit_ledger(ledger, repositories={"ipfs_accelerate_py": args.accelerate,
                                               "ipfs_datasets_py": args.datasets})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
