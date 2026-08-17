"""Qualification release manifest builder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType


def build_release(record):
    return MappingProxyType({"schema": "lgswf/qualification-release@1", "level": record.get("level") or "candidate"})


def verify(manifest_path: str, decision_path: str | None = None) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "lgswf/qualification-release@1":
        raise SystemExit("manifest schema rejected")
    result = {"manifest_ok": True, "schema": manifest["schema"]}
    if decision_path:
        decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        if decision.get("schema") != "lgswf/qualification-decision@1":
            raise SystemExit("decision schema rejected")
        result["decision_ok"] = True
        result["continuous_operation"] = decision.get("continuous_operation")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    verify_p = sub.add_parser("verify")
    verify_p.add_argument("--manifest", required=True)
    verify_p.add_argument("--decision")
    args = parser.parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify(args.manifest, args.decision)))
        return 0
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())

