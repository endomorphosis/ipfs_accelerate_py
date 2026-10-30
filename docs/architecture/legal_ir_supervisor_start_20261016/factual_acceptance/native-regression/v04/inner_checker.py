"""Pinned, network-isolated container checker; no host acceptance or provider API."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time


def main():
    if len(sys.argv) != 2 or not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
        raise SystemExit("isolated no-site no-bytecode checker required")
    manifest = json.loads(Path(sys.argv[1]).read_bytes())
    if set(manifest) != {"schema", "scope", "source_root", "report_path", "report_sha256", "evidence_path", "gate_path"} or manifest["schema"] != "legal-ir-docker-inner-factual-request@1" or manifest["scope"] != "synthetic-native-acceptance-regression-only":
        raise SystemExit("closed fixture checker request required")
    gate = Path(manifest["gate_path"])
    deadline = time.monotonic() + 180
    while True:
        if time.monotonic() >= deadline:
            raise SystemExit("actual container admission barrier expired")
        try:
            fd = os.open(gate, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != 1 or stream.read(2) != b"G":
                raise SystemExit("invalid actual container admission barrier")
        break
    spec = importlib.util.spec_from_file_location("legal_ir_factual_checker", "/opt/legal-ir/factual_candidate.py")
    facts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(facts)
    sources, frozen = facts.load_source_bundle(Path(manifest["source_root"]))
    report = facts.read_source_file(Path(manifest["report_path"]))
    if facts.sha(report) != manifest["report_sha256"]:
        raise SystemExit("exact fixture report pin differs")
    evidence = facts.canonical(facts.check_facts(report, sources, frozen))
    fd = os.open(manifest["evidence_path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(evidence)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
