"""Typed scientific Git/profile joins; no service, DB, provider or native authority.

The H174 owning tests are reused with their explicit source-only fixture parsers
and modeled Git observations. Actual SDK/CLI3 compatibility is a separate
qualification test with authenticated real source inputs, not a fake witness.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
from scripts.ops.agent_supervisor import install_fleet_watchdog as installer



def pin(path):
    raw=path.read_bytes()
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return pin(path)


def test_authentic_current_Git_delivery_metadata_roundtrip_uses_trusted_original_parsers(tmp_path, monkeypatch):
    """Authentic DATA and physical-source joins; Git I/O alone is modeled.

    This does not claim a new source qualification, native cycle, task, provider,
    database observation or current manager health. Those proofs are separate.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    reference = Path(__file__).with_name("typed-scientific-qualified-reference-map.json")
    actual = json.loads(reference.read_bytes())
    profile_ref = actual["profile_ref"]; profile = json.loads(Path(profile_ref["path"]).read_bytes())
    inventory = json.loads(Path(actual["source_inventory_ref"]["path"]).read_bytes())
    release = tmp_path / "trusted-release"
    for role in ("health_module", "driver_source"):
        source_ref = actual["trusted_metadata_component_sources"][role]
        source = Path(source_ref["path"])
        assert pin(source) == source_ref == profile[role]
        target = release / "ipfs_accelerate_py/agent_supervisor/rescue" / source.name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
        assert pin(target)["sha256"] == profile[role]["sha256"]
    binding = write(tmp_path / "explicit-current-science-Git.json", {"schema": installer.SCIENTIFIC_GIT_BINDING_SCHEMA,
        "profile": profile_ref, "source_commitment": actual["source_commitment_ref"], "observer_plan": actual["seed_observer_plan_ref"]})
    def git(root, argv, **kwargs):
        assert str(root) == actual["source_root"] and kwargs["timeout"] == 3
        tail = argv[4:]
        if tail == ["rev-parse", "--show-toplevel"]: return str(root) + "\n"
        if tail == ["rev-parse", "--verify", "HEAD"]: return actual["source_commit"] + "\n"
        if tail == ["rev-parse", actual["source_commit"] + "^{tree}"]: return actual["source_tree"] + "\n"
        if tail == ["status", "--porcelain=v1", "-z", "--untracked-files=all"]: return ""
        if tail == ["ls-tree", "-r", "-z", "--full-tree", actual["source_commit"]]:
            return "".join(("100755" if r["mode"] & 0o111 else "100644") + " blob " + r["git_blob"] + "\t" + r["relative_path"] + "\0" for r in inventory["files"]) + "100644 blob " + "f" * 40 + "\tREADME.md\0"
        raise AssertionError(tail)
    monkeypatch.setattr(owners, "git_command", git)
    assert installer._repair_scientific_profile(profile_ref, release, binding) == profile

