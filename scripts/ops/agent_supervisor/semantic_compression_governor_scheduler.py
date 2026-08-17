#!/usr/bin/env python3
"""Fail-closed supervisor launcher for the Semantic Compression Governor.

This is the reviewed SCG binding of the existing multi-lane implementation
supervisor.  It adds no scheduler, task authority, provider, or execution
profile; the implementation remains in the shared supervisor runtime.
"""

from __future__ import annotations

from pathlib import Path


_SOURCE = Path(__file__).with_name("incremental_verification_planner_scheduler.py")
_source = _SOURCE.read_text(encoding="utf-8")
for _old, _new in (
    ("incremental-verification planner", "semantic-compression governor"),
    ("incremental-verification-planner", "semantic-compression-governor"),
    ("incremental_verification_planner", "semantic_compression_governor"),
    ("IVP", "SCG"),
    ("ivp", "scg"),
    (
        "config/agent_supervisor_semantic_compression_governor_scheduler.json",
        "config/semantic_compression_governor_scheduler.json",
    ),
):
    _source = _source.replace(_old, _new)

# The SCG controller has three configured submodules (datasets, kit, MCP++) and
# nine protected controls. Bind the resulting current-tree CLI cardinality;
# the inherited check then semantically parses each lane and verifies its
# implement/refill/shard/reconciliation mapping rather than trusting length.
_source = _source.replace("len(common) != 59", "len(common) != 65")

# Execute a source-specialized copy so all constants, type names, lifecycle
# files, task prefix, and status schemas are SCG-bound before any command runs.
exec(compile(_source, str(Path(__file__)), "exec"), globals(), globals())
