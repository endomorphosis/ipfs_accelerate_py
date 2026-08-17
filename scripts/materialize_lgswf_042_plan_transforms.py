#!/usr/bin/env python3
"""Append LGSWF-042 proposal transforms onto ParallelPlanCompiler."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MARKER = "def propose_plan_transform"
EXTENSION = '''
ALLOWED_LGSWF_TRANSFORMS = frozenset({"split", "coalesce", "rewire", "speculate"})
IMMUTABLE_LGSWF_LIFECYCLES = frozenset({"claimed", "running", "accepted"})


class PlanTransformError(ValueError):
    """A plan transform proposal was rejected."""


def propose_plan_transform(request):
    """Propose a bounded split/coalesce/rewire/speculate PlanDelta only."""

    kind = request.get("kind")
    if kind not in ALLOWED_LGSWF_TRANSFORMS:
        raise PlanTransformError(f"unsupported transform {kind!r}")
    lifecycle = request.get("lifecycle") or "future"
    if lifecycle in IMMUTABLE_LGSWF_LIFECYCLES:
        raise PlanTransformError("claimed-through-accepted tasks are immutable")
    if kind == "speculate" and request.get("mutate_canonical"):
        raise PlanTransformError("speculation cannot mutate canonical authority")
    if int(request.get("amplification", 1)) > int(request.get("amplification_bound", 4)):
        raise PlanTransformError("unbounded task growth")
    return {
        "kind": kind,
        "accepted": False,
        "proposal": True,
        "evidence": request.get("evidence") or "coverage",
        "coverage_equivalent": bool(request.get("coverage_equivalent", True)),
        "predicted_parallelism": int(request.get("predicted_parallelism", 1)),
        "critical_path_delta": int(request.get("critical_path_delta", 0)),
        "resource_delta": int(request.get("resource_delta", 0)),
        "amplification_bound": int(request.get("amplification_bound", 4)),
        "dedup_key": str(request.get("dedup_key") or f"{kind}:{request.get('target')}"),
        "fallback": str(request.get("fallback") or "keep-current"),
        "human_review": bool(request.get("human_review", False)),
        "cancellable": True,
    }
'''
EXPORTS = (
    '    "PlanTransformError",\n'
    '    "propose_plan_transform",\n'
)


def apply_042(dest: Path) -> dict[str, object]:
    src_root = Path(__file__).resolve().parents[1]
    module = dest / "ipfs_accelerate_py/agent_supervisor/planning/parallel_plan_compiler.py"
    test_dst = dest / "test/api/test_agent_supervisor_lgswf_plan_transforms.py"
    text = module.read_text(encoding="utf-8")
    if MARKER not in text:
        if "\n__all__ = [\n" not in text:
            raise RuntimeError("parallel_plan_compiler.py missing __all__")
        text = text.replace("\n__all__ = [\n", "\n" + EXTENSION + "\n\n__all__ = [\n", 1)
        text = text.replace(
            '    "replay_parallel_execution_plan",\n]',
            '    "replay_parallel_execution_plan",\n' + EXPORTS + "]",
            1,
        )
        module.write_text(text, encoding="utf-8")
    test_src = src_root / "scripts/lgswf_payloads/test_agent_supervisor_lgswf_plan_transforms.py"
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.write_text(test_src.read_text(encoding="utf-8"), encoding="utf-8")
    outputs = [
        "ipfs_accelerate_py/agent_supervisor/planning/parallel_plan_compiler.py",
        "test/api/test_agent_supervisor_lgswf_plan_transforms.py",
    ]
    add = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *outputs],
        cwd=dest,
        text=True,
        capture_output=True,
        check=False,
    )
    return {"applied": MARKER in module.read_text(encoding="utf-8"), "staged": add.returncode == 0}


if __name__ == "__main__":
    print(json.dumps(apply_042(Path.cwd()), indent=2, sort_keys=True))
