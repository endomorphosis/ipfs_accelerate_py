#!/usr/bin/env python3
"""Opt-in HuggingFace model inference tests.

These tests execute the model-specific harnesses in test/test_hf_*.py. They are
explicitly opt-in because they can be slow and may download large models.
"""

import importlib.util
import os
from pathlib import Path
from typing import List

import pytest


if os.getenv("IPFS_ACCEL_RUN_HF_MODEL_TESTS", "").lower() not in ("1", "true", "yes"):
    pytest.skip(
        "HF model inference tests are opt-in. Set IPFS_ACCEL_RUN_HF_MODEL_TESTS=1 to enable.",
        allow_module_level=True,
    )


def _discover_model_test_files() -> List[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    test_dir = repo_root / "ipfs_accelerate_py" / "test"
    paths = sorted(test_dir.glob("test_hf_*.py"))
    # Skip files with hyphens (invalid module names and some contain syntax errors).
    return [path for path in paths if "-" not in path.name]


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_harness_classes(module) -> List[type]:
    harnesses: List[type] = []
    for name in dir(module):
        if not name.startswith("Test"):
            continue
        candidate = getattr(module, name)
        if isinstance(candidate, type) and hasattr(candidate, "run_tests"):
            harnesses.append(candidate)
    return harnesses


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.cuda
@pytest.mark.parametrize("path", _discover_model_test_files(), ids=lambda p: p.stem)
def test_hf_model_harness(path: Path):
    module = _load_module_from_path(path)
    harnesses = _find_harness_classes(module)
    assert harnesses, f"No model harness found in {path}"

    for harness in harnesses:
        instance = harness()
        results = instance.run_tests()
        assert isinstance(results, dict), "Expected dict results"
        assert "results" in results, "Missing results in output"
        assert "hardware" in results, "Missing hardware in output"
