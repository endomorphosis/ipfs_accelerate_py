#!/usr/bin/env python3
"""Smoke tests for deeper migrated module import surfaces."""

from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_module(name: str):
    return importlib.import_module(name)


def test_generator_module_imports(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    sample_test_generator = _import_module("scripts.generators.test_generators.sample_test_generator")
    skill_hf_bert = _import_module("scripts.generators.models.skill_hf_bert")
    model_templates = _import_module("scripts.generators.templates.model_templates")

    assert Path(sample_test_generator.__file__).is_file()
    assert Path(skill_hf_bert.__file__).is_file()
    assert Path(model_templates.__file__).is_file()


def test_duckdb_support_module_imports(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    duckdb_schema = _import_module("data.duckdb.schema")
    duckdb_schema_creation = _import_module("data.duckdb.schema.creation")
    cleanup_stale_reports = _import_module("data.duckdb.utils.cleanup_stale_reports")

    assert Path(duckdb_schema.__file__).is_file()
    assert Path(duckdb_schema_creation.__file__).is_file()
    assert Path(cleanup_stale_reports.__file__).is_file()