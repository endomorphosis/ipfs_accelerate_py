#!/usr/bin/env python3
"""Build the compact content-addressed SCG-040 fixture corpus manifest.

Recipes stay compact. Re-run this script after editing recipes to refresh
``manifest.json``. The manifest stores digests and oracles, not full file
bodies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent.parent.parent) not in sys.path:
    # Allow direct execution from a checkout without installing the package.
    sys.path.insert(0, str(PACKAGE_DIR.parent.parent.parent))

# Load as a standalone package under a stable name.
import importlib.util
from types import ModuleType


def _load_package() -> ModuleType:
    package_name = "scg_fixture_corpus_builder"
    if package_name in sys.modules:
        return sys.modules[package_name]

    init_path = PACKAGE_DIR / "__init__.py"
    package = ModuleType(package_name)
    package.__file__ = str(init_path)
    package.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    def _load(name: str, filename: str) -> ModuleType:
        qualname = f"{package_name}.{name}"
        path = PACKAGE_DIR / filename
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        return module

    _load("case_record", "case_record.py")
    _load("recipes", "recipes.py")
    _load("corpus", "corpus.py")
    init_spec = importlib.util.spec_from_file_location(
        package_name, init_path, submodule_search_locations=[str(PACKAGE_DIR)]
    )
    assert init_spec is not None and init_spec.loader is not None
    package.__spec__ = init_spec
    package.__package__ = package_name
    init_spec.loader.exec_module(package)
    return package


def main() -> int:
    pkg = _load_package()
    corpus = pkg.SemanticGovernorFixtureCorpus.load()
    manifest = corpus.to_manifest()
    out = PACKAGE_DIR / "manifest.json"
    out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {out} cases={manifest['case_count']} "
        f"digest={manifest['corpus_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
