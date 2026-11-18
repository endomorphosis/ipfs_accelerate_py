"""Full core dependency declarations can be read without executing setup.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import project_dependency_preflight as module


@pytest.mark.parametrize("source, expected", [
    ("from setuptools import setup\nsetup(install_requires=[])\n", []),
    ("from setuptools import setup as configure\nconfigure(install_requires=['attrs>=23', 'packaging'])\n",
     ["attrs>=23", "packaging"]),
])
def test_static_setup_core_is_complete_and_literal(source, expected):
    assert module._setup_core_literal_requirements(source.encode()) == expected


@pytest.mark.parametrize("source", [
    "from setuptools import setup\nsetup = other\nsetup(install_requires=[])\n",
    "from setuptools import setup\ndef setup(**kwargs): pass\nsetup(install_requires=[])\n",
    "from setuptools import setup\nsetup(install_requires=[])\nsetup(install_requires=[])\n",
    "from setuptools import setup\nif True: setup(install_requires=[])\n",
    "if True:\n from setuptools import setup\nsetup(install_requires=[])\n",
    "from setuptools import setup\nsetup(install_requires=requirements)\n",
    "from setuptools import setup\nsetup(install_requires=[name])\n",
    "from setuptools import setup\nsetup(install_requires=[], **options)\n",
    "from setuptools import setup\nfrom elsewhere import *\nsetup(install_requires=[])\n",
    "from elsewhere import setup\nsetup(install_requires=[])\n",
    "from setuptools import setup\nglobals()['setup'] = other\nsetup(install_requires=[])\n",
    "from setuptools import setup\nnamespace = vars\nnamespace()['setup'] = other\nsetup(install_requires=[])\n",
    "from setuptools import setup\nmodule.setup = other\nsetup(install_requires=[])\n",
])
def test_computed_shadowed_and_conditional_authority_is_rejected(source):
    with pytest.raises((ValueError, SyntaxError)):
        module._setup_core_literal_requirements(source.encode())


def _project(tmp_path: Path, setup_source: str, *, backend="setuptools.build_meta"):
    (tmp_path / "pyproject.toml").write_text(
        f'[build-system]\nbuild-backend = "{backend}"\nrequires = ["setuptools"]\n'
        '[project]\nname = "literal-core"\nversion = "1"\ndynamic = ["dependencies"]\n'
        '[project.optional-dependencies]\ntest = ["pytest>=8"]\n'
    )
    (tmp_path / "setup.py").write_text(setup_source)


@pytest.mark.parametrize("requirements", [[], ["attrs>=23"]])
def test_literal_core_manifest_and_pytest_runner_are_bound_and_probed(tmp_path, requirements):
    source = f"from setuptools import setup\nsetup(install_requires={requirements!r})\n"
    _project(tmp_path, source)
    payloads = []

    def probe(payload, **_kwargs):
        payloads.append(payload)
        return module._evaluate_dependency_payload(
            payload, version_getter={"attrs": "24.0.0", "pytest": "8.3.4"}.__getitem__,
        )

    receipt = module.preflight_validation_project_dependencies(
        tmp_path, ["python -m pytest tests/test_contract.py"], probe_runner=probe,
    )
    assert receipt["passed"] is True
    assert receipt["automatic_install_attempted"] is False
    assert len(payloads) == 1
    assert set(payloads[0]["projects"][0]["requirements"]) == {*requirements, "pytest", "pytest>=8"}
    project = receipt["projects"][0]
    assert project["dependency_source"] == "setuptools_setup_literal"
    assert project["dependency_manifests"][0]["content_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert project["dependency_manifests"][0]["bytes"] == len(source.encode())
    assert not (tmp_path / "executed").exists()


def test_setup_is_never_imported_and_manifest_tracks_file_changes(tmp_path):
    source = "from pathlib import Path\nPath('executed').write_text('bad')\nfrom setuptools import setup\nsetup(install_requires=[])\n"
    _project(tmp_path, source)
    project = module._bounded_static_project(tmp_path, ".")
    assert project["passed"] is True
    assert project["requirements"] == []
    assert project["dependency_manifests"][0]["sha256"] == hashlib.sha256(source.encode()).hexdigest()
    changed = source + "# changed metadata snapshot\n"
    (tmp_path / "setup.py").write_text(changed)
    successor = module._bounded_static_project(tmp_path, ".")
    assert successor["dependency_manifests"] != project["dependency_manifests"]
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize("backend", ["setuptools.build_meta", "hatchling.build"])
def test_nonliteral_or_foreign_backend_preserves_fail_closed_fallback(tmp_path, backend):
    source = "from setuptools import setup\nsetup(install_requires=requirements)\n" if backend.startswith("setuptools") else "from setuptools import setup\nsetup(install_requires=[])\n"
    _project(tmp_path, source, backend=backend)
    receipt = module.preflight_validation_project_dependencies(
        tmp_path, ["python -m pytest tests/test_contract.py"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail("unresolved metadata must not probe"),
    )
    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"
