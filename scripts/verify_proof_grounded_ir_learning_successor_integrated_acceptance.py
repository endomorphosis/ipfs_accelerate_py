#!/usr/bin/env python3
"""Independently replay the integrated PGIR-211 successor evidence.

The complete verdict is intentionally network-gated.  Plain/offline invocation
fails closed.  ``--component`` is a diagnostic-only static replay: it may emit
``component_verified=true`` but always emits ``verified=false`` and never
authorizes PGIR-205.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOURCE_ROOT
DATASETS = ROOT / "ipfs_datasets_py"
DIR = SOURCE_ROOT / "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance"
ACCEPTANCE = DIR / "integrated_acceptance.json"
HISTORICAL = DIR / "historical_closure_receipt.json"
TARGET = "75791d58beeab140c2a3ebaf9789705b3e75c151"
TARGET_TREE = "e092bc48487226229c0df5c47029c3db36004e18"
CURRENT = "2a06dfe8546cdde78ff6d101a94708be0e6bf6e6"
CURRENT_TREE = "7169c2a67929044a02350bc26d0a51c853a4981b"
SEALED = "8736a0023d5d3afe4d0e5b044a3e4480966a8bf7"
SEALED_TREE = "33ff339e87858fba7ee812db16fbc8c8565716ca"
R1_REVISION = "9313a20a3d7281e4ee9e8efdf907795c89ff4b65"
CAMPAIGN_OUTER_REVISION = "04fbb09b4a8b34e77d11bd8da6642e0978baa02c"
CAMPAIGN_NESTED_REVISION = "b20bd9e3cfae79e8888929daf64f52b2f8a5689a"
TOKENIZER_REVISION = "ec3cd078683c25cc4cecf96b00cb77adfd2e7231"
TOKENIZER_BLOB = "89ca3265b517900f99a3a9f6b3d4770d4002baf8"
CAMPAIGN_OBJECTIVE_CID = "baguqeeralbl2yjo6l5gazcmslpzqtu67un4txk3wwpjr45thh5sckwq67yhq"
HIDDEN = "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"
HOLDOUTS = (
    "compiler", "cross_reference", "domain", "exception", "length",
    "lineage", "notation", "premise", "proof_library", "publication",
    "rare_operator", "time", "type",
)

SOURCE_LINEAGE_PATH = "ipfs_datasets_py/logic/ir_core/source_lineage.py"
SOURCE_LINEAGE_R1_BLOB = "7109b47372c94f3097c2712247b4863f4b0ad438"
SOURCE_LINEAGE_CURRENT_BLOB = "b5ff8fc95b1c8c77a3d94ccd2d8d093ff0f8a715"
SOURCE_LINEAGE_CURRENT_ADDITIVE_SUFFIX = b'''\n\n__all__ = [
    "CORPUS_MANIFEST_SCHEMA",
    "DERIVED_ARTIFACT_SCHEMA",
    "LINEAGE_GRAPH_SCHEMA",
    "SOURCE_RECORD_SCHEMA",
    "SOURCE_RECORD_SCHEMA_V1_1",
    "SOURCE_RELEASE_SCHEMA",
    "CorpusManifest",
    "DerivedArtifactRecord",
    "LineageEdge",
    "LineageGraph",
    "LineageRelation",
    "RecordKind",
    "RightsDisposition",
    "RightsRecord",
    "SourceLineageError",
    "SourceRecord",
    "SourceRelease",
    "TemporalCoverage",
    "source_lineage_schema_ids",
    "source_lineage_schema_registry",
]\n'''

P208_CID = "baguqeeraburgmpdfo6weea57zlgkmppv7r34v2v3zstrepudxhrj6zrlgabq"
P209_CID = "baguqeerauh6r5lk47ecfmu5zjujmadrjiohd2ixkczcnurc33izkkvf2nb7q"
P210_CID = "baguqeera4ruaxwivpst2iwslorrgmbpuva6jqxczyjf6uditxg62atltnvkq"

PREDECESSOR_FILES = (
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance/README.md",
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance/source_chain_acceptance.json",
    "scripts/verify_proof_grounded_ir_learning_successor_source_chain.py",
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/baseline-acceptance/baseline_acceptance.json",
    "scripts/verify_proof_grounded_ir_learning_successor_baseline.py",
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance-v2/README.md",
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance-v2/network_replay_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance-v2/source_chain_acceptance_v2.json",
    "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/source-chain-acceptance-v2/verification_receipt.json",
    "scripts/verify_proof_grounded_ir_learning_successor_source_chain_v2.py",
)
PREDECESSOR_ROWS = tuple(
    [("PGIR-208", "edb419c80275757cbedac8c6731e49f85ad58d88", path) for path in PREDECESSOR_FILES[:3]]
    + [("PGIR-209", "7e7b49380c2b842f7ccce6b6f1bd414d0d14be14", path) for path in PREDECESSOR_FILES[3:5]]
    + [("PGIR-210", TARGET, path) for path in PREDECESSOR_FILES[5:]]
)

R1_FILES = (
    ("identities.json", "identities_cid"),
    ("manifest.json", "manifest_cid"),
    ("metric_catalog.json", "catalog_cid"),
    ("r1_baseline.json", "report_cid"),
    ("recipe.json", "recipe_cid"),
    ("strata.json", "strata_cid"),
    ("tool_versions.json", "tool_versions_cid"),
)

RETIREMENT_FILES = (
    "identities.json", "manifest.json", "recipe.json", "replay_receipt.json",
    "retirement_receipt.json", "strata.json", "tool_versions.json",
)

SEALED_PATHS = (
    "data/ir_learning/corpora/successor-v1/rights_manifest.json",
    "data/ir_learning/corpora/successor-v1/quarantine_manifest.json",
    "data/ir_learning/corpora/successor-v1/source_releases.json",
    "data/ir_learning/corpora/successor-v1/replay_receipt.json",
    "data/ir_learning/corpora/successor-v1/corpus_manifest.json",
    "data/ir_learning/corpora/successor-v1/count_receipt.json",
    "data/ir_learning/corpora/successor-v1/lineage_graph.json",
    "data/ir_learning/corpora/successor-v1/load_receipt.json",
    "data/ir_learning/corpora/successor-v1/corpus_root.json",
    "data/ir_learning/splits/successor-v1/holdout_report.json",
    "data/ir_learning/splits/successor-v1/leakage_report.json",
    "data/ir_learning/splits/successor-v1/ir_split_manifest.json",
    "data/ir_learning/splits/successor-v1/replay_receipt.json",
    "data/ir_learning/splits/successor-v1/split_root.json",
)

CAMPAIGN_COUNTS = {
    "compiler": 2,
    "corpus": 7,
    "decompiler": 2,
    "example_contracts": 5,
    "gap_matrix": 1,
    "lineage": 1,
    "policy": 2,
    "rights": 1,
    "schema_registry": 11,
    "source_snapshots": 6,
    "split": 5,
    "tokenizer_policy": 1,
}

OUTER_FOREST = (
    ("PGIR-209", "implementation", "b435a3cd881183badf6aceb8b833ed8c9674f76f", "62513790747295c3c0052f2ed75e614453da928a", ("2d5088fbac0f616fce13d2c468d3fda21ede3615",), CURRENT),
    ("PGIR-209", "merge", "cc9441d0808cea8159d96339c4b167ced1b89302", "b0dd0b85b1cc0d264398aed0c4bc970a714d9225", ("1d5c543e1f3471d273c8504390c44fc7c63ad20a", "b435a3cd881183badf6aceb8b833ed8c9674f76f"), CURRENT),
    ("PGIR-209", "completion", "7e7b49380c2b842f7ccce6b6f1bd414d0d14be14", "217bb8e88d0d62a183821198cf880219fe0d1dd0", ("cc9441d0808cea8159d96339c4b167ced1b89302",), CURRENT),
    ("PGIR-210", "implementation", "5e374b73ee521e077fb730f32f969019403dcd63", "9f6ff5eddd51878dffe8ac53eae7d4841f399eff", ("1d5c543e1f3471d273c8504390c44fc7c63ad20a",), CURRENT),
    ("PGIR-210", "merge", "7682ac51d13c29e9bf3f895686ea7b545e954941", "265bb65ffd4f1d7eff6d708e5e59c098dafc57ff", ("7e7b49380c2b842f7ccce6b6f1bd414d0d14be14", "5e374b73ee521e077fb730f32f969019403dcd63"), CURRENT),
    ("PGIR-210", "completion", TARGET, TARGET_TREE, ("7682ac51d13c29e9bf3f895686ea7b545e954941",), CURRENT),
)

CAS = (
    ("PGIR-200", "c30ccbec997868b061c4cadac38d30468c46ea2d", "0566a833e795b0f0596251c2e7e8ca7d8ec27836", "511ae84626a38dc43ed2851ca4a16c67ff1ac4ca", "2d5eee99c31f3649515d6ca076765f8080096af9", "1289be82793e590324bb4a830e0bd2584a70ec08"),
    ("PGIR-201", "0566a833e795b0f0596251c2e7e8ca7d8ec27836", "8cc72c77736d3ff2db7cc2530e619bf09b5be027", "b189f26e316d1bf6f7760bcfcef3e1d705011a6d", "e15127fc8e9de97cd6efa576d836ea995d301dc4", "c70dbcaed6bb66eb42517bb040f2d9602ab69a66"),
    ("PGIR-202", "8cc72c77736d3ff2db7cc2530e619bf09b5be027", SEALED, "f38fffb9b96ac06d7894ece8eb6030dc2b35bb83", "633411fa297bf5a315b6b7c1dccdd4c4a61626c2", "249e7fcac0d8e6e6baa0034ee4bb5b24034c74f5"),
)

CAS_TOPOLOGY = {
    "PGIR-200": {
        "prior": "ef3497886fc8b31cfd8f0a5ac45a8026fae0aee3",
        "implementation_tree": "197b6471baea6e86c0ec049f5b8c34082d552ec4",
        "merge_tree": "197b6471baea6e86c0ec049f5b8c34082d552ec4",
        "completion_tree": "c05fd5121bd3220025212adcdad7d1bed159dd37",
    },
    "PGIR-201": {
        "prior": "1289be82793e590324bb4a830e0bd2584a70ec08",
        "implementation_tree": "fbc96645c5860f17f93b249bbe97e6fa8085f5aa",
        "merge_tree": "fbc96645c5860f17f93b249bbe97e6fa8085f5aa",
        "completion_tree": "9656c43406e8ab00f757a80fe902fe9d674f1e45",
    },
    "PGIR-202": {
        "prior": "c70dbcaed6bb66eb42517bb040f2d9602ab69a66",
        "implementation_tree": "e882922764c6df3a7b3101e7b1ac80563e2cf4e4",
        "merge_tree": "e882922764c6df3a7b3101e7b1ac80563e2cf4e4",
        "completion_tree": "be65228bcf951461ee143d8d9fdbd376233a8d16",
    },
}

OUTER_CANDIDATES = (
    ("ef3497886fc8b31cfd8f0a5ac45a8026fae0aee3", ("PGIR200 implementation/merge parent", "source-chain reviewed-provider barrier")),
    ("511ae84626a38dc43ed2851ca4a16c67ff1ac4ca", ("PGIR200 implementation",)),
    ("2d5eee99c31f3649515d6ca076765f8080096af9", ("PGIR200 merge",)),
    ("1289be82793e590324bb4a830e0bd2584a70ec08", ("PGIR200 completion",)),
    ("b189f26e316d1bf6f7760bcfcef3e1d705011a6d", ("PGIR201 implementation",)),
    ("e15127fc8e9de97cd6efa576d836ea995d301dc4", ("PGIR201 merge",)),
    ("c70dbcaed6bb66eb42517bb040f2d9602ab69a66", ("PGIR201 completion",)),
    ("f38fffb9b96ac06d7894ece8eb6030dc2b35bb83", ("PGIR202 implementation",)),
    ("633411fa297bf5a315b6b7c1dccdd4c4a61626c2", ("PGIR202 merge",)),
    ("249e7fcac0d8e6e6baa0034ee4bb5b24034c74f5", ("PGIR202 completion", "PGIR208 barrier parent")),
    ("c4cf42fccb58d73b9f48c7f70799165b29cfe3a9", ("PGIR208 source-chain acceptance barrier", "PGIR208 implementation/merge parent")),
    ("effac002de038c02ef94cdb8e1b1b2c9a0a8d2e3", ("PGIR208 implementation",)),
    ("1ceb7183b16b3358dd956ebcd4917955e017ef7f", ("PGIR208 merge",)),
    ("edb419c80275757cbedac8c6731e49f85ad58d88", ("PGIR208 completion",)),
    ("b87bc6d28fbfefd836696f972f51ee9a677b5071", ("PGIR203 completion", "PGIR204 implementation/merge parent")),
    ("fdf3839bc11c60c7e0b801397499a7b4ce0d4634", ("PGIR204 implementation",)),
    ("a0f54e787e141d6d03fcef56b7d48ff42ce516f1", ("PGIR204 merge",)),
    ("b9c7aaece6c75b1413a6a951f647ca409834dff9", ("PGIR204 completion",)),
    ("1d5c543e1f3471d273c8504390c44fc7c63ad20a", ("recovered-lifecycle barrier", "PGIR210 implementation parent", "PGIR209 merge first parent")),
    ("2d5088fbac0f616fce13d2c468d3fda21ede3615", ("post-merge evidence barrier", "PGIR209 implementation parent")),
    ("b435a3cd881183badf6aceb8b833ed8c9674f76f", ("PGIR209 implementation",)),
    ("cc9441d0808cea8159d96339c4b167ced1b89302", ("PGIR209 merge",)),
    ("7e7b49380c2b842f7ccce6b6f1bd414d0d14be14", ("PGIR209 completion", "PGIR210 merge first parent")),
    ("5e374b73ee521e077fb730f32f969019403dcd63", ("PGIR210 implementation",)),
    ("7682ac51d13c29e9bf3f895686ea7b545e954941", ("PGIR210 merge",)),
    (TARGET, ("PGIR210 completion", "integrated outer target")),
    ("8d46a6d25dd006c8cab3c9d9612707d2a014e79c", ("campaign source_snapshots outer authority commit",)),
    (CAMPAIGN_OUTER_REVISION, ("campaign outer path-at-revision binding authority",)),
    (TOKENIZER_REVISION, ("campaign root and typed tokenizer-policy path-at-revision authority",)),
    ("8b42722897be2d2b88e416a40370c6a56b04bad8", ("PGIR001 executed task-identity board source",)),
    ("52756763d98e0a1b9ef40bcf842e41f79039886c", ("PGIR001 admitted task-identity board source",)),
    ("597a0285738c5878eed462593fd75e18715ff7f8", ("PGIR005 task-identity board source",)),
)

NESTED_CANDIDATES = (
    ("c30ccbec997868b061c4cadac38d30468c46ea2d", ("PGIR200 old gitlink", "nested chain root")),
    ("0566a833e795b0f0596251c2e7e8ca7d8ec27836", ("PGIR200 new gitlink", "PGIR201 old gitlink")),
    ("8cc72c77736d3ff2db7cc2530e619bf09b5be027", ("PGIR201 new gitlink", "PGIR202 old gitlink")),
    (SEALED, ("PGIR202 new gitlink", "sealed nested revision")),
    (CURRENT, ("PGIR204 nested revision", "integrated current gitlink")),
    (CAMPAIGN_NESTED_REVISION, ("campaign repository/dataset authority",)),
    ("978a4ef12d76860ca61feaecb8d9b5b2bf782d5d", ("campaign corpus implementation",)),
    ("21e1a2db5b5efe436ad450044fedffdc19b8a71f", ("campaign gap-matrix implementation",)),
    ("1f1aa38bdc769f414cc86417625f41871c317617", ("campaign schema-registry implementation",)),
    ("99717f2b7a7d2326064e3397d2442486778e13b1", ("campaign split implementation",)),
    ("df93e91e6338c84a17c3208ef68b88de8566f78c", ("campaign nested authority",)),
    (R1_REVISION, ("historical deterministic R1 path-at-revision authority",)),
    ("275807f04c9e4f6a8129fab639e8378c1168981c", ("R1 selected compiler adapter implementation",)),
    ("224a4f68d98b5bfb6bcdf619f001bdf4985591ac", ("R1 selected decompiler realizer implementation",)),
    ("2dca522faff7abe7e295f4c615ab3bf31f80dd4b", ("R1 replacement-matrix implementation",)),
)

OUTER_MISSING = tuple(sorted((CAMPAIGN_OUTER_REVISION, "597a0285738c5878eed462593fd75e18715ff7f8")))
NESTED_MISSING: tuple[str, ...] = ()
OUTER_REMOTE = "https://github.com/endomorphosis/ipfs_accelerate_py"
NESTED_REMOTE = "https://github.com/endomorphosis/ipfs_datasets_py.git"
VERIFIER_RELATIVE = "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py"
CAPTURE_RELATIVE = "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance/capture_evidence.py"
TASK_IDENTITY_SOURCE_REVISIONS = (
    ("8b42722897be2d2b88e416a40370c6a56b04bad8", ["PGIR-001 executed"]),
    ("52756763d98e0a1b9ef40bcf842e41f79039886c", ["PGIR-001 admitted"]),
    (CAMPAIGN_OUTER_REVISION, ["PGIR-002", "PGIR-003", "PGIR-014"]),
    ("597a0285738c5878eed462593fd75e18715ff7f8", ["PGIR-005"]),
)
SAFE_PYTHONPATH = "/home/barberb/.local/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages"
SAFE_SYS_PATH_TAIL = (
    "/home/barberb/.local/lib/python3.12/site-packages",
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
)
GIT_ENVIRONMENT_CONTROLS = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_COUNT": "0",
    "GIT_TERMINAL_PROMPT": "0",
}
BASE_SUBPROCESS_ENVIRONMENT = {
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
PYTHON_SUBPROCESS_ENVIRONMENT = {
    **BASE_SUBPROCESS_ENVIRONMENT,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONPATH": SAFE_PYTHONPATH,
}
EXECUTION_ENVIRONMENT_CONTROLS = {
    "environment_mode": "exact minimal environment; no inherited variables",
    "git": GIT_ENVIRONMENT_CONTROLS,
    "curl_configuration": "every curl argv begins /usr/bin/curl --disable",
    "home_and_xdg_variables_present": False,
    "proxy_and_credential_variables_present": False,
}


class EvidenceError(ValueError):
    """One immutable claim did not replay."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def portability_no_go_claim(missing_outer: Sequence[str], missing_nested: Sequence[str]) -> dict[str, Any]:
    """Derive the no-go and its first real blocker from exact missing sets."""
    outer = list(missing_outer)
    nested = list(missing_nested)
    require(outer == sorted(set(outer)) and set(outer).issubset({oid for oid, _roles in OUTER_CANDIDATES}), "outer portability missing population is invalid")
    require(nested == sorted(set(nested)) and set(nested).issubset({oid for oid, _roles in NESTED_CANDIDATES}), "nested portability missing population is invalid")
    require(bool(outer or nested), "portability_no_go requires at least one genuinely unreachable required candidate")
    blocker_repo, blocker_commit = ("outer", outer[0]) if outer else ("nested", nested[0])
    return {
        "status": "portability_no_go",
        "pgir_205_execution_authorized": False,
        "missing_outer_commits": outer,
        "missing_nested_commits": nested,
        "fresh_recursive_checkout": {
            "attempted": False,
            "blocker": {
                "repo": blocker_repo,
                "commit": blocker_commit,
                "reason": "required closure commit not reachable from any fetched advertised head or tag",
            },
        },
    }


def startup_environment_identity(expected_entrypoint_directory: Path) -> dict[str, Any]:
    require(sys.executable == "/usr/bin/python3.12", "verification did not use the canonical Python executable")
    require(sys.flags.no_site == 1 and sys.flags.no_user_site == 1, "verification requires Python -S and PYTHONNOUSERSITE=1")
    require(os.environ.get("PYTHONPATH") == SAFE_PYTHONPATH, "verification PYTHONPATH boundary drift")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1" and os.environ.get("PYTHONNOUSERSITE") == "1", "verification startup controls drift")
    require("PYTHONHOME" not in os.environ and "PYTEST_ADDOPTS" not in os.environ, "verification inherited prohibited Python/pytest controls")
    entrypoint = Path(sys.path[0] or os.getcwd()).resolve()
    require(entrypoint == expected_entrypoint_directory.resolve() and tuple(sys.path[1:]) == SAFE_SYS_PATH_TAIL, "verification exact sys.path boundary drift")
    meta_path = [f"{finder.__module__}.{finder.__name__}" for finder in sys.meta_path if isinstance(finder, type)]
    require(meta_path == ["_frozen_importlib.BuiltinImporter", "_frozen_importlib.FrozenImporter", "_frozen_importlib_external.PathFinder"], "verification meta_path contains an injected finder")
    forbidden_modules = [name for name in sys.modules if name in {"site", "sitecustomize", "usercustomize", "aae_mcplusplus_validators_bootstrap"} or name.startswith("__editable__")]
    require(not forbidden_modules, "verification startup loaded a site/bootstrap/editable hook")
    return {
        "executable": sys.executable,
        "no_site": True,
        "no_user_site": True,
        "pythonpath": SAFE_PYTHONPATH,
        "sys_path_tail": list(SAFE_SYS_PATH_TAIL),
        "meta_path": meta_path,
        "forbidden_modules_loaded": [],
        "entrypoint_directory": str(entrypoint),
        "shared_package_roots_are_raw_not_pth_processed": True,
        "hermetic": False,
    }


def expected_capture_startup_identity() -> dict[str, Any]:
    return {
        "executable": "/usr/bin/python3.12",
        "no_site": True,
        "no_user_site": True,
        "pythonpath": SAFE_PYTHONPATH,
        "sys_path_tail": list(SAFE_SYS_PATH_TAIL),
        "meta_path": ["_frozen_importlib.BuiltinImporter", "_frozen_importlib.FrozenImporter", "_frozen_importlib_external.PathFinder"],
        "forbidden_modules_loaded": [],
        "entrypoint_directory": str(Path(CAPTURE_RELATIVE).parent),
        "shared_package_roots_are_raw_not_pth_processed": True,
        "hermetic": False,
    }


def verify_capture_startup(receipt: Mapping[str, Any], label: str) -> None:
    require(receipt.get("capture_startup_environment") == expected_capture_startup_identity(), f"{label} capture startup boundary drift")


def test_toolchain_identity() -> dict[str, Any]:
    from importlib.metadata import distribution

    expected_versions = {
        "anyio": "4.14.2",
        "attrs": "26.1.0",
        "certifi": "2026.6.17",
        "charset-normalizer": "3.4.9",
        "filelock": "3.28.0",
        "fsspec": "2026.3.0",
        "hf-xet": "1.4.3",
        "huggingface-hub": "0.36.2",
        "idna": "3.18",
        "iniconfig": "2.3.0",
        "outcome": "1.3.0.post0",
        "packaging": "26.2",
        "pluggy": "1.6.0",
        "pygments": "2.17.2",
        "pytest": "9.1.1",
        "pytest-asyncio": "1.3.0",
        "pyyaml": "6.0.1",
        "requests": "2.34.2",
        "sniffio": "1.3.1",
        "sortedcontainers": "2.4.0",
        "tqdm": "4.67.3",
        "trio": "0.33.0",
        "typing_extensions": "4.16.0",
        "urllib3": "2.7.0",
    }
    closure_roles = {
        "anyio": ["pytest anyio plugin", "transitive async runtime"],
        "attrs": ["trio transitive runtime"],
        "certifi": ["requests TLS trust dependency"],
        "charset-normalizer": ["requests response-decoding dependency"],
        "filelock": ["huggingface-hub transitive runtime"],
        "fsspec": ["huggingface-hub transitive runtime"],
        "hf-xet": ["huggingface-hub transitive runtime"],
        "huggingface-hub": ["focused target test import"],
        "idna": ["anyio and requests transitive runtime"],
        "iniconfig": ["pytest configuration parser"],
        "outcome": ["trio transitive runtime"],
        "packaging": ["pytest and huggingface-hub transitive runtime"],
        "pluggy": ["pytest plugin manager"],
        "pygments": ["pytest terminal rendering"],
        "pytest": ["test runner"],
        "pytest-asyncio": ["explicit nested pytest configuration plugin"],
        "pyyaml": ["huggingface-hub transitive runtime"],
        "requests": ["huggingface-hub HTTP runtime"],
        "sniffio": ["anyio and trio transitive runtime"],
        "sortedcontainers": ["trio transitive runtime"],
        "tqdm": ["huggingface-hub transitive runtime"],
        "trio": ["focused target test async runtime"],
        "typing_extensions": ["anyio and huggingface-hub transitive runtime"],
        "urllib3": ["requests transport dependency"],
    }
    require(set(closure_roles) == set(expected_versions), "test toolchain closure-role population drift")
    no_record_roots = {"pygments": ("pygments",), "pyyaml": ("_yaml", "yaml")}
    shared_roots = tuple(Path(value).resolve() for value in SAFE_SYS_PATH_TAIL[:3])
    distributions: dict[str, Any] = {}
    all_mismatches = []
    for name, expected_version in expected_versions.items():
        package = distribution(name)
        require(package.version == expected_version, f"test harness distribution version drift: {name}")
        metadata_path = Path(package._path).resolve()
        containing_roots = [root for root in shared_roots if metadata_path.is_relative_to(root)]
        require(len(containing_roots) == 1, f"test harness metadata is outside the exact shared roots: {name}")
        shared_root = containing_roots[0]
        record_path = metadata_path / "RECORD"
        members = []
        hashed_members = 0
        record: dict[str, Any] | None
        if record_path.is_file():
            record_bytes = record_path.read_bytes()
            record = {"path": str(record_path), **identity(record_bytes)}
            for relative, declared_hash, declared_size in csv.reader(record_bytes.decode("utf-8").splitlines()):
                member_path = Path(package.locate_file(relative)).resolve()
                member_bytes = member_path.read_bytes()
                observed_hash = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(member_bytes).digest()).decode("ascii").rstrip("=")
                size_matches = not declared_size or int(declared_size) == len(member_bytes)
                hash_matches = not declared_hash or declared_hash == observed_hash
                if declared_hash:
                    algorithm, _encoded = declared_hash.split("=", 1)
                    require(algorithm == "sha256", f"unsupported distribution RECORD hash algorithm: {name}:{relative}")
                    hashed_members += 1
                if not (size_matches and hash_matches):
                    mismatch = {
                        "distribution": name,
                        "path": relative,
                        "declared_size": int(declared_size) if declared_size else None,
                        "observed_size": len(member_bytes),
                        "declared_hash": declared_hash or None,
                        "observed_hash": observed_hash,
                        "loaded_during_tests": name == "pytest-asyncio" and relative == "pytest_asyncio/plugin.py",
                        "test_relevant": name == "pytest-asyncio" and relative == "pytest_asyncio/plugin.py",
                        "integrity_disposition": (
                            "test_toolchain_integrity_no_go"
                            if name == "pytest-asyncio" and relative == "pytest_asyncio/plugin.py"
                            else "unused_closure_record_mismatch"
                        ),
                    }
                    all_mismatches.append(mismatch)
                members.append({"path": relative, **identity(member_bytes)})
            integrity_source = "wheel_record_plus_observed_member_manifest"
            top_level_paths: list[str] = []
        else:
            require(name in no_record_roots, f"unexpected distribution without RECORD: {name}")
            record = None
            top_level_paths = list(no_record_roots[name])
            tree_paths = [metadata_path, *(shared_root / value for value in top_level_paths)]
            for tree_path in tree_paths:
                require(tree_path.exists() and tree_path.is_relative_to(shared_root), f"unbound no-RECORD package root: {name}:{tree_path}")
                candidates = [tree_path] if tree_path.is_file() else sorted(path for path in tree_path.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
                for member_path in candidates:
                    members.append({"path": str(member_path.relative_to(shared_root)), **identity(member_path.read_bytes())})
            integrity_source = "observed_installed_tree_manifest_no_record_available"
        members.sort(key=lambda row: row["path"])
        distributions[name] = {
            "version": package.version,
            "metadata_path": str(metadata_path),
            "metadata_shared_root": str(shared_root),
            "integrity_source": integrity_source,
            "record": record,
            "top_level_paths": top_level_paths,
            "member_count": len(members),
            "record_hashed_member_count": hashed_members,
            "member_manifest": identity(canonical(members)),
        }
    all_mismatches.sort(key=lambda row: (row["distribution"], row["path"]))
    expected_mismatches = [
        {
            "distribution": "huggingface-hub",
            "path": "../../../bin/hf",
            "declared_size": 221,
            "observed_size": 325,
            "declared_hash": "sha256=j-MzP6VPofsu5aNDw0rkhpGH7p0glkFLMJdCeYXVdvs",
            "observed_hash": "sha256=5bSIHcQgVxzWgepj7HjmFjaEoWPlLsf7U2RwIpFqMqg",
            "loaded_during_tests": False,
            "test_relevant": False,
            "integrity_disposition": "unused_closure_record_mismatch",
        },
        {
            "distribution": "pytest-asyncio",
            "path": "pytest_asyncio/plugin.py",
            "declared_size": 29856,
            "observed_size": 38200,
            "declared_hash": "sha256=Ehy5jXRZap5dsyaaS-cTG67uqzdUtZlqmybCKJEIY6o",
            "observed_hash": "sha256=ukfCWkoisReJHpIUlAcpxBr5NtPa-7GLQoTeiojtM3U",
            "loaded_during_tests": True,
            "test_relevant": True,
            "integrity_disposition": "test_toolchain_integrity_no_go",
        },
    ]
    expected_mismatches.sort(key=lambda row: (row["distribution"], row["path"]))
    require(all_mismatches == expected_mismatches, "test toolchain exact RECORD mismatch population drift")
    return {
        "schema": "proof-grounded-ir-learning/python-test-toolchain/v1",
        "shared_package_roots": [str(root) for root in shared_roots],
        "pth_processing_enabled": False,
        "closure_roles": closure_roles,
        "distributions": distributions,
        "record_mismatches": all_mismatches,
        "integrity_status": "test_toolchain_integrity_no_go",
        "behavioral_test_results_authority": "observed_behavior_only",
    }


def validate(value: Any, where: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{where} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{where} has a non-string key")
        for key, item in value.items():
            validate(item, f"{where}.{key}")
        return
    raise EvidenceError(f"{where} has unsupported type {type(value).__name__}")


def canonical(value: Any) -> bytes:
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def raw_cid(data: bytes) -> str:
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + hashlib.sha256(data).digest()).decode("ascii").rstrip("=").lower()


def dag_cid(value: Any) -> str:
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + hashlib.sha256(canonical(value)).digest()).decode("ascii").rstrip("=").lower()


def identity(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "raw_cid": raw_cid(data)}


def parse_utc(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise EvidenceError(f"invalid UTC timestamp for {label}: {value}") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed), f"{label} is not UTC")
    return parsed


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def source_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(SOURCE_ROOT)),
        "prospective_git_blob": git_blob_oid(data),
        "containing_commit_claimed": False,
        **identity(data),
    }


def expected_sources() -> dict[str, Any]:
    return {
        "verifier": source_identity(SOURCE_ROOT / VERIFIER_RELATIVE),
        "capture": source_identity(SOURCE_ROOT / CAPTURE_RELATIVE),
    }


def configure_target_root(path: Path) -> None:
    global ROOT, DATASETS
    ROOT = path.resolve()
    DATASETS = ROOT / "ipfs_datasets_py"


def verify_execution(execution: Mapping[str, Any], *, stdout_text: bool = False, stderr_text: bool = False) -> None:
    require(execution.get("environment_controls") == EXECUTION_ENVIRONMENT_CONTROLS, "execution Git/curl environment-control drift")
    require(isinstance(execution.get("argv"), list) and all(isinstance(value, str) for value in execution["argv"]), "execution argv is not an exact string array")
    expected_environment = PYTHON_SUBPROCESS_ENVIRONMENT if execution["argv"][:1] == ["/usr/bin/python3.12"] else BASE_SUBPROCESS_ENVIRONMENT
    require(execution.get("environment") == expected_environment, "execution exact minimal environment drift")
    started = parse_utc(execution.get("started_at_utc"), "execution.started_at_utc")
    ended = parse_utc(execution.get("ended_at_utc"), "execution.ended_at_utc")
    require(started <= ended, "execution end precedes its start")
    target_time = datetime.fromisoformat(git(ROOT, "show", "-s", "--format=%cI", TARGET))
    require(started > target_time, "execution is not post-target evidence")
    require(isinstance(execution.get("exit_code"), int), "execution exit code is not an integer")
    for stream, require_text in (("stdout", stdout_text), ("stderr", stderr_text)):
        row = execution.get(stream)
        require(isinstance(row, dict) and set(identity(b"")).issubset(row), f"execution {stream} identity missing")
        if require_text:
            data = retained_stream_bytes(row, f"execution {stream}")
            require({key: row[key] for key in identity(b"")} == identity(data), f"execution {stream} retained-byte identity drift")


def retained_stream_bytes(row: Mapping[str, Any], label: str) -> bytes:
    if isinstance(row.get("utf8"), str):
        return row["utf8"].encode("utf-8")
    encoded = row.get("gzip_base64")
    require(isinstance(encoded, str), f"{label} bytes were not retained")
    try:
        return gzip.decompress(base64.b64decode(encoded, validate=True))
    except (ValueError, OSError) as exc:
        raise EvidenceError(f"invalid retained gzip stream for {label}: {exc}") from exc


def strict_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(EvidenceError(f"non-finite number {token!r} in {label}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON in {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must contain an object")
    validate(value, label)
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        return strict_json_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc


def run(repository: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.run(("/usr/bin/git", "-C", str(repository), *args), capture_output=True, timeout=timeout, env=BASE_SUBPROCESS_ENVIRONMENT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(f"git {' '.join(args)} unavailable: {exc}") from exc
    return process


def git(repository: Path, *args: str) -> str:
    process = run(repository, *args)
    require(process.returncode == 0, f"git {' '.join(args)} failed: {(process.stderr or process.stdout).decode(errors='replace').strip()}")
    return process.stdout.decode("utf-8").strip()


def git_blob(repository: Path, revision: str, path: str) -> tuple[str, bytes]:
    oid = git(repository, "rev-parse", f"{revision}:{path}")
    require(re.fullmatch(r"[0-9a-f]{40}", oid) is not None, f"invalid Git blob oid for {revision}:{path}")
    process = run(repository, "show", f"{revision}:{path}")
    require(process.returncode == 0, f"Git blob unavailable: {revision}:{path}")
    require(git(repository, "cat-file", "-t", oid) == "blob", f"Git identity is not a blob: {revision}:{path}")
    return oid, process.stdout


def git_json(repository: Path, revision: str, path: str) -> dict[str, Any]:
    _oid, data = git_blob(repository, revision, path)
    return strict_json_bytes(data, f"{revision}:{path}")


def gitlink(commit: str) -> str:
    fields = git(ROOT, "ls-tree", commit, "ipfs_datasets_py").split()
    require(len(fields) == 4 and fields[:2] == ["160000", "commit"], f"missing gitlink at {commit}")
    return fields[2]


def record_for(path: str, repository: Path, revision: str, relative: str | None = None) -> dict[str, Any]:
    oid, data = git_blob(repository, revision, relative or path)
    return {"path": path, "repository": "ipfs_datasets_py" if repository == DATASETS else "ipfs_accelerate_py", "revision": revision, "git_blob": oid, **identity(data)}


def verify_file_record(record: Mapping[str, Any], *, disk_required: bool = True) -> bytes:
    path = str(record["path"])
    repository = DATASETS if record["repository"] == "ipfs_datasets_py" else ROOT
    relative = path.removeprefix("ipfs_datasets_py/") if repository == DATASETS else path
    oid, data = git_blob(repository, str(record["revision"]), relative)
    require(record["git_blob"] == oid, f"Git blob drift: {path}")
    require({key: record[key] for key in ("size_bytes", "sha256", "raw_cid")} == identity(data), f"byte identity drift: {path}")
    if disk_required:
        require((ROOT / path).read_bytes() == data, f"working bytes differ from declared Git blob: {path}")
    return data


def expected_predecessor_records() -> list[dict[str, Any]]:
    result = []
    for task_id, revision, path in PREDECESSOR_ROWS:
        row = record_for(path, ROOT, revision)
        row["task_id"] = task_id
        result.append(row)
    return result


def expected_task_identity_source_records() -> list[dict[str, Any]]:
    rows = []
    for revision, task_ids in TASK_IDENTITY_SOURCE_REVISIONS:
        row = record_for("docs/architecture/proof_grounded_ir_learning.todo.md", ROOT, revision)
        row["task_identities"] = task_ids
        rows.append(row)
    return rows


def verify_acceptance(receipt: Mapping[str, Any]) -> None:
    verify_capture_startup(receipt, "acceptance")
    require(
        (receipt.get("schema"), receipt.get("task_id"), receipt.get("result_identity"), receipt.get("decision"))
        == ("proof-grounded-ir-learning/successor-integrated-acceptance/v2", "PGIR-211", "RESULT(PGIR-211)", "permanent_no_go"),
        "unsafe PGIR-211 identity or decision",
    )
    require(receipt.get("completion_authoritative") is False, "PGIR-211 asserted authoritative completion")
    require(receipt.get("pgir_205_execution_authorized") is False, "PGIR-211 authorized PGIR-205")
    require(receipt.get("containing_commit_claimed") is False, "PGIR-211 made a circular containing-commit claim")
    require(receipt.get("acceptance_identity_derivation") == "canonical DAG-JSON/SHA-256 of this document after omitting acceptance_sha256 and acceptance_cid; the RESULT(PGIR-211) supersession row uses a field-name pointer and contains no derived self value", "PGIR-211 acceptance identity derivation drift")
    projection = {key: value for key, value in receipt.items() if key not in {"acceptance_sha256", "acceptance_cid"}}
    require(receipt.get("acceptance_sha256") == "sha256:" + hashlib.sha256(canonical(projection)).hexdigest(), "PGIR-211 acceptance SHA drift")
    require(receipt.get("acceptance_cid") == dag_cid(projection), "PGIR-211 acceptance CID drift")
    require(receipt.get("verifier_source") == source_identity(Path(__file__)), "verifier source identity drift")
    require(receipt.get("supersession_chain") == [
        {"result_identity": "RESULT(PGIR-208)", "acceptance_cid": P208_CID},
        {"result_identity": "RESULT(PGIR-210)", "acceptance_cid": P210_CID},
        {"result_identity": "RESULT(PGIR-211)", "self_reference_field": "acceptance_cid"},
    ], "208 -> 210 -> 211 supersession chain drift")
    require(receipt.get("predecessor_acceptance_cids") == {"PGIR-208": P208_CID, "PGIR-209": P209_CID, "PGIR-210": P210_CID}, "predecessor CID closure drift")
    require(receipt.get("permanent_no_go_reason_codes") == [
        "remote_commit_population_incomplete",
        "historical_measured_adapter_cid_mismatch",
        "historical_policy_revision_opaque",
        "historical_dirty_snapshot_cid_unsealed_mismatch",
        "test_toolchain_loaded_record_mismatch",
        "test_toolchain_unused_console_record_mismatch",
        "zero_rights_admitted_materialized_rows",
        "tokenizer_not_admitted",
        "current_baseline_retired",
    ], "PGIR-211 permanent no-go reason population/order drift")
    require(receipt.get("unresolved_links") == [{
        "field": "campaign.bindings.policy.policy_revision",
        "value": "sha256:27c8da23ef92ab263ac0c144f2414fd40bdb30aace98b88f7dd76d36db26e142",
        "status": "opaque_non_authoritative_no_derivation_contract",
    }], "PGIR-211 exact unresolved-link adjudication drift")
    components = receipt.get("component_results")
    require(isinstance(components, dict), "PGIR-211 component-result aggregation missing")
    policy_opaque = receipt["unresolved_links"][0]
    measured_adapter = components["historical"]["campaign"]["semantics"]["measured_adapter_mismatch"]
    dirty_snapshot = components["historical"]["campaign"]["inventories"]["baseline_recursive"]["dirty_snapshot_mismatch"]
    require(receipt.get("historical_recursive_defects") == {"opaque_links": [policy_opaque], "mismatches": [measured_adapter, dirty_snapshot]}, "PGIR-211 authoritative historical defect aggregation drift")
    toolchain_mismatches = components["tests"]["toolchain_record_mismatches"]
    require(receipt.get("execution_environment_defects") == {
        "toolchain_integrity_status": "test_toolchain_integrity_no_go",
        "toolchain_record_mismatches": toolchain_mismatches,
        "test_results_authority": "observed_behavior_only",
    }, "PGIR-211 test-toolchain defect aggregation drift")
    require(receipt.get("target") == {"outer_commit": TARGET, "outer_tree": TARGET_TREE, "nested_gitlink": CURRENT, "nested_tree": CURRENT_TREE}, "integrated target drift")
    expected_names = {
        "README.md", "capture_evidence.py", "component_verification_receipt.json",
        "historical_closure_receipt.json", "network_receipt.json",
        "portability_receipt.json", "test_receipt.json",
    }
    closure = receipt.get("canonical_closure")
    require(isinstance(closure, dict) and set(closure) == expected_names, "canonical RESULT(PGIR-211) closure population drift")
    for name in sorted(expected_names):
        data = (DIR / name).read_bytes()
        require(closure[name] == {"path": name, **identity(data)}, f"canonical closure identity drift: {name}")


def verify_predecessors(closure: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = closure.get("predecessor_files")
    require(isinstance(records, list) and len(records) == 10, "exact ten-file predecessor closure is required")
    require([row.get("path") for row in records] == list(PREDECESSOR_FILES), "predecessor path population/order drift")
    require(records == expected_predecessor_records(), "predecessor Git/byte closure drift")
    for row in records:
        verify_file_record(row)
    pg208 = read_json(ROOT / PREDECESSOR_FILES[1])
    pg209 = read_json(ROOT / PREDECESSOR_FILES[3])
    pg210 = read_json(ROOT / PREDECESSOR_FILES[7])
    require(pg208.get("acceptance_cid") == P208_CID, "PGIR-208 acceptance CID drift")
    require(pg209.get("acceptance_cid") == P209_CID, "PGIR-209 acceptance CID drift")
    require(pg210.get("acceptance_cid") == P210_CID, "PGIR-210 acceptance CID drift")
    for task_id, document in (("PGIR-208", pg208), ("PGIR-209", pg209), ("PGIR-210", pg210)):
        projection = {key: value for key, value in document.items() if key not in {"acceptance_cid", "acceptance_sha256"}}
        require(document["acceptance_sha256"] == "sha256:" + hashlib.sha256(canonical(projection)).hexdigest() and document["acceptance_cid"] == dag_cid(projection), f"{task_id} canonical acceptance identities drift")
    return pg208, pg209, pg210


def expected_outer_forest() -> list[dict[str, Any]]:
    return [
        {"task_id": task, "role": role, "commit": commit, "tree": tree, "parents": list(parents), "gitlink": link}
        for task, role, commit, tree, parents, link in OUTER_FOREST
    ]


def verify_forest(closure: Mapping[str, Any], pg208: Mapping[str, Any], pg210: Mapping[str, Any]) -> None:
    forest = closure.get("integrated_forest")
    require(isinstance(forest, dict), "missing integrated forest")
    require(forest.get("target") == {"commit": TARGET, "tree": TARGET_TREE, "gitlink": CURRENT}, "forest target drift")
    require(forest.get("outer_commits") == expected_outer_forest(), "full PGIR-209/210 forest drift")
    for row in forest["outer_commits"]:
        commit = row["commit"]
        require(git(ROOT, "cat-file", "-t", commit) == "commit", f"outer object is not a commit: {commit}")
        require(git(ROOT, "rev-parse", f"{commit}^{{tree}}") == row["tree"], f"outer tree drift: {commit}")
        require(git(ROOT, "show", "-s", "--format=%P", commit).split() == row["parents"], f"outer parents drift: {commit}")
        require(gitlink(commit) == row["gitlink"], f"outer gitlink drift: {commit}")
        require(git(ROOT, "merge-base", "--is-ancestor", commit, TARGET) == "", f"outer forest commit is not in integrated target: {commit}")
    require(git(ROOT, "rev-parse", f"{TARGET}^{{tree}}") == TARGET_TREE and gitlink(TARGET) == CURRENT, "integrated target object drift")
    require(git(ROOT, "merge-base", "--is-ancestor", TARGET, "HEAD") == "", "current checkout does not descend from integrated target")
    require(git(DATASETS, "rev-parse", "HEAD") == CURRENT, "nested checkout commit drift")
    require(git(DATASETS, "rev-parse", "HEAD^{tree}") == CURRENT_TREE, "nested checkout tree drift")
    require(git(DATASETS, "status", "--porcelain", "--untracked-files=all") == "", "nested checkout is not clean")

    expected_cas = [
        {"task_id": task, "old_gitlink": old, "new_gitlink": new, "implementation": implementation, "merge": merge, "completion": completion}
        for task, old, new, implementation, merge, completion in CAS
    ]
    require(forest.get("compare_and_swap") == expected_cas, "three-task CAS population drift")
    source_tasks = pg210["forest"]["compare_and_swap"]["tasks"]
    require(len(source_tasks) == 3, "PGIR-210 CAS source population drift")
    for expected, source in zip(expected_cas, source_tasks, strict=True):
        require((source["task_id"], source["old_gitlink"], source["new_gitlink"]) == (expected["task_id"], expected["old_gitlink"], expected["new_gitlink"]), f"CAS identity drift: {expected['task_id']}")
        topology = CAS_TOPOLOGY[expected["task_id"]]
        require(gitlink(topology["prior"]) == expected["old_gitlink"], f"CAS prior-target old gitlink drift: {expected['task_id']}")
        require(git(DATASETS, "cat-file", "-t", expected["new_gitlink"]) == "commit", f"CAS new gitlink is not a commit: {expected['task_id']}")
        require(git(DATASETS, "show", "-s", "--format=%P", expected["new_gitlink"]).split() == [expected["old_gitlink"]], f"CAS nested sole parent drift: {expected['task_id']}")
        expected_parents = {
            "implementation": [topology["prior"]],
            "merge": [topology["prior"], expected["implementation"]],
            "completion": [expected["merge"]],
        }
        for role in ("implementation", "merge", "completion"):
            row = source[role]
            require(row["commit"] == expected[role], f"CAS outer {role} drift: {expected['task_id']}")
            require(row["tree"] == topology[f"{role}_tree"] == git(ROOT, "rev-parse", f"{row['commit']}^{{tree}}"), f"CAS outer tree drift: {row['commit']}")
            require(row["parents"] == expected_parents[role] == git(ROOT, "show", "-s", "--format=%P", row["commit"]).split(), f"CAS outer parent drift: {row['commit']}")
            require(gitlink(row["commit"]) == expected["new_gitlink"], f"CAS outer gitlink drift: {row['commit']}")
            require(git(ROOT, "merge-base", "--is-ancestor", row["commit"], expected["completion"]) == "", f"CAS role is not an ancestor of completion: {row['commit']}")
            require(git(ROOT, "merge-base", "--is-ancestor", row["commit"], TARGET) == "", f"CAS role is not an ancestor of integrated target: {row['commit']}")
        require(source["implementation"]["tree"] == source["merge"]["tree"], f"CAS implementation/merge tree inequality: {expected['task_id']}")

    sealed_records = closure.get("sealed_successor_files")
    require(isinstance(sealed_records, list) and [row.get("path") for row in sealed_records] == list(SEALED_PATHS), "exact unique fourteen-path sealed set drift")
    pg208_paths = [str(row["path"]).removeprefix("ipfs_datasets_py/") for row in pg208["payloads"]]
    require(len(pg208_paths) == len(set(pg208_paths)) == 14 and pg208_paths == list(SEALED_PATHS), "PGIR-208 payload set does not equal exact sealed set")
    require(git(DATASETS, "merge-base", "--is-ancestor", SEALED, CURRENT) == "", "sealed commit is not an ancestor of current nested commit")
    require(git(DATASETS, "show", "-s", "--format=%P", CURRENT).split() == [SEALED], "PGIR-204 nested commit is not the sole child of sealed PGIR-202")
    require(git(DATASETS, "rev-parse", f"{SEALED}^{{tree}}") == SEALED_TREE and git(DATASETS, "rev-parse", f"{CURRENT}^{{tree}}") == CURRENT_TREE, "sealed/current nested tree identity drift")
    for row in sealed_records:
        expected = record_for(row["path"], DATASETS, SEALED, row["path"])
        expected["current_git_blob"] = git(DATASETS, "rev-parse", f"{CURRENT}:{row['path']}")
        require(row == expected, f"sealed path identity drift: {row['path']}")
        _, old = git_blob(DATASETS, SEALED, row["path"])
        _, new = git_blob(DATASETS, CURRENT, row["path"])
        require(old == new and row["git_blob"] == row["current_git_blob"], f"PGIR-204 changed sealed bytes: {row['path']}")
    path_args = [row["path"] for row in sealed_records]
    require(git(DATASETS, "diff", "--name-status", SEALED, CURRENT, "--", *path_args) == "", "PGIR-204 sealed name/status diff is nonempty")
    require(git(DATASETS, "diff", "--binary", SEALED, CURRENT, "--", *path_args) == "", "PGIR-204 sealed byte diff is nonempty")


def import_r1_module() -> Any:
    sys.path.insert(0, str(DATASETS))
    path = DATASETS / "benchmarks/semantic_roundtrip/deterministic_baseline.py"
    spec = importlib.util.spec_from_file_location("pgir211_r1_replay", path)
    require(spec is not None and spec.loader is not None, "cannot load deterministic R1 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def without_latencies(case: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(case))
    value["compiler"].pop("latency_seconds", None)
    value["decompiler"].pop("latency_seconds", None)
    return value


def replay_r1_cases() -> dict[str, Any]:
    module = import_r1_module()
    frozen = read_json(DATASETS / "data/ir_learning/evaluations/deterministic/r1_baseline.json")
    records = module.load_measured_fixture_cases()
    require(len(records) == 13, "R1 measured fixture population is not 13")
    compiler = module.TypedDeonticCanonicalCompiler()
    decompiler = module.SourceWithheldCanonicalDecompiler()
    actual = [module._compact_case(module._measure_case(row, compiler=compiler, decompiler=decompiler)) for row in records]
    require([row["case_id"] for row in actual] == [row["case_id"] for row in frozen["cases"]], "R1 case order/population drift")
    for observed, expected in zip(actual, frozen["cases"], strict=True):
        require(without_latencies(observed) == without_latencies(expected), f"deterministic R1 case replay drift: {expected['case_id']}")
    require(frozen["case_count"] == 13, "frozen R1 case count drift")
    require(frozen["populations"] == {
        "pilot": {"case_count": 5, "case_ids": [row["case_id"] for row in frozen["cases"][:5]]},
        "repair_development": {"case_count": 8, "case_ids": [row["case_id"] for row in frozen["cases"][5:]]},
    }, "R1 population split drift")
    require(frozen["independent_replay"] == {"compiler_all_matched": True, "decompiler_matched_count": 13, "decompiler_replay_denominator": 13, "formal_proof_replayed": False, "formal_proof_reason": "frozen fixtures carry no independently checkable proof obligations"}, "R1 replay summary drift")
    return {
        "case_count": 13,
        "case_ids": [row["case_id"] for row in frozen["cases"]],
        "compared_fields": "all compact deterministic case fields excluding only compiler.latency_seconds and decompiler.latency_seconds",
        "hidden_fixture_loaded": False,
        "compiler_all_matched": True,
        "decompiler_all_matched": True,
    }


def verify_r1_loaded_sources(recipe: Mapping[str, Any]) -> dict[str, Any]:
    replay_paths = ["benchmarks/semantic_roundtrip/deterministic_baseline.py", *module_implementation_paths().values(), *recipe["fixture_paths"].values()]
    fixture_blobs = {
        str(recipe["fixture_paths"]["pilot"]): "2ee2d28d09d06f6538154045182018bd0e2826e1",
        str(recipe["fixture_paths"]["repair_development"]): "bb626a456605acc992b4a816fdcf22acc224005d",
    }
    records = []
    for relative in replay_paths:
        r1_oid, r1_bytes = git_blob(DATASETS, R1_REVISION, relative)
        current_oid, current_bytes = git_blob(DATASETS, CURRENT, relative)
        if relative in fixture_blobs:
            require(r1_oid == fixture_blobs[relative], f"R1 exact fixture Git blob drift: {relative}")
        require(r1_oid == current_oid and r1_bytes == current_bytes, f"R1/current replay source blob drift: {relative}")
        require((DATASETS / relative).read_bytes() == r1_bytes, f"loaded replay source does not equal R1 Git bytes: {relative}")
        records.append({"path": "ipfs_datasets_py/" + relative, "r1_git_blob": r1_oid, "current_git_blob": current_oid, **identity(r1_bytes)})
    holdout = "tests/fixtures/semantic_roundtrip/holdout_cases.json"
    r1_holdout_oid = git(DATASETS, "rev-parse", f"{R1_REVISION}:{holdout}")
    current_holdout_oid = git(DATASETS, "rev-parse", f"{CURRENT}:{holdout}")
    require(r1_holdout_oid == current_holdout_oid and (DATASETS / holdout).is_file(), "R1 holdout presence/blob equality drift")
    return {"loaded_records": records, "holdout": {"path": "ipfs_datasets_py/" + holdout, "r1_git_blob": r1_holdout_oid, "current_git_blob": current_holdout_oid, "bytes_read": False}}


def verify_r1_capabilities(identities: Mapping[str, Any]) -> dict[str, Any]:
    path = "workspace/benchmarks/semantic-roundtrip-compositions/replacement_capabilities.json"
    oid, data = git_blob(DATASETS, R1_REVISION, path)
    require(oid == "2841705347a2cfb8fd0e74a74ba3c6d9425120e4", "R1 replacement-capability Git blob drift")
    replacement = strict_json_bytes(data, "R1 replacement capabilities")
    arm_id = "typed_deontic__no_guidance__no_repair__not_applicable__deterministic"
    selected = [row for row in replacement["plan"]["arms"] if row.get("cell_id") == arm_id]
    require(len(selected) == 1, "R1 selected arm population drift")
    arm = selected[0]
    arm_projection = {key: arm[key] for key in ("cell_id", "composition", "realizer", "adapter_bindings", "capability_record_cids", "route_requirements")}
    arm_cid = dag_cid(arm_projection)
    require(arm["arm_identity_cid"] == identities["compiler"]["configuration"]["selection"]["arm_identity_cid"] == arm_cid == "baguqeeraylvbngffosmvcvwowelspcdbbk5wom5itjvfanbzty4eioxsauhq", "R1 selected arm identity drift")
    capability_cids: dict[str, str] = {}
    base_fields = {"checks", "effective_identity", "id", "reason", "requested_identity", "status", "substitute_identity", "substitute_used"}
    for capability_id in ("python", "multiformats", "hammer_cvc5", "lean"):
        row = replacement["bindings"]["capabilities"][capability_id]
        require(set(row) == base_fields | {"requested_identity_cid", "effective_identity_cid", "record_cid"}, f"R1 capability record shape drift: {capability_id}")
        base = {key: row[key] for key in base_fields}
        require(row["id"] == capability_id and row["requested_identity_cid"] == dag_cid(row["requested_identity"]), f"R1 requested capability identity drift: {capability_id}")
        expected_effective = None if row["effective_identity"] is None else dag_cid(row["effective_identity"])
        require(row["effective_identity_cid"] == expected_effective and row["record_cid"] == dag_cid(base), f"R1 effective/record capability identity drift: {capability_id}")
        require(arm["capability_record_cids"][capability_id] == row["record_cid"], f"R1 arm capability link drift: {capability_id}")
        capability_cids[capability_id] = row["record_cid"]
    implementations = {
        "typed_deontic": ("275807f04c9e4f6a8129fab639e8378c1168981c", "benchmarks/semantic_roundtrip/constructors/typed_deontic.py", "bafkreig2yeibug44tbffleyvju4zvo62thdqkpht3n2qn6guefkvbv7z2a"),
        "source_withheld_paraphrase": ("224a4f68d98b5bfb6bcdf619f001bdf4985591ac", "benchmarks/semantic_roundtrip/realizers/source_withheld_paraphrase.py", "bafkreifrmafgdy5wajq7sepxxatwc2mnnubqt2c7kwped456vukyptfi6y"),
        "coordinate_runner": ("2dca522faff7abe7e295f4c615ab3bf31f80dd4b", "benchmarks/semantic_roundtrip/replacement_matrix.py", "bafkreigxa4dwxkqipox36emw7m3qyd7axxb3cqwfv4ckiniewtw5ftix6u"),
    }
    implementation_records = []
    for adapter_id, (revision, relative, expected_raw) in implementations.items():
        implementation_oid, implementation = git_blob(DATASETS, revision, relative)
        require(raw_cid(implementation) == arm["adapter_bindings"][adapter_id]["raw_cid"] == expected_raw, f"R1 selected adapter implementation identity drift: {adapter_id}")
        implementation_records.append({"adapter_id": adapter_id, "revision": revision, "path": "ipfs_datasets_py/" + relative, "git_blob": implementation_oid, **identity(implementation)})
    require(identities["compiler"]["configuration"]["constructor"]["adapter_raw_cid"] == implementations["typed_deontic"][2], "R1 compiler constructor/arm adapter link drift")
    return {"replacement_capabilities": {"path": "ipfs_datasets_py/" + path, "revision": R1_REVISION, "git_blob": oid, **identity(data)}, "arm_identity_cid": arm_cid, "capability_record_cids": capability_cids, "selected_implementations": implementation_records}


def verify_source_lineage_contract_equivalence() -> dict[str, Any]:
    """Bind the historical contract and prove the current-only export is irrelevant."""
    relative = SOURCE_LINEAGE_PATH
    r1_oid, r1_bytes = git_blob(DATASETS, R1_REVISION, relative)
    current_oid, current_bytes = git_blob(DATASETS, CURRENT, relative)
    require(
        (r1_oid, current_oid) == (SOURCE_LINEAGE_R1_BLOB, SOURCE_LINEAGE_CURRENT_BLOB),
        "source-lineage historical/current Git blob drift",
    )
    require(
        current_bytes == r1_bytes + SOURCE_LINEAGE_CURRENT_ADDITIVE_SUFFIX,
        "source-lineage current change is not the exact additive __all__ export",
    )
    require(
        (DATASETS / relative).read_bytes() == current_bytes,
        "loaded source-lineage bytes do not equal the bound current Git blob",
    )
    relevant_projection = identity(r1_bytes)
    return {
        "path": "ipfs_datasets_py/" + relative,
        "historical": {"revision": R1_REVISION, "git_blob": r1_oid, **identity(r1_bytes)},
        "current": {"revision": CURRENT, "git_blob": current_oid, **identity(current_bytes)},
        "current_additive_suffix": {
            "classification": "module_export_list_only",
            **identity(SOURCE_LINEAGE_CURRENT_ADDITIVE_SUFFIX),
        },
        "relevant_projection": {
            "definition": "exact historical module bytes; current bytes after removing the bound additive __all__ suffix",
            "historical": relevant_projection,
            "current_without_additive_export": relevant_projection,
            "equal": True,
        },
    }


def independent_corpus_manifest_cid(manifest: Mapping[str, Any]) -> str:
    """Replay CorpusManifest.record_cid from the historical contract projection."""
    expected_fields = {
        "derived_artifact_ids", "derived_count", "kind", "lineage_graph_id",
        "manifest_id", "record_cid", "rights", "schema_version", "source_count",
        "source_record_ids",
    }
    require(set(manifest) == expected_fields, "R1 corpus manifest field population drift")
    sources = manifest["source_record_ids"]
    derived = manifest["derived_artifact_ids"]
    require(isinstance(sources, list) and isinstance(derived, list) and sources and len(sources) == len(set(sources)), "R1 corpus manifest source population is invalid")
    require(len(derived) == len(set(derived)) and not set(sources).intersection(derived), "R1 corpus manifest derived population is invalid")
    require(
        manifest["kind"] == "corpus_manifest"
        and manifest["schema_version"] == "ir-corpus-manifest/v1"
        and manifest["source_count"] == len(sources)
        and manifest["derived_count"] == len(derived),
        "R1 corpus manifest kind/schema/count drift",
    )
    payload = {
        "derived_artifact_ids": derived,
        "lineage_graph_id": manifest["lineage_graph_id"],
        "manifest_id": manifest["manifest_id"],
        "rights": manifest["rights"],
        "schema_version": manifest["schema_version"],
        "source_record_ids": sources,
    }
    preimage = canonical({
        "canonicalization": "ir-canonical-json-v1",
        "collection_semantics": {},
        "domain": "ir.source-lineage",
        "identity_profile": "ir-canonical-identity-v1",
        "payload": payload,
        "schema_version": manifest["schema_version"],
    })
    return raw_cid(preimage)


def verify_r1_corpus_split(identities: Mapping[str, Any]) -> dict[str, Any]:
    corpus_root_path = "data/ir_learning/corpora/corpus_root.json"
    corpus_oid, corpus_bytes = git_blob(DATASETS, R1_REVISION, corpus_root_path)
    require(corpus_oid == "4ed0d041d0bbd101abc0a7d53f1ec6e8aabe6df5", "R1 corpus-root blob drift")
    corpus_root = strict_json_bytes(corpus_bytes, "R1 corpus root")
    require(identities["corpus"]["path"] == corpus_root_path and identities["corpus"]["root_sha256"] == identity(corpus_bytes)["sha256"], "R1 corpus root path/SHA link drift")
    require((identities["corpus"]["source_count"], identities["corpus"]["derived_count"], identities["corpus"]["training_admitted_rows"], identities["corpus"]["materialized"]) == (corpus_root["source_count"], corpus_root["derived_count"], corpus_root["training_admitted_rows"], corpus_root["materialized"]), "R1 corpus root population link drift")
    for name, artifact in corpus_root["artifacts"].items():
        artifact_oid, artifact_bytes = git_blob(DATASETS, R1_REVISION, f"data/ir_learning/corpora/{name}")
        campaign_oid, campaign_bytes = git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, f"data/ir_learning/corpora/{name}")
        require(artifact_oid == campaign_oid and artifact_bytes == campaign_bytes, f"R1/campaign corpus artifact drift: {name}")
        require(artifact == {"path": name, "sha256": hashlib.sha256(artifact_bytes).hexdigest(), "content_cid": raw_cid(artifact_bytes), "size_bytes": len(artifact_bytes)}, f"R1 corpus artifact identity drift: {name}")
    manifest_oid, manifest_bytes = git_blob(DATASETS, R1_REVISION, "data/ir_learning/corpora/corpus_manifest.json")
    lineage_oid, lineage_bytes = git_blob(DATASETS, R1_REVISION, "data/ir_learning/corpora/lineage_graph.json")
    require(manifest_oid == "e8c227b6cadd57e4dd788caeabea5e927fb4367d" and lineage_oid == "918a3d24eb3300c0d3ea156ab3c9a928942cee0d", "R1 corpus manifest/lineage blob drift")
    manifest = strict_json_bytes(manifest_bytes, "R1 corpus manifest")
    lineage = strict_json_bytes(lineage_bytes, "R1 lineage graph")
    source_contract = verify_source_lineage_contract_equivalence()
    manifest_cid = independent_corpus_manifest_cid(manifest)
    lineage_cid = raw_cid(canonical(lineage))
    require(corpus_root["manifest_cid"] == identities["corpus"]["manifest_cid"] == manifest["record_cid"] == manifest_cid == "bafkreiha35x7mcukzzb5x67hmykwsny5wipf5jb4do5gpsl24mxvix55n4", "R1 corpus manifest CID drift")
    require(corpus_root["lineage_graph_cid"] == identities["corpus"]["lineage_graph_cid"] == lineage_cid == "bafkreia5jirpcpummrddhczxebz554lkd7wrq4o5ynizjlgbczyzuwhakq", "R1 lineage graph CID drift")

    split_root_path = "data/ir_learning/splits/split_root.json"
    split_oid, split_bytes = git_blob(DATASETS, R1_REVISION, split_root_path)
    _campaign_split_oid, campaign_split_bytes = git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, split_root_path)
    require(split_bytes == campaign_split_bytes and identities["split"]["path"] == split_root_path and identities["split"]["root_sha256"] == identity(split_bytes)["sha256"], "R1 split-root path/SHA/campaign equality drift")
    split_root = strict_json_bytes(split_bytes, "R1 split root")
    manifest_path = str(identities["split"]["manifest_path"])
    split_manifest_oid, split_manifest_bytes = git_blob(DATASETS, R1_REVISION, manifest_path)
    _campaign_manifest_oid, campaign_manifest_bytes = git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, manifest_path)
    require(split_manifest_bytes == campaign_manifest_bytes and identities["split"]["manifest_size_bytes"] == len(split_manifest_bytes) and identities["split"]["split_manifest_sha256"] == hashlib.sha256(split_manifest_bytes).hexdigest(), "R1 split manifest byte identity drift")
    split_manifest = strict_json_bytes(split_manifest_bytes, "R1 split manifest")
    require(split_root["split_manifest_sha256"] == identities["split"]["split_manifest_sha256"] and split_root["split_manifest_digest"] == identities["split"]["split_manifest_digest"] == split_manifest["split_manifest_digest"], "R1 split root/manifest digest link drift")
    require(split_root["hidden_test_commitment"] == identities["split"]["hidden_test_commitment"] == HIDDEN and split_root["leakage_passed"] is identities["split"]["leakage_passed"] is True, "R1 split hidden/leakage link drift")
    return {
        "corpus_root": {"path": "ipfs_datasets_py/" + corpus_root_path, "revision": R1_REVISION, "git_blob": corpus_oid, **identity(corpus_bytes)},
        "manifest_cid": manifest_cid,
        "lineage_graph_cid": lineage_cid,
        "source_contract_equivalence": source_contract,
        "split_root": {"path": "ipfs_datasets_py/" + split_root_path, "revision": R1_REVISION, "git_blob": split_oid, **identity(split_bytes)},
        "split_manifest": {"path": "ipfs_datasets_py/" + manifest_path, "revision": R1_REVISION, "git_blob": split_manifest_oid, **identity(split_manifest_bytes)},
    }


def verify_r1(closure: Mapping[str, Any], *, replay: bool = True) -> dict[str, Any]:
    records = closure.get("r1_files")
    expected_paths = [f"ipfs_datasets_py/data/ir_learning/evaluations/deterministic/{name}" for name, _ in R1_FILES]
    require(isinstance(records, list) and [row.get("path") for row in records] == expected_paths, "true seven-file R1 population drift")
    documents: dict[str, dict[str, Any]] = {}
    for row, (name, self_field) in zip(records, R1_FILES, strict=True):
        expected = record_for(expected_paths[len(documents)], DATASETS, R1_REVISION, f"data/ir_learning/evaluations/deterministic/{name}")
        expected["current_git_blob"] = git(DATASETS, "rev-parse", f"{CURRENT}:data/ir_learning/evaluations/deterministic/{name}")
        require(row == expected, f"R1 path/Git/byte identity drift: {name}")
        data = verify_file_record(row)
        require(row["git_blob"] == row["current_git_blob"], f"R1 bytes changed between 9313 and current: {name}")
        document = strict_json_bytes(data, name)
        projection = dict(document)
        claimed = projection.pop(self_field)
        require(claimed == dag_cid(projection), f"R1 self CID drift: {name}")
        documents[name] = document
    manifest = documents["manifest.json"]
    require(set(manifest["files"]) == {name for name, _ in R1_FILES if name != "manifest.json"}, "R1 manifest exact six-payload population drift")
    for name, self_field in R1_FILES:
        if name == "manifest.json":
            continue
        entry = manifest["files"][name]
        row = next(item for item in records if item["path"].endswith("/" + name))
        require(entry == {"path": f"data/ir_learning/evaluations/deterministic/{name}", "sha256": row["sha256"], "cid": documents[name][self_field]}, f"R1 manifest link drift: {name}")
    require(manifest["report_cid"] == documents["r1_baseline.json"]["report_cid"], "R1 manifest/report link drift")
    identities = documents["identities.json"]
    recipe = documents["recipe.json"]
    report = documents["r1_baseline.json"]
    catalog = documents["metric_catalog.json"]
    strata = documents["strata.json"]
    tools = documents["tool_versions.json"]
    require(report["identities_cid"] == recipe["identities_cid"] == identities["identities_cid"], "R1 report/recipe identities link drift")
    require(report["catalog_cid"] == catalog["catalog_cid"] and report["recipe_cid"] == recipe["recipe_cid"] and report["strata_cid"] == strata["strata_cid"] and report["tool_versions_cid"] == tools["tool_versions_cid"], "R1 report recursive document links drift")
    require(recipe["metrics"] == [row["metric_id"] for row in catalog["metrics"]] and recipe["surfaces"] == catalog["surfaces"] and recipe["missing_metric_as_zero"] == catalog["missing_as_zero"] is False, "R1 recipe/catalog semantic link drift")
    require(report["experiment_id"] == recipe["experiment_id"] == strata["experiment_id"] == identities["experiment_id"] == "R1", "R1 experiment identity drift")
    require(recipe["fixture_cids"] == {"pilot": "bafkreidngtg5cojnhkmwj4coijqpoixao25hxfwdzxjpywlusrqhk3hrm4", "repair_development": "bafkreickaddpda2fuwh2p675vg6vw2jpliixhhrhanz2yk72hyqcol5zfu"}, "R1 fixture CID population drift")
    for population, relative in recipe["fixture_paths"].items():
        _oid, data = git_blob(DATASETS, R1_REVISION, relative)
        require(raw_cid(data) == recipe["fixture_cids"][population], f"R1 fixture raw CID drift: {population}")
    require((DATASETS / "tests/fixtures/semantic_roundtrip/holdout_cases.json").is_file(), "R1 holdout presence drift")
    require(recipe["hidden_test_selection"] is False and report["hidden_test_selection"] is False and report["hidden_labels_opened"] is False, "R1 hidden-input boundary drift")
    require(recipe["learned_inference"] is False and report["learned_inference"] is False, "R1 learned inference drift")
    require(recipe["hidden_test_commitment"] == report["hidden_test_commitment"] == identities["split"]["hidden_test_commitment"] == HIDDEN, "R1 hidden commitment drift")
    require(report["corpus_manifest_cid"] == recipe["corpus_manifest_cid"] == identities["corpus"]["manifest_cid"], "R1 corpus manifest link drift")
    require(report["split_manifest_digest"] == recipe["split_manifest_digest"] == identities["split"]["split_manifest_digest"], "R1 split manifest digest link drift")
    loaded_sources = verify_r1_loaded_sources(recipe)
    capabilities = verify_r1_capabilities(identities)
    corpus_split = verify_r1_corpus_split(identities)
    module = import_r1_module()
    for name, relative in module_implementation_paths().items():
        _oid, implementation = git_blob(DATASETS, R1_REVISION, relative)
        require(raw_cid(implementation) == identities["implementation_raw_cids"][name] == report["implementation_raw_cids"][name], f"R1 implementation raw CID drift: {name}")
    require(recipe["compiler_configuration_cid"] == report["compiler_configuration_cid"] == identities["compiler"]["configuration_cid"], "R1 compiler configuration link drift")
    require(recipe["decompiler_configuration_cid"] == report["decompiler_configuration_cid"] == identities["decompiler"]["configuration_cid"], "R1 decompiler configuration link drift")
    require(recipe["roundtrip_configuration_cid"] == report["roundtrip_configuration_cid"] == identities["roundtrip"]["configuration_cid"], "R1 roundtrip configuration link drift")
    require(report["policy_cid"] == identities["roundtrip"]["policy_cid"], "R1 policy link drift")
    require(dag_cid(identities["compiler"]["configuration"]) == identities["compiler"]["configuration_cid"] == module.TYPED_DEONTIC_COMPILER_CONFIG_CID, "R1 compiler configuration recomputation drift")
    require(dag_cid(module.compiler_configuration()) == module.TYPED_DEONTIC_COMPILER_CONFIG_CID, "current compiler configuration payload drift")
    from ipfs_datasets_py.logic.legal_ir import canonical_contracts, canonical_decompiler, canonical_roundtrip
    require(dag_cid(canonical_decompiler.frozen_decompiler_config()) == identities["decompiler"]["configuration_cid"] == module.SOURCE_WITHHELD_DECOMPILER_CONFIG_CID, "R1 decompiler configuration recomputation drift")
    require(dag_cid(canonical_decompiler._rendering_spec_payload()) == identities["decompiler"]["rendering_spec_cid"] == module.SOURCE_WITHHELD_RENDERING_SPEC_CID, "R1 decompiler rendering specification drift")
    require(dag_cid(canonical_roundtrip.roundtrip_configuration()) == identities["roundtrip"]["configuration_cid"] == module.CANONICAL_SEMANTIC_ROUNDTRIP_CONFIG_CID, "R1 roundtrip configuration recomputation drift")
    parity_policy = canonical_contracts.load_parity_policy()
    require(parity_policy.policy_cid == identities["roundtrip"]["policy_cid"] == module.CANONICAL_PARITY_POLICY_CID, "R1 parity policy recomputation drift")
    replay_result = replay_r1_cases() if replay else {"case_count": 13, "replay_deferred": True}
    return {**replay_result, "loaded_sources": loaded_sources, "capabilities": capabilities, "corpus_split": corpus_split}


def module_implementation_paths() -> dict[str, str]:
    return {
        "compiler": "ipfs_datasets_py/logic/legal_ir/canonical_compiler.py",
        "decompiler": "ipfs_datasets_py/logic/legal_ir/canonical_decompiler.py",
        "roundtrip": "ipfs_datasets_py/logic/legal_ir/canonical_roundtrip.py",
        "contracts": "ipfs_datasets_py/logic/legal_ir/canonical_contracts.py",
    }


def campaign_record(record: Mapping[str, Any], binding_name: str, index: int) -> dict[str, Any]:
    path = str(record["path"])
    repository_name = str(record["repository"])
    if repository_name == "ipfs_datasets_py":
        repository, revision, relative = DATASETS, CAMPAIGN_NESTED_REVISION, path.removeprefix("ipfs_datasets_py/")
        method = "declared_path_at_campaign_nested_revision"
    elif repository_name == "ipfs_accelerate_py":
        repository, revision, relative = ROOT, CAMPAIGN_OUTER_REVISION, path
        method = "declared_path_at_campaign_outer_revision"
    else:
        require(repository_name == "pgir-freeze" and path == "data/agent_supervisor/proof_grounded_ir_learning/freeze/tokenizer_policy.json", f"untyped campaign file exception: {path}")
        repository, revision, relative = ROOT, TOKENIZER_REVISION, path
        method = "typed_tokenizer_policy_exception_at_freeze_commit"
    oid, data = git_blob(repository, revision, relative)
    if method == "typed_tokenizer_policy_exception_at_freeze_commit":
        require("git_blob" not in record and oid == TOKENIZER_BLOB, "tokenizer exception identity drift")
    else:
        require(record.get("git_blob") == oid, f"campaign declared path-at-revision blob drift: {path}")
    require(identity(data) == {key: record[key] for key in ("size_bytes", "sha256", "raw_cid")}, f"campaign file identity drift: {path}")
    return {
        "binding": binding_name,
        "occurrence_index": index,
        "path": path,
        "repository": repository_name,
        "revision": revision,
        "declared_git_blob": record.get("git_blob"),
        "resolved_git_blob": oid,
        "identity_method": method,
        **identity(data),
    }


def expected_campaign_bindings(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(CAMPAIGN_COUNTS):
        binding = campaign["bindings"][name]
        records = ([binding["file"]] if "file" in binding else []) + list(binding.get("files", []))
        rows.append({
            "binding": name,
            "binding_cid": binding["binding_cid"],
            "record_count": len(records),
            "records": [campaign_record(record, name, index) for index, record in enumerate(records)],
        })
    return rows


def projection_cid(document: Mapping[str, Any], field: str) -> str:
    projection = dict(document)
    claimed = projection.pop(field)
    require(claimed == dag_cid(projection), f"projection CID drift: {field}")
    return str(claimed)


def projection_sha(document: Mapping[str, Any], field: str, *, prefixed: bool = False) -> str:
    projection = dict(document)
    claimed = str(projection.pop(field))
    expected = hashlib.sha256(canonical(projection)).hexdigest()
    require(claimed == (("sha256:" + expected) if prefixed else expected), f"projection SHA drift: {field}")
    return claimed


def verify_dag_projection_identity(document: Mapping[str, Any], *, cid_field: str, identity_field: str, label: str) -> str:
    projection = dict(document)
    claimed_cid = str(projection.pop(cid_field))
    claimed_identity = str(projection.pop(identity_field))
    require(claimed_cid == dag_cid(projection) and claimed_identity == "sha256:" + hashlib.sha256(canonical(projection)).hexdigest(), f"{label} canonical CID/SHA projection drift")
    return claimed_cid


def verify_historical_baseline(
    baseline: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    supervisor: Mapping[str, Any],
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = baseline["evidence_bundle"]
    expected_log_cids = {
        "artifact_validation_stdout": "bafkreia65aaygzsczs5ctpyesb4wojblgyzqni52u4dw6eqqtd7yulqjpy",
        "accelerator_stdout": "bafkreiae2epk3af6dm5ptcogewgv6g7qnrasyhcwynbqhibjrzgmtreloe",
        "datasets_stdout": "bafkreiajcywqzlfteswjan7pgrd2lxng25r6tla36tcpp2bvndwhoqlnkq",
        "accelerator_source_binding_stdout": "bafkreiat3vv6yusciqefpu3acgt44vamrjbywm2546tjc735bzdpiad7xe",
        "focused_validation_stdout": "bafkreicfrlxpcflwh443hjhthzoslxv4mmslxhjocqlskd6vy7x3dwrrum",
        "datasets_source_binding_stdout": "bafkreicv5bsbwccxcj5opuvk75hlu533224qhbhy7g7ipiruooihahqv7y",
    }
    logs = bundle["embedded_logs"]
    require(set(logs) == set(expected_log_cids), "historical baseline embedded-log population drift")
    decoded_logs: dict[str, bytes] = {}
    log_refs: dict[str, dict[str, str]] = {}
    for log_id, expected_cid in expected_log_cids.items():
        row = logs[log_id]
        require(row["encoding"] == "base64(gzip -n -9(exact stdout bytes))", f"historical baseline embedded-log encoding drift: {log_id}")
        try:
            decoded = gzip.decompress(base64.b64decode(row["payload"], validate=True))
        except (ValueError, OSError) as exc:
            raise EvidenceError(f"invalid historical baseline embedded log {log_id}: {exc}") from exc
        require((row["uncompressed_byte_count"], row["sha256"], row["cid"]) == (len(decoded), hashlib.sha256(decoded).hexdigest(), raw_cid(decoded)) and row["cid"] == expected_cid, f"historical baseline embedded-log byte identity drift: {log_id}")
        decoded_logs[log_id] = decoded
        log_refs[log_id] = {"embedded_log_id": log_id, "sha256": row["sha256"], "cid": row["cid"]}

    test_receipts = bundle["test_receipts"]
    test_categories = ["accelerator", "datasets", "focused-validation"]
    test_cids = [
        "baguqeeray6vfewxu4uhzjttspwrmeshmswm3gcfqknvhpwtkvveb2ahdud4a",
        "baguqeera3exv5nz4a3h2lqoic2v3grbdm73mbpheelu73xztazpwzbnf554q",
        "baguqeera2x4rdczh5bft26uv6c7p7nhvfaxsfppiohc3tor6rih6flsz3byq",
    ]
    test_log_ids = ["accelerator_stdout", "datasets_stdout", "focused_validation_stdout"]
    require([row["category"] for row in test_receipts] == test_categories, "historical baseline test-receipt category/order drift")
    empty_stderr = {"byte_count": 0, "sha256": hashlib.sha256(b"").hexdigest(), "cid": "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"}
    for row, expected_cid, log_id in zip(test_receipts, test_cids, test_log_ids, strict=True):
        require(verify_dag_projection_identity(row, cid_field="receipt_cid", identity_field="receipt_identity", label=f"historical baseline test {row['category']}") == expected_cid, f"historical baseline test receipt CID drift: {row['category']}")
        require(row["raw_log_ref"] == log_refs[log_id] and row["stderr"] == empty_stderr, f"historical baseline test log/stderr link drift: {row['category']}")
    require([bundle["coverage_accounting"][name]["receipt_cid"] for name in ("accelerator", "datasets", "required_validation")] == test_cids, "historical baseline coverage/test receipt links drift")
    require(bundle["bundle_index"]["test_receipt_cids"] == test_cids and bundle["bundle_index"]["embedded_log_cids"] == [expected_log_cids[name] for name in test_log_ids], "historical baseline bundle test/log index drift")

    source_receipts = bundle["source_binding_receipts"]
    source_cids = [
        "baguqeerabbgrmhjhxu7ty5zhqlcc4uia5uwfo3ugikteqf6r3qmrokierbuq",
        "baguqeerasiihzlunix2qwifuwv6dxig3j4vy5x36ye6jsmjg5fbb5ahue7ca",
    ]
    source_log_ids = ["accelerator_source_binding_stdout", "datasets_source_binding_stdout"]
    require([row["repository_id"] for row in source_receipts] == ["SRC-ACCEL-AUTH-1", "SRC-DATASETS-AUTH-1"], "historical baseline source receipt population/order drift")
    for row, expected_cid, log_id in zip(source_receipts, source_cids, source_log_ids, strict=True):
        require(verify_dag_projection_identity(row, cid_field="receipt_cid", identity_field="receipt_identity", label=f"historical baseline source {row['repository_id']}") == expected_cid and row["raw_log_ref"] == log_refs[log_id], f"historical baseline source receipt/log link drift: {row['repository_id']}")
    require([test_receipts[index]["source_binding"]["source_binding_receipt_cid"] for index in (0, 1)] == source_cids and bundle["bundle_index"]["source_binding_receipt_cids"] == source_cids, "historical baseline test/source binding links drift")

    capability = bundle["capability_receipt"]
    capability_cid = "baguqeerambrs2h4t34he6spelik6co6jec53vkgfnkr2remr4bnyz4bn6yia"
    require(verify_dag_projection_identity(capability, cid_field="receipt_cid", identity_field="receipt_identity", label="historical baseline capability") == capability_cid, "historical baseline capability receipt CID drift")
    artifact_validation = bundle["artifact_validation_receipt"]
    artifact_cid = "baguqeerapro3c5roqitkvbeg5zxkpcleeawrsnzp4jay744rzzazrq57kbva"
    require(verify_dag_projection_identity(artifact_validation, cid_field="receipt_cid", identity_field="receipt_identity", label="historical baseline artifact validation") == artifact_cid and artifact_validation["raw_log_ref"] == log_refs["artifact_validation_stdout"], "historical baseline artifact-validation receipt/log drift")
    require(bundle["bundle_index"]["capability_receipt_cid"] == capability_cid and bundle["bundle_index"]["artifact_validation_receipt_cid"] == artifact_cid, "historical baseline capability/artifact bundle links drift")

    proposals = bundle["prerequisite_dispositions"]
    proposal_cids = [
        "baguqeera4jz46uxax5saxk275bjptynipdhjyw4kksen445eqznk5hoqclba",
        "baguqeera4orjzamfvz5o7qotkalha4r4h2twyxnw2xwrmsuti2pieyhflasa",
        "baguqeeradttvo5ivx6rxy3upfekzdabwrkdyv5zbzrkog3bscohpagmuv4va",
    ]
    for row, expected_cid in zip(proposals, proposal_cids, strict=True):
        require(verify_dag_projection_identity(row, cid_field="proposal_cid", identity_field="proposal_identity", label=f"historical baseline proposal {row['category']}") == expected_cid, f"historical baseline proposal CID drift: {row['category']}")
        require(row["trigger_receipt_cid"] == row["repair_task_index"]["trigger_receipt_cid"] == test_cids[1], f"historical baseline proposal trigger link drift: {row['category']}")
    require(bundle["bundle_index"]["prerequisite_proposal_cids"] == proposal_cids, "historical baseline proposal bundle-index drift")

    expected_evidence = [
        *({"kind": "test_receipt", "category": category, "cid": cid} for category, cid in zip(("accelerator", "datasets", "required_validation"), test_cids, strict=True)),
        {"kind": "capability_receipt", "category": "formal_verification_capabilities", "cid": capability_cid},
        *({"kind": "bounded_repair_proposal", "category": row["category"], "cid": cid} for row, cid in zip(proposals, proposal_cids, strict=True)),
        *({"kind": "source_binding_receipt", "category": category, "cid": cid} for category, cid in zip(("accelerator", "datasets"), source_cids, strict=True)),
        {"kind": "artifact_validation_receipt", "category": "artifact-validation", "cid": artifact_cid},
    ]
    require(baseline["evidence"] == expected_evidence, "historical baseline evidence index drift")
    result_projection = baseline["result_projection"]
    require(result_projection["test_receipt_cids"] == test_cids and result_projection["source_binding_receipt_cids"] == source_cids and result_projection["capability_receipt_cid"] == capability_cid and result_projection["artifact_validation_receipt_cid"] == artifact_cid and result_projection["bounded_repair_proposal_cids"] == proposal_cids, "historical baseline result receipt/proposal links drift")

    bundle_cid = verify_dag_projection_identity(bundle, cid_field="bundle_cid", identity_field="bundle_identity", label="historical baseline evidence bundle")
    require(bundle_cid == "baguqeerar3jycpkvsx45hnfnd7u4bh37p2lkvsnl373w2b2ogaj2ixxuwhkq", "historical baseline evidence-bundle CID drift")
    require(baseline["artifact"]["evidence_bundle_cid"] == baseline["summary_index"]["bundle_cid"] == bundle_cid and baseline["artifact"]["evidence_bundle_identity"] == bundle["bundle_identity"], "historical baseline artifact/summary bundle links drift")
    require(result_projection["effects"]["allowed_output_cids"] == [bundle_cid], "historical baseline allowed-output bundle link drift")
    result_cid = dag_cid(result_projection)
    require(baseline["result_cid"] == baseline["summary_index"]["result_cid"] == result_cid == "baguqeerabt5f2oxnlybr33a6azszdctygqthph2ww5wrk2dcqxutym4j6yjq" and baseline["result_identity"] == "sha256:" + hashlib.sha256(canonical(result_projection)).hexdigest(), "historical baseline result identity/index drift")
    summary_projection = {key: value for key, value in baseline.items() if key not in {"summary_cid", "summary_identity"}}
    require(baseline["summary_cid"] == dag_cid(summary_projection) == "baguqeerasr4hxpxwkhe64btsxu56moh7v7ww2a6covaavjd7qnzb4xgl5cpq" and baseline["summary_identity"] == "sha256:" + hashlib.sha256(canonical(summary_projection)).hexdigest(), "historical baseline summary identity drift")

    manifest_cid = projection_cid(manifest, "manifest_cid")
    supervisor_cid = projection_cid(supervisor, "inventory_cid")
    require(manifest_cid == "baguqeerasownoxqyrppw3ft3us3yvd26ghvqnjl74nr2rw5o7sm3sjehip7a" and supervisor_cid == "baguqeerablvf72zunpjvbievbspxqnc4eqgxneqjwrg5v6imr7edavovmwca", "historical baseline source manifest/inventory CID drift")
    require(baseline["input_identities"]["source_manifest"]["cid"] == manifest_cid and baseline["input_identities"]["source_manifest"]["result_identity"] == result["result_identity"], "historical baseline source manifest/result cross-link drift")
    require(capability["source_inventory"] == {"task_id": "PGIR-003", "path": "docs/architecture/proof_grounded_ir_learning/inventory/supervisor.json", "inventory_cid": supervisor_cid}, "historical baseline capability/supervisor inventory link drift")

    task_specs = (
        ("8b42722897be2d2b88e416a40370c6a56b04bad8", "PGIR-001", "06d73b3257d1d58b9d06615401b305c6932f8911da57a54d028c09791183ff63", "baguqeeraa3ltwmsx2hkyxhigmfkadmyfy2js7cir3jl2kticrqexsemd75rq"),
        ("52756763d98e0a1b9ef40bcf842e41f79039886c", "PGIR-001", "83e5d82c072ca7c9b40c52402fdef6a37ba69aab4289effb1558df2c13a5b077", "baguqeeraqps5qlahfst4tnamkjac7xxwun52ngvlike676yvldpsye5fwb3q"),
        (CAMPAIGN_OUTER_REVISION, "PGIR-002", "16f98828073c8b57213876ce126ca5d536ef3bfde6fdfbb59e9bd992fd10130f", "baguqeerac34yqkahhsfvoijyo3hbe3ff2u3o6o754367xnm6tpmzf7iqcmhq"),
        (CAMPAIGN_OUTER_REVISION, "PGIR-003", "9702aecea1283d19902d303cf9983bd915e7c737931e24c99ccec96f75571eb6", "baguqeeras4bk5tvbfa6rtebnga6ptgb33ek6przxsmpcjsm4z3ew65kxd23a"),
        ("597a0285738c5878eed462593fd75e18715ff7f8", "PGIR-005", "7a229062dbbad8cda79fa1f63c89f69bbe2b8d13d5ec17be95cf111facc83af7", "baguqeerapirjayw3xlmm3j47uh3dzcpwto7cxdit2xwbppuvz4ir7lgihl3q"),
    )
    task_replays = [replay_historical_task_identity(*spec) for spec in task_specs]
    executed, admitted, pgir002, pgir003, pgir005 = [row["canonical_task_cid"] for row in task_replays]
    dependencies = {row["task_id"]: row for row in baseline["input_identities"]["upstream_dependencies"]}
    require(manifest["host_task_tree_id"] == task_specs[0][0] and manifest["task_revision"] == result["executed_task_revision"] == dependencies["PGIR-001"]["executed_task_revision"] == executed, "historical baseline PGIR-001 executed revision links drift")
    require(result["plan_projection_repair_commit"] == task_specs[1][0] and result["admitted_task_revision"] == dependencies["PGIR-001"]["admitted_task_revision"] == admitted, "historical baseline PGIR-001 admitted revision links drift")
    require(dependencies["PGIR-002"]["task_revision"] == modules["objective_revision"] == pgir002 and dependencies["PGIR-003"]["task_revision"] == supervisor["objective_revision"] == pgir003, "historical baseline PGIR-002/003 objective revision links drift")
    pgir005_occurrences = [
        baseline["executed_task_revision"], baseline["admitted_task_revision"],
        test_receipts[2]["source_binding"]["task_revision"], bundle["bundle_index"]["task_revision"],
        result_projection["task_revision"], baseline["summary_index"]["task_revision"],
    ]
    require(baseline["input_identities"]["task_base_commit"] == task_specs[4][0] and pgir005_occurrences == [pgir005] * 6, "historical baseline PGIR-005 task revision occurrence links drift")

    campaign = git_json(ROOT, TOKENIZER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json")
    compiler_binding = campaign["bindings"]["compiler"]["files"][0]
    compiler_path = compiler_binding["path"].removeprefix("ipfs_datasets_py/")
    compiler_observations = []
    for revision in (R1_REVISION, CAMPAIGN_NESTED_REVISION, CURRENT):
        compiler_oid, compiler_bytes = git_blob(DATASETS, revision, compiler_path)
        compiler_observations.append({"revision": revision, "path": compiler_binding["path"], "git_blob": compiler_oid, **identity(compiler_bytes)})
    live_compiler_cid = "bafkreihkepmhrqadg7uldzbmnqdpvklwuyod55vui23dywqegj5diz4lci"
    require(all(row["raw_cid"] == live_compiler_cid for row in compiler_observations) and compiler_binding["raw_cid"] == live_compiler_cid, "historical baseline live compiler identity drift")
    failure = test_receipts[1]["outcome"]["first_failed_assertion"]
    snapshot_path = str(failure["snapshot_path"])
    snapshot_observations = []
    for revision in (R1_REVISION, CAMPAIGN_NESTED_REVISION, CURRENT):
        snapshot_oid, snapshot_bytes = git_blob(DATASETS, revision, snapshot_path)
        snapshot = strict_json_bytes(snapshot_bytes, f"historical baseline snapshot {revision}")
        observed_value = snapshot["lineage"]["implementation_raw_cids"]["compiler"]
        snapshot_observations.append({"revision": revision, "path": "ipfs_datasets_py/" + snapshot_path, "git_blob": snapshot_oid, "compiler_raw_cid": observed_value, **identity(snapshot_bytes)})
    dirty_value = "bafkreiazl2bdiqresmibt4tjhqhgfd66utqvu4drn642rpykn36yarx6em"
    require(failure["live_expected_value"] == live_compiler_cid and failure["snapshot_value"] == dirty_value and all(row["compiler_raw_cid"] == live_compiler_cid for row in snapshot_observations), "historical baseline dirty-snapshot mismatch drift")
    require(dirty_value.encode() in decoded_logs["datasets_stdout"], "historical baseline dirty snapshot value is absent from retained dataset log")
    dirty_mismatch = {"status": "historical_dirty_checkout_unsealed_mismatch", "declared_unsealed_snapshot_raw_cid": dirty_value, "immutable_committed_snapshot_raw_cid": live_compiler_cid, "matches": False, "snapshot_observations": snapshot_observations, "compiler_observations": compiler_observations}

    distinct_cids = sorted(set(re.findall(r"\b(?:bafkrei|baguqeera)[a-z2-7]+", json.dumps(baseline, sort_keys=True))))
    require(len(distinct_cids) == 29, "historical baseline distinct CID population drift")
    return {
        "distinct_cid_count": 29,
        "embedded_log_count": 6,
        "test_receipt_cids": test_cids,
        "source_binding_receipt_cids": source_cids,
        "capability_receipt_cid": capability_cid,
        "artifact_validation_receipt_cid": artifact_cid,
        "proposal_cids": proposal_cids,
        "bundle_cid": bundle_cid,
        "result_cid": result_cid,
        "summary_cid": baseline["summary_cid"],
        "source_manifest_cid": manifest_cid,
        "supervisor_inventory_cid": supervisor_cid,
        "task_identity_replays": task_replays,
        "dirty_snapshot_mismatch": dirty_mismatch,
    }


def verify_campaign_inventories() -> dict[str, Any]:
    manifest = git_json(ROOT, CAMPAIGN_OUTER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json")
    result = git_json(ROOT, CAMPAIGN_OUTER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/result.json")
    supervisor = git_json(ROOT, CAMPAIGN_OUTER_REVISION, "docs/architecture/proof_grounded_ir_learning/inventory/supervisor.json")
    modules = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "docs/architecture/proof_grounded_ir_learning/inventory/modules.json")
    inventory = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/source_inventory/release_inventory.json")
    gap = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "docs/architecture/proof_grounded_ir_learning/gap_matrix.json")
    baseline = git_json(ROOT, CAMPAIGN_OUTER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/baseline_tests/summary.json")

    manifest_cid = projection_cid(manifest, "manifest_cid")
    result_projection = dict(result)
    result_identity = str(result_projection.pop("result_identity"))
    require(result_identity == "sha256:" + hashlib.sha256(canonical(result_projection)).hexdigest(), "source-revision result identity drift")
    _manifest_oid, manifest_bytes = git_blob(ROOT, CAMPAIGN_OUTER_REVISION, result["artifact"]["path"])
    manifest_blob = git(ROOT, "rev-parse", f"{CAMPAIGN_OUTER_REVISION}:{result['artifact']['path']}")
    require(result["artifact"] == {
        "path": "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json",
        "git_blob": manifest_blob,
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "cid": manifest_cid,
        "byte_count": len(manifest_bytes),
        "commit_count": result["artifact"]["commit_count"],
    }, "source-revision result/manifest artifact link drift")
    require(result["artifact"]["commit_count"] == sum(len(row["commit_manifest"]["commits"]) for row in manifest["repositories"]), "source-revision commit population drift")

    inventory_cid = projection_cid(supervisor, "inventory_cid")
    modules_sha = projection_sha(modules, "inventory_sha256")
    release_sha = projection_sha(inventory, "inventory_sha256")
    gap_projection = dict(gap)
    gap_cid = str(gap_projection.pop("matrix_cid"))
    gap_sha = str(gap_projection.pop("matrix_sha256"))
    require(gap_cid == dag_cid(gap_projection) and gap_sha == hashlib.sha256(canonical(gap_projection)).hexdigest(), "gap-matrix identities drift")
    upstream = {"manifest_cid": manifest_cid, "result_identity": result_identity}
    require(supervisor["source_binding"]["upstream_dependency"]["manifest_cid"] == upstream["manifest_cid"] and supervisor["source_binding"]["upstream_dependency"]["result_identity"] == upstream["result_identity"], "supervisor upstream link drift")
    require(modules["source_binding"]["upstream_result"]["manifest_cid"] == upstream["manifest_cid"] and modules["source_binding"]["upstream_result"]["result_identity"] == upstream["result_identity"], "module inventory upstream link drift")
    require(inventory["source_binding"]["upstream_result"]["manifest_cid"] == upstream["manifest_cid"] and inventory["source_binding"]["upstream_result"]["result_identity"] == upstream["result_identity"], "release inventory upstream link drift")
    gap_inputs = {row["task_id"]: row for row in gap["input_artifacts"]}
    require(gap_inputs["PGIR-002"]["identity"] == "sha256:" + modules_sha, "gap/module inventory link drift")
    require(gap_inputs["PGIR-003"]["identity"] == inventory_cid, "gap/supervisor inventory link drift")
    require(gap_inputs["PGIR-004"]["identity"] == "sha256:" + release_sha, "gap/release inventory link drift")
    baseline_projection = {key: value for key, value in baseline.items() if key not in {"summary_cid", "summary_identity"}}
    require(baseline["summary_cid"] == dag_cid(baseline_projection) and baseline["summary_identity"] == "sha256:" + hashlib.sha256(canonical(baseline_projection)).hexdigest(), "baseline summary identity drift")
    require(baseline["result_cid"] == dag_cid(baseline["result_projection"]) and baseline["result_identity"] == "sha256:" + hashlib.sha256(canonical(baseline["result_projection"])).hexdigest(), "baseline result projection drift")
    require(gap_inputs["PGIR-005"]["identity"] == baseline["summary_cid"] and gap_inputs["PGIR-005"]["result_cid"] == baseline["result_cid"], "gap/baseline summary link drift")
    baseline_recursive = verify_historical_baseline(baseline, manifest=manifest, result=result, supervisor=supervisor, modules=modules)

    try:
        import yaml
        _pinset_oid, pinset_bytes = git_blob(ROOT, CAMPAIGN_OUTER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/justice_dao_pinset.yaml")
        pinset = yaml.safe_load(pinset_bytes.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"cannot safely parse pinset: {exc}") from exc
    require(isinstance(pinset, dict) and isinstance(pinset.get("repositories"), list), "pinset shape drift")
    pin_pairs = [(str(row["id"]), str(row["revision"])) for row in pinset["repositories"]]
    inventory_pairs = [(str(row["id"]), str(row["revision"])) for row in inventory["repositories"]]
    require(len(pin_pairs) == len(set(pin_pairs)) == 21 and pin_pairs == inventory_pairs, "pinset/release inventory exact-revision population drift")
    require(inventory["counts"]["repository_count"] == 21 and inventory["counts"]["inventory_candidate_source_rows"] == 7173 and inventory["counts"]["training_admitted_source_rows"] == 0, "release inventory population drift")
    return {
        "source_revision_manifest_cid": manifest_cid,
        "source_revision_result_identity": result_identity,
        "supervisor_inventory_cid": inventory_cid,
        "module_inventory_sha256": modules_sha,
        "release_inventory_sha256": release_sha,
        "gap_matrix_cid": gap_cid,
        "baseline_recursive": baseline_recursive,
        "pinset_repository_count": 21,
    }


def verify_campaign_rebuild() -> dict[str, Any]:
    sys.path.insert(0, str(DATASETS))
    from ipfs_datasets_py.huggingface.corpus import seal_corpus
    from ipfs_datasets_py.logic.ir_core.source_lineage import CorpusManifest, SourceRelease
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.legal_ir_eval_splits import (
        LegalIREvalSplitConfig,
        campaign_samples_from_corpus_root,
        seal_ir_campaign_splits,
    )
    inventory = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/source_inventory/release_inventory.json")
    with tempfile.TemporaryDirectory(prefix="pgir211-corpus-replay-") as temporary:
        rebuilt = Path(temporary)
        seal_corpus(inventory, rebuilt, materialize=False)
        for name in ("corpus_manifest.json", "corpus_root.json", "lineage_graph.json", "quarantine_manifest.json", "reconciliation_receipt.json", "rights_manifest.json", "source_releases.json"):
            _oid, historical = git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, f"data/ir_learning/corpora/{name}")
            require((rebuilt / name).read_bytes() == historical, f"historical corpus deterministic replay drift: {name}")
    corpus_manifest = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/corpora/corpus_manifest.json")
    manifest_object = CorpusManifest.from_dict(corpus_manifest)
    releases = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/corpora/source_releases.json")["releases"]
    require(len(releases) == 2, "historical source-release record population drift")
    for row in releases:
        require(SourceRelease.from_dict(row).to_dict() == row, f"source-release CID replay drift: {row.get('release_id')}")
    samples = campaign_samples_from_corpus_root(corpus_manifest)
    require(len(samples) == 7173, "historical campaign sample population drift")
    with tempfile.TemporaryDirectory(prefix="pgir211-split-replay-") as temporary:
        rebuilt = Path(temporary)
        split_root = seal_ir_campaign_splits(samples, rebuilt, config=LegalIREvalSplitConfig(seed="pgir-012-jdao-pinset-1"))
        for name in ("ir_split_manifest.json", "split_root.json", "holdout_report.json", "leakage_report.json"):
            _oid, historical = git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, f"data/ir_learning/splits/{name}")
            require((rebuilt / name).read_bytes() == historical, f"historical split deterministic replay drift: {name}")
    split_manifest = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/splits/ir_split_manifest.json")
    require(len(split_manifest["assignments"]) == 7173 and split_manifest["config_digest"] == "4cb61f288c66ac8363bab2e2196904678551e6ad7cc8ef4a02d9f095a364c677" and split_manifest["split_manifest_digest"] == "047b263b85067aa3dad6760f623c2855fbaf776d565ec9c273c49425fcc14eb4", "historical split digest/population drift")
    require(split_manifest["hidden_test_commitment"] == HIDDEN and split_root["hidden_test_commitment"] == HIDDEN, "historical hidden commitment drift")
    return {
        "source_count": manifest_object.source_count,
        "derived_count": manifest_object.derived_count,
        "source_release_count": 2,
        "split_example_count": len(samples),
        "split_assignment_count": len(split_manifest["assignments"]),
        "split_manifest_digest": split_manifest["split_manifest_digest"],
        "hidden_inputs_loaded": False,
        "materialized_source_rows": 0,
    }


def verify_campaign_semantics(campaign: Mapping[str, Any]) -> dict[str, Any]:
    bindings = campaign["bindings"]
    r1_identities = git_json(DATASETS, R1_REVISION, "data/ir_learning/evaluations/deterministic/identities.json")
    corpus_root = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/corpora/corpus_root.json")
    corpus_manifest = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/corpora/corpus_manifest.json")
    lineage = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/corpora/lineage_graph.json")
    rights = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/corpora/rights_manifest.json")
    split_root = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/splits/split_root.json")
    split_manifest = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/splits/ir_split_manifest.json")
    require(bindings["compiler"]["configuration_cid"] == r1_identities["compiler"]["configuration_cid"] and bindings["compiler"]["repository_commit"] == CAMPAIGN_NESTED_REVISION, "campaign compiler semantic link drift")
    require(bindings["decompiler"]["configuration_cid"] == r1_identities["decompiler"]["configuration_cid"] and bindings["decompiler"]["rendering_spec_cid"] == r1_identities["decompiler"]["rendering_spec_cid"] and bindings["decompiler"]["repository_commit"] == CAMPAIGN_NESTED_REVISION, "campaign decompiler semantic link drift")
    require(bindings["corpus"]["manifest_cid"] == corpus_root["manifest_cid"] == corpus_manifest["record_cid"], "campaign corpus manifest link drift")
    require((bindings["corpus"]["source_count"], bindings["corpus"]["derived_count"], bindings["corpus"]["training_admitted_rows"], bindings["corpus"]["materialized"]) == (corpus_manifest["source_count"], corpus_manifest["derived_count"], corpus_root["training_admitted_rows"], corpus_root["materialized"]), "campaign corpus population/materialization link drift")
    require(bindings["lineage"]["lineage_graph_cid"] == corpus_root["lineage_graph_cid"] == raw_cid(canonical(lineage)), "campaign lineage graph link drift")
    require(bindings["lineage"]["source_count"] == corpus_manifest["source_count"] and bindings["lineage"]["derived_count"] == corpus_manifest["derived_count"] and sum(row["source_group_count"] for row in lineage["populations"]) == 7173, "campaign lineage population drift")
    require(bindings["rights"]["source_count"] == rights["source_count"] == 7173 and bindings["rights"]["admitted_source_count"] == bindings["rights"]["training_admitted_rows"] == rights["training_admitted_rows"] == 0, "campaign rights semantic link drift")
    require(bindings["split"]["split_manifest_digest"].removeprefix("sha256:") == split_root["split_manifest_digest"] == split_manifest["split_manifest_digest"], "campaign split digest link drift")
    require(bindings["split"]["split_manifest_sha256"].removeprefix("sha256:") == split_root["split_manifest_sha256"] == hashlib.sha256(git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/splits/ir_split_manifest.json")[1]).hexdigest(), "campaign split SHA link drift")
    require(bindings["split"]["hidden_test_commitment"] == split_root["hidden_test_commitment"] == HIDDEN and bindings["split"]["leakage_passed"] is True, "campaign split hidden/leakage link drift")
    snapshots = bindings["source_snapshots"]
    require(snapshots["selected_repository_tree"] == CAMPAIGN_OUTER_REVISION and snapshots["selected_datasets_commit"] == CAMPAIGN_NESTED_REVISION and snapshots["authority_commits"] == {"ipfs_accelerate_py": "8d46a6d25dd006c8cab3c9d9612707d2a014e79c", "ipfs_datasets_py": "df93e91e6338c84a17c3208ef68b88de8566f78c"}, "campaign source-snapshot revision links drift")
    manifest = git_json(ROOT, CAMPAIGN_OUTER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json")
    source_result = git_json(ROOT, CAMPAIGN_OUTER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/result.json")
    require(snapshots["source_manifest_cid"] == manifest["manifest_cid"] and snapshots["source_result_identity"] == source_result["result_identity"], "campaign source-snapshot identity links drift")
    tokenizer = git_json(ROOT, TOKENIZER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/freeze/tokenizer_policy.json")
    tokenizer_projection = {key: value for key, value in tokenizer.items() if key not in {"policy_cid", "policy_sha256"}}
    require(tokenizer["policy_cid"] == dag_cid(tokenizer_projection) and tokenizer["policy_sha256"] == "sha256:" + hashlib.sha256(canonical(tokenizer_projection)).hexdigest(), "campaign tokenizer policy self identities drift")
    require(bindings["tokenizer_policy"]["policy_cid"] == tokenizer["policy_cid"] and bindings["tokenizer_policy"]["status"] == tokenizer["status"] == "no_learned_tokenizer_admitted" and tokenizer["training_policy"]["authorized"] is False, "campaign tokenizer semantic link drift")
    require(tokenizer["unknown_token_behavior"] == "fail_closed" and tokenizer["canonical_tokenization"]["learned_vocabulary_identity"] == tokenizer["canonical_tokenization"]["model_checkpoint_identity"] == "none" and tokenizer["training_policy"]["superseding_root_required"] is True and tokenizer["training_policy"]["reason"] == "PGIR-050 must define and freeze a compatible tokenizer before learned training", "campaign tokenizer learned/fail-closed policy drift")
    require(bindings["schema_registry"]["tree_oid"] == git(DATASETS, "rev-parse", f"{bindings['schema_registry']['implementation_commit']}:ipfs_datasets_py/logic/ir_core") == git(DATASETS, "rev-parse", f"{CAMPAIGN_NESTED_REVISION}:ipfs_datasets_py/logic/ir_core"), "campaign schema-registry implementation subtree drift")
    schema_source = b"".join(git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, row["path"].removeprefix("ipfs_datasets_py/"))[1] for row in bindings["schema_registry"]["files"])
    require(all(schema_id.encode() in schema_source for schema_id in bindings["schema_registry"]["schema_ids"]), "campaign schema registry declared IDs are not present in bound source")
    contract_source = b"".join(git_blob(DATASETS, CAMPAIGN_NESTED_REVISION, row["path"].removeprefix("ipfs_datasets_py/"))[1] for row in bindings["example_contracts"]["files"])
    require(all(schema_id.encode() in contract_source for schema_id in bindings["example_contracts"]["schema_ids"]), "campaign example-contract declared IDs are not present in bound source")
    semantic_commits = {
        value
        for binding in bindings.values()
        for key, value in binding.items()
        if (key.endswith("_commit") or key == "repository_commit") and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
    }
    known = {oid for oid, _roles in NESTED_CANDIDATES} | {oid for oid, _roles in OUTER_CANDIDATES}
    require(semantic_commits.issubset(known), "campaign semantic commit population is absent from portability closure")
    adapter_path = "benchmarks/semantic_roundtrip/constructors/typed_deontic.py"
    observed_adapter = []
    for revision in (R1_REVISION, CAMPAIGN_NESTED_REVISION, CURRENT):
        oid, data = git_blob(DATASETS, revision, adapter_path)
        observed_adapter.append({"revision": revision, "path": "ipfs_datasets_py/" + adapter_path, "git_blob": oid, **identity(data)})
    observed_raw = "bafkreifvgezdodtjnaejikc5wf56qebbpo36jytkivz7ekvcjcisvag5ha"
    declared_raw = "bafkreife5avbe5esju4frufsogvzlaew5x5qw5h4qlefvgx2qdbamqsyny"
    require(all(row["raw_cid"] == observed_raw for row in observed_adapter) and bindings["compiler"]["measured_adapter_raw_cid"] == declared_raw and declared_raw != observed_raw, "campaign measured-adapter historical mismatch drift")
    adapter_mismatch = {"status": "typed_permanent_no_go", "declared_raw_cid": declared_raw, "observed_raw_cid": observed_raw, "matches": False, "observations": observed_adapter}
    policy_opaque = {"field": "campaign.bindings.policy.policy_revision", "value": "sha256:27c8da23ef92ab263ac0c144f2414fd40bdb30aace98b88f7dd76d36db26e142", "status": "opaque_non_authoritative_no_derivation_contract"}
    require(bindings["policy"]["policy_revision"] == policy_opaque["value"], "campaign opaque historical policy revision drift")
    return {"semantic_commit_count": len(semantic_commits), "tokenizer_authorized": False, "schema_id_count": len(bindings["schema_registry"]["schema_ids"]), "example_contract_schema_id_count": len(bindings["example_contracts"]["schema_ids"]), "measured_adapter_mismatch": adapter_mismatch, "opaque_policy_revision": policy_opaque}


def historical_task_material(revision: str, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = "docs/architecture/proof_grounded_ir_learning.todo.md"
    source_record = record_for(path, ROOT, revision)
    source = git_blob(ROOT, revision, path)[1].decode("utf-8")
    tasks: dict[str, tuple[str, list[str]]] = {}
    current_id = ""
    current_title = ""
    block: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_title, block
        if current_id:
            require(current_id not in tasks, f"duplicate historical task: {current_id}")
            tasks[current_id] = (current_title, block)
        current_id, current_title, block = "", "", []

    for line in source.splitlines():
        if line.startswith("## "):
            flush()
            if line.startswith("## PGIR-"):
                header = line[3:].strip()
                current_id, _, current_title = header.partition(" ")
        elif current_id:
            block.append(line)
    flush()
    require(task_id in tasks, f"{task_id} is absent from historical task board at {revision}")
    title, task_block = tasks[task_id]
    metadata: dict[str, str] = {}
    for line in task_block:
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            key, value = stripped[2:].split(":", 1)
            metadata[key.strip().lower()] = value.strip()

    def split_csv(value: Any) -> list[str]:
        return [item for raw in str(value or "").split(",") if (item := raw.strip()) and item.lower() not in {"none", "n/a"}]

    def normalize_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    def normalize_path(value: Any) -> str:
        text = str(value or "").strip().replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        return re.sub(r"/+", "/", text).rstrip("/")

    outputs = sorted({normalized for item in split_csv(metadata.get("outputs")) if (normalized := normalize_path(item))})
    acceptance = [normalized for item in split_csv(metadata.get("acceptance") or metadata.get("acceptance criteria")) if (normalized := normalize_text(item))]
    evidence = sorted({normalized for item in split_csv(metadata.get("missing evidence")) if (normalized := normalize_text(item))})
    goal = normalize_text(metadata.get("goal id") or metadata.get("goal packet key") or metadata.get("goal"))
    hint = normalize_text(metadata.get("semantic key") or metadata.get("bundle key") or metadata.get("work scope") or metadata.get("fingerprint"))
    semantic = {key: value for key, value in {
        "title": normalize_text(title), "outputs": outputs, "acceptance": acceptance,
        "evidence": evidence, "goal": goal, "semantic_hint": hint,
    }.items() if value}
    require(bool(semantic), f"{task_id} has no historical semantic identity")
    return {"schema": "ipfs_accelerate_py/agent-supervisor/task-identity@1", "semantic": semantic}, source_record


def replay_historical_task_identity(revision: str, task_id: str, expected_fingerprint: str, expected_cid: str) -> dict[str, Any]:
    material, source = historical_task_material(revision, task_id)
    fingerprint = hashlib.sha256(canonical(material)).hexdigest()
    require(fingerprint == expected_fingerprint and dag_cid(material) == expected_cid, f"historical canonical task identity drift: {task_id}@{revision}")
    return {"task_id": task_id, "source": source, "material": material, "semantic_fingerprint": fingerprint, "canonical_task_key": "task/v1/" + fingerprint, "canonical_task_cid": expected_cid}


def verify_campaign_objective(campaign: Mapping[str, Any], closure: Mapping[str, Any]) -> dict[str, Any]:
    path = "docs/architecture/proof_grounded_ir_learning.todo.md"
    expected = record_for(path, ROOT, CAMPAIGN_OUTER_REVISION)
    require(expected == {
        "path": path,
        "repository": "ipfs_accelerate_py",
        "revision": CAMPAIGN_OUTER_REVISION,
        "git_blob": "83222f6e464139016aa6ff1d974760ee851826f5",
        "size_bytes": 102946,
        "sha256": "sha256:674ff7d34ae41125c1105ef1a0aba7d1e0502284a22d03764c7bac92cfb2eb20",
        "raw_cid": "bafkreidhj735gsxeces4cec66gqkxj6r4bicfbfcfubxmtd3vsjm7mxlea",
    }, "historical PGIR-014 task-board byte identity drift")
    require(closure.get("campaign_objective_source") == expected, "historical PGIR-014 source receipt drift")
    verify_file_record(expected, disk_required=False)
    task_replay = replay_historical_task_identity(
        CAMPAIGN_OUTER_REVISION, "PGIR-014",
        "5857ac25de5f4c0c89925bf309d3dfa3793bab76b3d31e76673f64255a1efe0f",
        CAMPAIGN_OBJECTIVE_CID,
    )
    material = task_replay["material"]
    require(material == {
        "schema": "ipfs_accelerate_py/agent-supervisor/task-identity@1",
        "semantic": {
            "acceptance": ["full referential integrity and reproducible cids; zero training task eligible on unresolved rights/leakage/compiler identity"],
            "outputs": ["data/agent_supervisor/proof_grounded_ir_learning/freeze"],
            "title": "freeze semantic campaign inputs",
        },
    }, "PGIR-014 normalized semantic task material drift")
    policy = campaign["bindings"]["policy"]
    require(campaign["objective_revision"] == policy["objective_revision"] == CAMPAIGN_OBJECTIVE_CID, "campaign PGIR-014 objective link drift")
    protected_paths = policy["protected_paths"]
    file_paths = [row["path"] for row in policy["files"]]
    require(protected_paths == [
        "data/agent_supervisor/proof_grounded_ir_learning/justice_dao_pinset.yaml",
        "docs/architecture/proof_grounded_ir_learning.objectives.md",
        path,
    ] and len(protected_paths) == len(set(protected_paths)), "campaign exact protected-path population/order drift")
    require(file_paths == protected_paths[:2] and len(file_paths) == len(set(file_paths)), "campaign policy exact bound-file population/order drift")
    return {**task_replay, "protected_unbound_path": path}


def verify_campaign(closure: Mapping[str, Any], *, rebuild: bool = True) -> dict[str, Any]:
    campaign = git_json(ROOT, TOKENIZER_REVISION, "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json")
    projection = {key: value for key, value in campaign.items() if key not in {"root_sha256", "root_cid"}}
    require(campaign["root_sha256"] == "sha256:" + hashlib.sha256(canonical(projection)).hexdigest(), "campaign root SHA drift")
    require(campaign["root_cid"] == dag_cid(projection), "campaign root CID drift")
    require(set(campaign["bindings"]) == set(CAMPAIGN_COUNTS), "campaign binding domain drift")
    for name, binding in campaign["bindings"].items():
        projected = dict(binding)
        claimed = projected.pop("binding_cid")
        require(claimed == dag_cid(projected), f"campaign binding CID drift: {name}")
    expected = expected_campaign_bindings(campaign)
    require(closure.get("campaign_bindings") == expected, "campaign binding occurrence receipt drift")
    require({row["binding"]: row["record_count"] for row in expected} == CAMPAIGN_COUNTS, "campaign category counts drift")
    occurrences = [record for row in expected for record in row["records"]]
    paths = [row["path"] for row in occurrences]
    require(len(occurrences) == 44 and len(set(paths)) == 41, "campaign 44-occurrence/41-path population drift")
    duplicates = Counter(paths)
    require(sorted(path for path, count in duplicates.items() if count == 2) == [
        "ipfs_datasets_py/data/ir_learning/corpora/lineage_graph.json",
        "ipfs_datasets_py/data/ir_learning/corpora/rights_manifest.json",
        "ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_contracts.py",
    ] and all(count in {1, 2} for count in duplicates.values()), "campaign duplicate multiset drift")
    expected_root = record_for("data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json", ROOT, TOKENIZER_REVISION)
    expected_root.update({"root_sha256": campaign["root_sha256"], "root_cid": campaign["root_cid"]})
    require(closure.get("campaign_root") == expected_root, "campaign root byte/Git receipt drift")
    inventories = verify_campaign_inventories()
    rebuilt = verify_campaign_rebuild() if rebuild else {"rebuild_deferred": True}
    semantics = verify_campaign_semantics(campaign)
    objective = verify_campaign_objective(campaign, closure)
    return {"record_occurrences": 44, "unique_paths": 41, "binding_count": 12, "inventories": inventories, "rebuild": rebuilt, "semantics": semantics, "objective": objective}


def verify_successor_population() -> dict[str, Any]:
    """Reconstruct the sealed 7,173-row no-go and its empty successor split."""
    corpus_dir = DATASETS / "data/ir_learning/corpora/successor-v1"
    split_dir = DATASETS / "data/ir_learning/splits/successor-v1"
    corpus_names = (
        "rights_manifest.json", "quarantine_manifest.json", "source_releases.json",
        "replay_receipt.json", "corpus_manifest.json", "count_receipt.json",
        "lineage_graph.json", "load_receipt.json", "corpus_root.json",
    )
    split_names = (
        "holdout_report.json", "leakage_report.json", "ir_split_manifest.json",
        "replay_receipt.json", "split_root.json",
    )
    corpus = {name: read_json(corpus_dir / name) for name in corpus_names}
    split = {name: read_json(split_dir / name) for name in split_names}

    rights = corpus["rights_manifest.json"]
    quarantine = corpus["quarantine_manifest.json"]
    releases = corpus["source_releases.json"]
    rights_replay = corpus["replay_receipt.json"]
    manifest = corpus["corpus_manifest.json"]
    counts = corpus["count_receipt.json"]
    lineage = corpus["lineage_graph.json"]
    load = corpus["load_receipt.json"]
    corpus_root = corpus["corpus_root.json"]

    require(rights["admission_decision"] == "permanent_zero_for_jdao_pinset_1" and rights["training_eligible"] is False, "successor rights decision drift")
    require(rights["training_admitted_rows"] == 0 and rights["admitted_source_record_ids"] == [], "successor rights admission is nonempty")
    require(rights["quarantined_source_record_count"] == 7173, "successor rights quarantine count drift")
    require(rights["row_disposition_artifact"] == "quarantine_manifest.json#row_dispositions" and rights["source_release_artifact"] == "source_releases.json", "successor rights artifact links drift")
    require(rights["permanent_no_go"]["reason_code"] == "missing_exact_source_and_transformation_rights_authority", "successor rights no-go reason drift")

    ranges = quarantine["row_dispositions"]
    require(len(ranges) == 2, "successor quarantine must contain two source populations")
    release_by_id = {row["id"]: row for row in releases["releases"]}
    exact_ranges = (
        {
            "release_id": "justicedao/patent-legal-ir-graphrag",
            "record_id_format": "src:patent:%04d",
            "record_id_range": {"first": 0, "last": 2173},
            "row_count": 2174,
        },
        {
            "release_id": "justicedao/wetwijzer_netherlands_legal_corpus",
            "record_id_format": "src:dutch-law:%04d",
            "record_id_range": {"first": 0, "last": 4998},
            "row_count": 4999,
        },
    )
    expanded: set[str] = set()
    population_counts: dict[str, int] = {}
    for row, exact in zip(ranges, exact_ranges, strict=True):
        require({key: row[key] for key in exact} == exact, f"successor exact quarantine range drift: {exact['release_id']}")
        require(row["citation_id"] == release_by_id[row["release_id"]]["citation"]["id"], f"successor quarantine citation link drift: {row['release_id']}")
        first, last = row["record_id_range"]["first"], row["record_id_range"]["last"]
        ids = {row["record_id_format"] % index for index in range(first, last + 1)}
        require(len(ids) == row["row_count"] and not expanded.intersection(ids), "successor quarantine ranges overlap or have gaps")
        require(row["disposition"] == "permanently_quarantined_for_this_pinset", "successor quarantine disposition drift")
        expanded.update(ids)
        population_counts[row["record_id_format"].split(":")[1]] = row["row_count"]
    require(len(expanded) == 7173 and sorted(population_counts.values()) == [2174, 4999], "successor 2,174 + 4,999 population reconstruction drift")
    require(quarantine["training_eligible_rows"] == 0 and quarantine["all_quarantined_rows_have_a_citation"] is True, "successor quarantine boundary drift")
    require(len(releases["releases"]) == 21 and releases["training_admitted_rows"] == 0, "successor exact 21-release population drift")
    require(len({(row["id"], row["revision"]) for row in releases["releases"]}) == 21, "successor release identities are not unique")
    require(all(row["training_admitted_rows"] == 0 and row["citation"]["observed_revision"] == row["revision"] for row in releases["releases"]), "successor release citation/admission drift")
    require(quarantine["quarantined_release_ids"] == sorted(row["id"] for row in releases["releases"] if row["disposition"] == "permanently_quarantined_for_this_pinset"), "successor exact 12-release quarantine set drift")
    require(quarantine["rejected_release_ids"] == sorted(row["id"] for row in releases["releases"] if row["disposition"] == "permanently_rejected_for_this_pinset"), "successor exact 9-release rejected set drift")
    require(len(quarantine["quarantined_release_ids"]) == 12 and len(quarantine["rejected_release_ids"]) == 9 and not set(quarantine["quarantined_release_ids"]).intersection(quarantine["rejected_release_ids"]), "successor 12+9 release disposition population drift")
    require(rights_replay["result_identity"] == "RESULT(PGIR-200)" and rights_replay["training_admitted_rows"] == 0, "successor rights replay link drift")
    require(counts["population_candidate_source_rows_excluded_by_rights"] == population_counts, "successor count-receipt population split drift")
    lineage_rows = lineage["candidate_populations_excluded_by_rights"]
    require(len(lineage_rows) == 2, "successor lineage excluded-population count drift")
    for lineage_row, quarantine_row in zip(lineage_rows, ranges, strict=True):
        require(lineage_row["candidate_source_rows"] == quarantine_row["row_count"] and lineage_row["citation_id"] == quarantine_row["citation_id"] and lineage_row["source_release_id"] == quarantine_row["release_id"] and lineage_row["record_id_format"] == quarantine_row["record_id_format"] and lineage_row["record_id_range"] == quarantine_row["record_id_range"], f"successor lineage/quarantine population link drift: {quarantine_row['release_id']}")
        population_id = quarantine_row["record_id_format"].split(":")[1]
        require(lineage_row["population_id"] == population_id and lineage_row["lineage_group_id_format"] == f"grp:{population_id}:%04d", f"successor lineage group identity drift: {quarantine_row['release_id']}")
    require(lineage["validation"] == {"every_materialized_row_has_admitted_rights": True, "every_materialized_row_has_lineage_group": True, "materialized_row_count": 0}, "successor lineage validation drift")
    inventory = git_json(DATASETS, CAMPAIGN_NESTED_REVISION, "data/ir_learning/source_inventory/release_inventory.json")
    require(rights_replay["artifact_paths"] == ["rights_manifest.json", "quarantine_manifest.json", "source_releases.json", "replay_receipt.json"], "successor rights replay artifact population drift")
    require(rights_replay["input_inventory"] == {"canonical_inventory_sha256": inventory["inventory_sha256"], "path": "ipfs_datasets_py/data/ir_learning/source_inventory/release_inventory.json", "pinset_id": "JDAO-PINSET-1", "source_revision_rule": "Every source release must use the exact 40-hex revision recorded in the inventory; repository visibility, default branches, and card declarations do not admit rights."}, "successor rights replay inventory link drift")
    require(rights_replay["citation_replay"]["expected_exact_revision_count"] == 21 and rights_replay["citation_replay"]["network_execution_required"] is False and rights_replay["decision_replay"]["expected_source_rows"] == 7173 and rights_replay["decision_replay"]["expected_training_admitted_rows"] == 0, "successor rights replay expected populations drift")
    require(len({(row["citation"]["id"], row["citation"]["url"], row["citation"]["observed_revision"], row["citation"]["response_sha256"]) for row in releases["releases"]}) == 21, "successor citation tuple population is not unique")

    expected_counts = {
        "admitted_source_rows": 0,
        "candidate_source_rows_excluded_by_rights": 7173,
        "materialized_derived_artifacts": 0,
        "materialized_source_rows": 0,
        "observed_historical_derived_artifacts": 38690,
    }
    require(manifest["counts"] == counts["counts"] == corpus_root["counts"] == expected_counts, "successor corpus counts cross-link drift")
    require(manifest["result_identity"] == corpus_root["result_identity"] == load["result_identity"] == "RESULT(PGIR-201)", "successor corpus result identity drift")
    require(manifest["input_rights_result_identity"] == corpus_root["source_rights_result_identity"] == load["no_go"]["result_identity"] == "RESULT(PGIR-200)", "successor rights-to-corpus link drift")
    require(manifest["admitted_source_record_ids"] == manifest["materialized_source_record_ids"] == manifest["materialized_derived_artifact_ids"] == [], "successor manifest materialized population is nonempty")
    require(manifest["materialized_source_rows"] == [] and manifest["materialization_status"] == "not_materialized_permanent_no_go", "successor manifest materialization drift")
    require(corpus_root["materialized"] is False and corpus_root["materialized_source_record_ids"] == corpus_root["materialized_derived_artifact_ids"] == [], "successor root materialization drift")
    require(lineage["admitted_lineage_groups"] == lineage["edges"] == lineage["materialized_row_lineage"] == [], "successor lineage graph is nonempty")
    require(all(counts["validation"].values()), "successor count validation drift")
    require(load["deterministic_load"]["status"] == "not_materialized_permanent_no_go" and load["deterministic_load"]["loaded_source_record_ids"] == load["deterministic_load"]["loaded_derived_artifact_ids"] == [], "successor deterministic load drift")
    require(load["input_artifacts"] == ["rights_manifest.json", "quarantine_manifest.json", "source_releases.json", "replay_receipt.json"], "successor load input link population drift")

    require(set(corpus_root["artifacts"]) == set(corpus_names) - {"corpus_root.json"}, "successor corpus root exact eight-artifact population drift")
    for name, row in corpus_root["artifacts"].items():
        data = (corpus_dir / name).read_bytes()
        require(row == {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "content_cid": raw_cid(data)}, f"successor corpus root artifact drift: {name}")
    require(corpus_root["manifest_cid"] == corpus_root["artifacts"]["corpus_manifest.json"]["content_cid"], "successor corpus manifest CID link drift")
    require(corpus_root["lineage_graph_cid"] == corpus_root["artifacts"]["lineage_graph.json"]["content_cid"], "successor lineage CID link drift")
    require(manifest["lineage_graph_id"] == lineage["graph_id"] and manifest["manifest_id"] == corpus_root["manifest_id"], "successor corpus semantic ID links drift")
    require(all(document["pinset_id"] == "JDAO-PINSET-1" for document in (rights, quarantine, releases, manifest, counts, lineage, load, corpus_root)), "successor corpus pinset cross-link drift")

    holdout = split["holdout_report.json"]
    leakage = split["leakage_report.json"]
    split_manifest = split["ir_split_manifest.json"]
    split_replay = split["replay_receipt.json"]
    split_root = split["split_root.json"]
    require(tuple(holdout["in_scope_holdouts"]) == HOLDOUTS, "successor exact 13 in-scope holdouts drift")
    require(set(holdout["holdouts"]) == set(split_manifest["holdouts"]) == set(split_root["holdouts"]) == set(HOLDOUTS) | {"family", "jurisdiction"}, "successor declared holdout population drift")
    for name in HOLDOUTS:
        expected = {"count": 0, "permanent_no_go_reason": "no_rights_admitted_materialized_rows", "split": holdout["holdouts"][name]["split"], "status": "permanent_no_go"}
        require(holdout["holdouts"][name] == split_manifest["holdouts"][name] == split_root["holdouts"][name] == expected, f"successor holdout no-go drift: {name}")
    require(holdout["all_declared_holdouts_resolved"] is True and holdout["no_go"]["permanent"] is True and holdout["no_go"]["authority"] == "RESULT(PGIR-201)", "successor holdout authority drift")
    require(split_manifest["assignments"] == {} and split_manifest["assignment_conflicts"] == {} and split_manifest["examples"] == [], "successor split assignments/examples are nonempty")
    exact_partitions = ["train", "validation", "canary", "holdout", "statute_family", "jurisdiction", "temporal", "external_test", "lineage", "publication", "domain", "notation", "type", "compiler", "proof_library", "premise", "length", "rare_operator", "exception", "cross_reference"]
    require(split_manifest["partition_names"] == exact_partitions and split_manifest["protected_splits"] == exact_partitions[1:] and split_manifest["samples_by_split"] == {name: [] for name in exact_partitions}, "successor split samples/protection population drift")
    require(split_manifest["config"] == {"principal_split": "lineage-group", "seed": "pgir-202-successor-v1"} and split_manifest["config_digest"] == "c654ca8b754f160f687a911230c8c14d053bbd36b68d15b2938a56372535d113", "successor split configuration drift")
    require(split_manifest["split_manifest_digest"] == split_root["split_manifest_digest"] == "38f1a19d90045f2cd58ad92ce52f187d5c90030a81c3bf608162785f6d0226e1", "successor split manifest digest drift")
    require(split_manifest["legal_ir_split_guard"] == {"blocked_operations": [], "passed": True, "violations": []}, "successor split guard drift")
    require(split_manifest["input_corpus"] == {"admitted_source_rows": 0, "corpus_root_path": "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json", "materialized": False, "result_identity": "RESULT(PGIR-201)"} and split_manifest["holdout_report_path"] == "holdout_report.json", "successor split input/path link drift")
    require(split_manifest["metadata"] == {"hidden_test_commitment": HIDDEN, "holdout_resolution": "permanent_no_go", "holdouts": split_manifest["holdouts"], "ir_split_schema": "IRSplitManifest@1", "principal_split": "lineage-group", "seed": "pgir-202-successor-v1"}, "successor split metadata cross-link drift")
    require(leakage["passed"] is True and leakage["violations"] == [] and leakage["blocked_operations"] == [], "successor leakage report drift")
    require(leakage["audit_scope"] == {"assignment_count": 0, "principal_split": "lineage-group", "source_corpus_result_identity": "RESULT(PGIR-201)"} and leakage["leakage_check"] == "passed", "successor leakage audit-scope drift")
    require(split_replay["result_identity"] == split_root["result_identity"] == "RESULT(PGIR-202)" and split_replay["input_corpus"]["result_identity"] == split_root["input_corpus_result_identity"] == "RESULT(PGIR-201)", "successor corpus-to-split result link drift")
    require(split_replay["deterministic_replay"]["expected_assignment_count"] == 0 and split_replay["deterministic_replay"]["expected_materialized_source_rows"] == 0, "successor split replay is nonempty")
    require(split_replay["deterministic_replay"]["expected_leakage_passed"] is True and split_replay["input_corpus"] == {"corpus_manifest_path": "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_manifest.json", "corpus_root_path": "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json", "result_identity": "RESULT(PGIR-201)"}, "successor split replay input link drift")
    require(split_root["leakage_passed"] is True and split_root["status"] == "permanent_no_go", "successor split-root decision drift")
    require(split_root["split_manifest_path"] == "ir_split_manifest.json" and split_root["holdout_report_path"] == "holdout_report.json" and split_root["leakage_report_path"] == "leakage_report.json" and split_root["replay_receipt_path"] == "replay_receipt.json", "successor split-root path links drift")
    require(split_root["supersedes"] == {"historical_split_root_path": "ipfs_datasets_py/data/ir_learning/splits/split_root.json", "mode": "new_generation_without_historical_root_mutation"}, "successor split-root supersession link drift")
    require(holdout["no_go"]["corpus_root_path"] == "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json" and holdout["no_go"]["reason_code"] == "no_rights_admitted_materialized_rows", "successor holdout no-go corpus/reason link drift")
    require(split_root["split_manifest_sha256"] == hashlib.sha256((split_dir / "ir_split_manifest.json").read_bytes()).hexdigest(), "successor split manifest SHA link drift")
    for document in (holdout, split_manifest, split_root):
        require(document["hidden_test_commitment"] == HIDDEN, "successor hidden-test commitment drift")
    require(split_replay["hidden_test_commitment"] == {"commitment": HIDDEN, "predecessor_split_root_path": "ipfs_datasets_py/data/ir_learning/splits/split_root.json", "status": "unchanged_inherited", "verification": "Compare the public commitment string only; do not inspect hidden-test examples or select replacements."}, "successor replay hidden-test commitment drift")
    for document in (holdout, split_manifest, split_root):
        require(document["hidden_test_commitment_status"] == "unchanged_inherited", "successor hidden-test status drift")
    return {
        "candidate_source_rows": len(expanded),
        "population_counts": population_counts,
        "historical_derived_artifacts": 38690,
        "training_eligible_rows": 0,
        "training_admitted_rows": 0,
        "materialized_source_rows": 0,
        "materialized_derived_artifacts": 0,
        "release_count": 21,
        "in_scope_holdout_count": 13,
        "split_assignment_count": 0,
        "hidden_inputs_loaded": False,
    }


def verify_retirement(closure: Mapping[str, Any], pg209: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "data/ir_learning/evaluations/deterministic/successor-v1/"
    expected_paths = ["ipfs_datasets_py/" + prefix + name for name in RETIREMENT_FILES]
    records = closure.get("retirement_files")
    require(isinstance(records, list) and [row.get("path") for row in records] == expected_paths, "exact seven-file PGIR-204 retirement population drift")
    require([row.get("path") for row in pg209["payloads"]] == expected_paths, "PGIR-209 retirement payload order/population drift")
    documents: dict[str, dict[str, Any]] = {}
    for row, source, name in zip(records, pg209["payloads"], RETIREMENT_FILES, strict=True):
        expected = record_for("ipfs_datasets_py/" + prefix + name, DATASETS, CURRENT, prefix + name)
        expected.update({"task_id": "PGIR-204", "result_identity": "RESULT(PGIR-204)"})
        require(row == expected, f"PGIR-204 retirement Git/byte identity drift: {name}")
        require({key: row[key] for key in ("task_id", "result_identity", "path", "size_bytes", "sha256", "raw_cid")} == source, f"PGIR-209 retirement payload link drift: {name}")
        documents[name] = strict_json_bytes(verify_file_record(row), f"retirement/{name}")
        require(documents[name]["result_identity"] == "RESULT(PGIR-204)", f"retirement result identity drift: {name}")
    retirement = documents["retirement_receipt.json"]
    retirement_projection = dict(retirement)
    claimed_retirement = retirement_projection.pop("retirement_cid")
    require(claimed_retirement == dag_cid(retirement_projection), "PGIR-204 retirement self CID drift")
    manifest = documents["manifest.json"]
    manifest_projection = dict(manifest)
    claimed_manifest = manifest_projection.pop("manifest_cid")
    require(claimed_manifest == dag_cid(manifest_projection), "PGIR-204 retirement manifest self CID drift")
    require(set(manifest["artifacts"]) == set(RETIREMENT_FILES) - {"manifest.json"}, "PGIR-204 retirement manifest six-artifact population drift")
    for name, artifact in manifest["artifacts"].items():
        row = next(record for record in records if record["path"].endswith("/" + name))
        expected_artifact = {"content_cid": row["raw_cid"]}
        if name == "retirement_receipt.json":
            expected_artifact["retirement_cid"] = claimed_retirement
        else:
            expected_artifact["schema_cid"] = dag_cid(documents[name])
        require(artifact == expected_artifact, f"PGIR-204 retirement manifest artifact link drift: {name}")
    require(manifest["report_cid"] == retirement["retirement_cid"] == claimed_retirement, "PGIR-204 retirement report link drift")
    identities = documents["identities.json"]
    current_links = {
        "corpus": ("data/ir_learning/corpora/successor-v1/corpus_root.json", "RESULT(PGIR-201)"),
        "rights": ("data/ir_learning/corpora/successor-v1/rights_manifest.json", "RESULT(PGIR-200)"),
        "split": ("data/ir_learning/splits/successor-v1/split_root.json", "RESULT(PGIR-202)"),
    }
    for name, (relative, result_identity) in current_links.items():
        data = (DATASETS / relative).read_bytes()
        link = identities["current_source_inputs"][name]
        require(link == {"path": "ipfs_datasets_py/" + relative, "result_identity": result_identity, "sha256": identity(data)["sha256"], "content_cid": raw_cid(data)}, f"PGIR-204 retirement current-input link drift: {name}")
    r1_manifest = read_json(DATASETS / "data/ir_learning/evaluations/deterministic/manifest.json")
    r1_report = read_json(DATASETS / "data/ir_learning/evaluations/deterministic/r1_baseline.json")
    require(identities["historical_r1"] == {"task_id": "PGIR-023", "identities_cid": read_json(DATASETS / "data/ir_learning/evaluations/deterministic/identities.json")["identities_cid"], "manifest_cid": r1_manifest["manifest_cid"], "report_cid": r1_report["report_cid"]}, "PGIR-204 retirement historical R1 link drift")
    replay = documents["replay_receipt.json"]
    for label, path in {
        "historical_r1_manifest": "data/ir_learning/evaluations/deterministic/manifest.json",
        "historical_r1_report": "data/ir_learning/evaluations/deterministic/r1_baseline.json",
        "successor_corpus_root": "data/ir_learning/corpora/successor-v1/corpus_root.json",
        "successor_holdout_report": "data/ir_learning/splits/successor-v1/holdout_report.json",
        "successor_leakage_report": "data/ir_learning/splits/successor-v1/leakage_report.json",
        "successor_rights_manifest": "data/ir_learning/corpora/successor-v1/rights_manifest.json",
        "successor_split_root": "data/ir_learning/splits/successor-v1/split_root.json",
    }.items():
        require(replay["input_content_cids"][label] == raw_cid((DATASETS / path).read_bytes()), f"PGIR-204 retirement replay input link drift: {label}")
    require(retirement["decision"]["status"] == "retired" and retirement["denominators"]["current_eligible_rows"] == 0 and retirement["acceptance"] == {"current_input_qualified_r1_cid": None, "historical_baseline_retired": True, "satisfied": True}, "PGIR-204 retirement decision drift")
    return {"file_count": 7, "manifest_cid": claimed_manifest, "retirement_cid": claimed_retirement, "decision": "retired"}


def verify_historical_receipt(closure: Mapping[str, Any]) -> dict[str, Any]:
    verify_capture_startup(closure, "historical")
    require(closure.get("schema") == "proof-grounded-ir-learning/integrated-historical-closure/v2", "historical receipt schema drift")
    require(closure.get("task_id") == "PGIR-211" and closure.get("sources") == expected_sources(), "historical receipt source binding drift")
    require(closure.get("task_identity_sources") == expected_task_identity_source_records(), "historical task-identity source population/Git closure drift")
    parse_utc(closure.get("captured_at_utc"), "historical.captured_at_utc")
    require(closure.get("target") == {"outer_commit": TARGET, "outer_tree": TARGET_TREE, "nested_gitlink": CURRENT, "nested_tree": CURRENT_TREE}, "historical receipt target drift")
    pg208, pg209, pg210 = verify_predecessors(closure)
    verify_forest(closure, pg208, pg210)
    r1 = verify_r1(closure)
    retirement = verify_retirement(closure, pg209)
    campaign = verify_campaign(closure)
    successor = verify_successor_population()
    require(closure.get("replay_summary") == {"r1": r1, "retirement": retirement, "campaign": campaign, "successor": successor}, "historical replay summary drift")
    return {"r1": r1, "retirement": retirement, "campaign": campaign, "successor": successor}


def target_identity() -> dict[str, Any]:
    return {"outer_commit": TARGET, "outer_tree": TARGET_TREE, "nested_gitlink": CURRENT, "nested_tree": CURRENT_TREE}


FOCUSED_TEST_ARGV = [
    "/usr/bin/python3.12", "-S", "-m", "pytest", "-q",
    "ipfs_datasets_py/tests/unit/logic/intent_ir/graphrag/test_skillcenter_hf_release.py",
    "ipfs_datasets_py/tests/unit/logic/ir_learning/source/test_corpus_build.py",
    "ipfs_datasets_py/tests/unit/logic/ir_learning/source/test_successor_rights.py",
    "ipfs_datasets_py/tests/unit/optimizers/logic_theorem_optimizer/test_legal_ir_eval_splits.py",
]
SUPPLEMENTARY_TEST_ARGV = [
    "/usr/bin/python3.12", "-S", "-m", "pytest", "-q",
    "ipfs_datasets_py/tests/integration/logic/test_canonical_semantic_roundtrip.py",
]
FOCUSED_COLLECT_ARGV = [
    "/usr/bin/python3.12", "-S", "-m", "pytest", "--collect-only", "-q",
    *FOCUSED_TEST_ARGV[5:],
]
SUPPLEMENTARY_COLLECT_ARGV = [
    "/usr/bin/python3.12", "-S", "-m", "pytest", "--collect-only", "-q",
    *SUPPLEMENTARY_TEST_ARGV[5:],
]
TARGET_IMPORT_PROBE = "import json,ipfs_datasets_py;print(json.dumps({'module_file':ipfs_datasets_py.__file__},sort_keys=True,separators=(',',':')))"


def execution_stdout(execution: Mapping[str, Any], label: str) -> str:
    try:
        return retained_stream_bytes(execution["stdout"], label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"non-UTF-8 stdout for {label}") from exc


def pytest_session_body(stdout: str, count: int, checkout_path: str, label: str) -> tuple[list[str], str]:
    nonempty = [line for line in stdout.splitlines() if line]
    expected_prefix = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0",
        f"rootdir: {Path(checkout_path).resolve() / 'ipfs_datasets_py'}",
        "configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)",
        "asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function",
        f"collected {count} items",
    ]
    require(len(nonempty) >= 7 and nonempty[:6] == expected_prefix, f"{label} pytest session header/configuration drift")
    return nonempty[6:-1], nonempty[-1]


def collected_node_ids(stdout: str, count: int, label: str, checkout_path: str) -> list[str]:
    tree_lines, terminal = pytest_session_body(stdout, count, checkout_path, label)
    summary_pattern = re.compile(r"=+ (\d+) tests? collected in [0-9]+(?:\.[0-9]+)?s =+")
    summary = summary_pattern.fullmatch(terminal)
    require(summary is not None, f"{label} collection terminal summary is absent or malformed")
    expected_count = int(summary.group(1))
    require(expected_count == count, f"{label} collection header/summary population drift")
    stack: dict[int, tuple[str, str]] = {}
    node_ids: list[str] = []
    for line in tree_lines:
        match = re.fullmatch(r"( *)(<(?P<kind>Package|Dir|Module|Class|Function) (?P<name>[^>]+)>)", line)
        require(match is not None and len(match.group(1)) % 2 == 0, f"{label} collection contains a malformed tree row")
        indent = len(match.group(1))
        kind, name = match.group("kind"), match.group("name")
        stack = {level: value for level, value in stack.items() if level < indent}
        if kind != "Function":
            stack[indent] = (kind, name)
            continue
        ordered = [stack[level] for level in sorted(stack)]
        path_parts = [value for item_kind, value in ordered if item_kind in {"Package", "Dir", "Module"}]
        if path_parts[:1] == ["ipfs_datasets_py"]:
            path_parts = path_parts[1:]
        class_parts = [value for item_kind, value in ordered if item_kind == "Class"]
        require(path_parts and path_parts[-1].endswith(".py"), f"{label} function has no module path")
        node_ids.append("/".join(path_parts) + "::" + "::".join([*class_parts, name]))
    require(len(node_ids) == expected_count, f"{label} collection summary/node-id population drift")
    require(len(node_ids) == len(set(node_ids)), f"{label} collection contains duplicate node IDs")
    return node_ids


def strict_passing_outcomes(stdout: str, count: int, label: str, checkout_path: str, expected_paths: Sequence[str]) -> dict[str, int]:
    progress_lines, terminal_text = pytest_session_body(stdout, count, checkout_path, label)
    summary_pattern = re.compile(
        r"=+ (\d+) passed in [0-9]+(?:\.[0-9]+)?s(?: \([0-9]+:[0-9]{2}:[0-9]{2}\))? =+"
    )
    terminal = summary_pattern.fullmatch(terminal_text)
    require(terminal is not None and terminal.group(1) == str(count), f"{label} does not contain one exact {count}-passed terminal summary")
    progress_pattern = re.compile(r"(?:(?P<path>\S+\.py) )?(?P<dots>\.+) +\[(?P<percent>[ 0-9]{3})%\]")
    observed_paths: list[str] = []
    passed_markers = 0
    percentages: list[int] = []
    for line in progress_lines:
        progress = progress_pattern.fullmatch(line)
        require(progress is not None, f"{label} contains output other than exact pytest progress and its terminal summary")
        if progress.group("path"):
            observed_paths.append(progress.group("path"))
        passed_markers += len(progress.group("dots"))
        percentages.append(int(progress.group("percent")))
    require(observed_paths == list(expected_paths), f"{label} progress path population/order drift")
    require(passed_markers == count and percentages and percentages[-1] == 100 and percentages == sorted(percentages), f"{label} progress marker/count/percentage drift")
    return {
        "passed": count,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "deselected": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def verify_worktree_registration(
    execution: Mapping[str, Any],
    *,
    repository: str,
    target_path: str,
    expected_registered: bool,
    label: str,
) -> None:
    verify_execution(execution, stdout_text=True, stderr_text=True)
    require(
        execution.get("argv") == [
            "/usr/bin/git", "-C", repository, "worktree", "list", "--porcelain", "-z",
        ]
        and execution.get("cwd") == repository
        and execution.get("exit_code") == 0,
        f"{label} worktree-registration execution drift",
    )
    paths = [
        str(Path(field.removeprefix("worktree ")).resolve())
        for field in execution_stdout(execution, label).split("\0")
        if field.startswith("worktree ")
    ]
    observed = str(Path(target_path).resolve()) in paths
    require(
        execution.get("target_path") == str(Path(target_path).resolve())
        and execution.get("registered") is expected_registered
        and observed is expected_registered,
        f"{label} worktree-registration status drift",
    )


def verify_clean_target_checkout(checkout: Mapping[str, Any]) -> str:
    require(checkout.get("fresh") is True and checkout.get("initialized_submodules") == ["ipfs_datasets_py"] and checkout.get("recursive_submodule_update") is False and checkout.get("repository_forest_complete_for_task") is True and checkout.get("removed_after_capture") is True, "capture checkout initialization/ephemeral claim drift")
    path, nested_path = checkout.get("path"), checkout.get("nested_path")
    require(isinstance(path, str) and Path(path).is_absolute() and nested_path == str(Path(path) / "ipfs_datasets_py"), "capture checkout paths drift")
    creation = checkout.get("creation")
    removal = checkout.get("removal")
    precreation = checkout.get("precreation")
    nested_preparation = checkout.get("nested_path_preparation")
    nested_absence = checkout.get("nested_absence_after_preparation")
    nested_restoration = checkout.get("nested_path_restoration")
    outer_status_after_restoration = checkout.get("outer_status_after_nested_restoration")
    source_repositories = checkout.get("source_repositories")
    registration_after_creation = checkout.get("registration_after_creation")
    registration_before_removal = checkout.get("registration_before_removal")
    registration_after_removal = checkout.get("registration_after_removal")
    require(isinstance(creation, dict) and set(creation) == {"outer", "nested"}, "capture checkout creation proof missing")
    require(isinstance(removal, dict) and set(removal) == {"nested", "outer"}, "capture checkout removal proof missing")
    require(isinstance(precreation, dict) and set(precreation) == {"outer", "nested"}, "capture checkout precreation absence proof missing")
    require(isinstance(source_repositories, dict) and set(source_repositories) == {"outer", "nested"} and all(Path(value).is_absolute() for value in source_repositories.values()), "capture source repository paths drift")
    require(isinstance(registration_after_creation, dict) and set(registration_after_creation) == {"outer", "nested"}, "capture post-creation registration proof missing")
    require(isinstance(registration_before_removal, dict) and set(registration_before_removal) == {"outer", "nested"}, "capture pre-removal registration proof missing")
    require(isinstance(registration_after_removal, dict) and set(registration_after_removal) == {"outer", "nested"}, "capture post-removal registration proof missing")
    for role in ("outer", "nested"):
        verify_execution(precreation[role], stdout_text=True, stderr_text=True)
        target_path = path if role == "outer" else nested_path
        require(precreation[role]["argv"] == ["/usr/bin/test", "!", "-e", target_path] and precreation[role]["cwd"] == str(Path(path).parent) and precreation[role]["exit_code"] == 0, f"capture {role} path was not proven absent")
        verify_execution(creation[role], stdout_text=True, stderr_text=True)
        require(creation[role]["exit_code"] == 0, f"capture {role} worktree creation failed")
    require(creation["outer"]["argv"] == ["/usr/bin/git", "-C", source_repositories["outer"], "worktree", "add", "--detach", path, TARGET] and creation["outer"]["cwd"] == source_repositories["outer"], "outer target worktree creation argv/cwd drift")
    require(isinstance(nested_preparation, dict), "nested checkout path-preparation proof missing")
    verify_execution(nested_preparation, stdout_text=True, stderr_text=True)
    require(nested_preparation["argv"] == ["/usr/bin/rmdir", nested_path] and nested_preparation["cwd"] == path and nested_preparation["exit_code"] == 0, "nested checkout path-preparation drift")
    require(isinstance(nested_absence, dict), "nested post-preparation absence proof missing")
    verify_execution(nested_absence, stdout_text=True, stderr_text=True)
    require(nested_absence["argv"] == ["/usr/bin/test", "!", "-e", nested_path] and nested_absence["cwd"] == path and nested_absence["exit_code"] == 0, "nested post-preparation absence proof drift")
    require(creation["nested"]["argv"] == ["/usr/bin/git", "-C", source_repositories["nested"], "worktree", "add", "--detach", nested_path, CURRENT] and creation["nested"]["cwd"] == source_repositories["nested"], "nested target worktree creation argv/cwd drift")
    require(isinstance(nested_restoration, dict), "nested empty gitlink path-restoration proof missing")
    verify_execution(nested_restoration, stdout_text=True, stderr_text=True)
    require(nested_restoration["argv"] == ["/usr/bin/mkdir", "--", nested_path] and nested_restoration["cwd"] == path and nested_restoration["exit_code"] == 0, "nested empty gitlink path-restoration proof drift")
    require(isinstance(outer_status_after_restoration, dict), "outer status after nested gitlink path-restoration proof missing")
    verify_execution(outer_status_after_restoration, stdout_text=True, stderr_text=True)
    require(outer_status_after_restoration["argv"] == ["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"] and outer_status_after_restoration["cwd"] == path and outer_status_after_restoration["exit_code"] == 0 and execution_stdout(outer_status_after_restoration, "outer status after nested restoration") == "", "outer status after nested gitlink path-restoration drift")
    for role in ("outer", "nested"):
        target_path = path if role == "outer" else nested_path
        verify_worktree_registration(
            registration_after_creation[role],
            repository=source_repositories[role],
            target_path=target_path,
            expected_registered=True,
            label=f"capture {role} after creation",
        )
        verify_worktree_registration(
            registration_before_removal[role],
            repository=source_repositories[role],
            target_path=target_path,
            expected_registered=True,
            label=f"capture {role} before removal",
        )
        verify_worktree_registration(
            registration_after_removal[role],
            repository=source_repositories[role],
            target_path=target_path,
            expected_registered=False,
            label=f"capture {role} after removal",
        )
    for role in ("nested", "outer"):
        verify_execution(removal[role], stdout_text=True, stderr_text=True)
        target_path = nested_path if role == "nested" else path
        require(removal[role]["argv"] == ["/usr/bin/git", "-C", source_repositories[role], "worktree", "remove", target_path] and removal[role]["cwd"] == source_repositories[role] and removal[role]["exit_code"] == 0, f"capture {role} worktree removal proof drift")
    expected_commands = {
        "outer_head": (["/usr/bin/git", "rev-parse", "HEAD"], TARGET + "\n", path),
        "outer_tree": (["/usr/bin/git", "rev-parse", "HEAD^{tree}"], TARGET_TREE + "\n", path),
        "outer_gitlink": (["/usr/bin/git", "ls-tree", "HEAD", "ipfs_datasets_py"], f"160000 commit {CURRENT}\tipfs_datasets_py\n", path),
        "outer_status": (["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"], "", path),
        "nested_head": (["/usr/bin/git", "rev-parse", "HEAD"], CURRENT + "\n", nested_path),
        "nested_tree": (["/usr/bin/git", "rev-parse", "HEAD^{tree}"], CURRENT_TREE + "\n", nested_path),
        "nested_status": (["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"], "", nested_path),
    }
    for phase in ("before", "after"):
        rows = checkout.get(phase)
        require(isinstance(rows, dict) and set(rows) == set(expected_commands), f"capture checkout {phase} proof population drift")
        for name, (argv, stdout, cwd) in expected_commands.items():
            execution = rows[name]
            verify_execution(execution, stdout_text=True, stderr_text=True)
            require(execution["argv"] == argv and execution["cwd"] == cwd and execution["exit_code"] == 0 and execution_stdout(execution, f"checkout.{phase}.{name}") == stdout, f"capture checkout {phase} proof drift: {name}")
    return path


def verify_capture_interval(receipt: Mapping[str, Any], checkout: Mapping[str, Any], claimed: Sequence[Mapping[str, Any]], label: str) -> None:
    observed_start = parse_utc(receipt.get("observed_start_utc"), f"{label}.observed_start_utc")
    observed_end = parse_utc(receipt.get("observed_end_utc"), f"{label}.observed_end_utc")
    all_executions = (
        list(checkout["precreation"].values())
        + [checkout["creation"]["outer"], checkout["registration_after_creation"]["outer"], checkout["nested_path_preparation"], checkout["nested_absence_after_preparation"], checkout["creation"]["nested"], checkout["registration_after_creation"]["nested"]]
        + list(checkout["before"].values())
        + list(claimed)
        + list(checkout["after"].values())
        + [checkout["registration_before_removal"]["nested"], checkout["removal"]["nested"], checkout["registration_after_removal"]["nested"], checkout["registration_before_removal"]["outer"], checkout["nested_path_restoration"], checkout["outer_status_after_nested_restoration"], checkout["removal"]["outer"], checkout["registration_after_removal"]["outer"]]
    )
    starts = [parse_utc(row["started_at_utc"], f"{label}.execution.start") for row in all_executions]
    ends = [parse_utc(row["ended_at_utc"], f"{label}.execution.end") for row in all_executions]
    require(observed_start <= min(starts) <= max(ends) <= observed_end, f"{label} executions fall outside observation interval")
    precreation_end = max(parse_utc(row["ended_at_utc"], f"{label}.precreation.end") for row in checkout["precreation"].values())
    outer_creation_start = parse_utc(checkout["creation"]["outer"]["started_at_utc"], f"{label}.outer_creation.start")
    outer_creation_end = parse_utc(checkout["creation"]["outer"]["ended_at_utc"], f"{label}.outer_creation.end")
    outer_registered_start = parse_utc(checkout["registration_after_creation"]["outer"]["started_at_utc"], f"{label}.outer_registered.start")
    outer_registered_end = parse_utc(checkout["registration_after_creation"]["outer"]["ended_at_utc"], f"{label}.outer_registered.end")
    preparation_start = parse_utc(checkout["nested_path_preparation"]["started_at_utc"], f"{label}.preparation.start")
    preparation_end = parse_utc(checkout["nested_path_preparation"]["ended_at_utc"], f"{label}.preparation.end")
    nested_absence_start = parse_utc(checkout["nested_absence_after_preparation"]["started_at_utc"], f"{label}.nested_absence.start")
    nested_absence_end = parse_utc(checkout["nested_absence_after_preparation"]["ended_at_utc"], f"{label}.nested_absence.end")
    nested_creation_start = parse_utc(checkout["creation"]["nested"]["started_at_utc"], f"{label}.nested_creation.start")
    creation_end = parse_utc(checkout["creation"]["nested"]["ended_at_utc"], f"{label}.nested_creation.end")
    nested_registered_start = parse_utc(checkout["registration_after_creation"]["nested"]["started_at_utc"], f"{label}.nested_registered.start")
    nested_registered_end = parse_utc(checkout["registration_after_creation"]["nested"]["ended_at_utc"], f"{label}.nested_registered.end")
    before_end = max(parse_utc(row["ended_at_utc"], f"{label}.before.end") for row in checkout["before"].values())
    claimed_start = min(parse_utc(row["started_at_utc"], f"{label}.claimed.start") for row in claimed)
    claimed_end = max(parse_utc(row["ended_at_utc"], f"{label}.claimed.end") for row in claimed)
    after_start = min(parse_utc(row["started_at_utc"], f"{label}.after.start") for row in checkout["after"].values())
    after_end = max(parse_utc(row["ended_at_utc"], f"{label}.after.end") for row in checkout["after"].values())
    cleanup_rows = [
        checkout["registration_before_removal"]["nested"],
        checkout["removal"]["nested"],
        checkout["registration_after_removal"]["nested"],
        checkout["registration_before_removal"]["outer"],
        checkout["nested_path_restoration"],
        checkout["outer_status_after_nested_restoration"],
        checkout["removal"]["outer"],
        checkout["registration_after_removal"]["outer"],
    ]
    cleanup_bounds = [
        (parse_utc(row["started_at_utc"], f"{label}.cleanup.start"), parse_utc(row["ended_at_utc"], f"{label}.cleanup.end"))
        for row in cleanup_rows
    ]
    require(
        precreation_end <= outer_creation_start <= outer_creation_end
        <= outer_registered_start <= outer_registered_end
        <= preparation_start <= preparation_end
        <= nested_absence_start <= nested_absence_end
        <= nested_creation_start <= creation_end
        <= nested_registered_start <= nested_registered_end
        <= before_end <= claimed_start <= claimed_end <= after_start <= after_end
        <= cleanup_bounds[0][0]
        and all(left[1] <= right[0] for left, right in zip(cleanup_bounds, cleanup_bounds[1:])),
        f"{label} absence/create/register/clean-before/execution/clean-after/remove chronology drift",
    )


def verify_test_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    verify_capture_startup(receipt, "tests")
    require(receipt.get("schema") == "proof-grounded-ir-learning/integrated-test-receipt/v2", "test receipt schema drift")
    require(receipt.get("task_id") == "PGIR-211" and receipt.get("target") == target_identity(), "test receipt target drift")
    require(receipt.get("sources") == expected_sources(), "test receipt source binding drift")
    require(receipt.get("controlled_environment") == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": SAFE_PYTHONPATH,
    }, "test execution environment-control drift")
    require(receipt.get("unset_environment") == ["PYTHONHOME", "PYTEST_ADDOPTS"], "test unset-environment boundary drift")
    require(receipt.get("toolchain") == test_toolchain_identity(), "test toolchain distribution closure drift")
    expected_pytest_configuration = {
        "outer": record_for("pytest.ini", ROOT, TARGET),
        "nested": record_for("ipfs_datasets_py/pytest.ini", DATASETS, CURRENT, "pytest.ini"),
    }
    require(receipt.get("pytest_configuration") == expected_pytest_configuration, "test pytest configuration Git/byte identity drift")
    _pytest_oid, nested_pytest_bytes = git_blob(DATASETS, CURRENT, "pytest.ini")
    require(b"-p anyio.pytest_plugin" in nested_pytest_bytes and b"-p pytest_asyncio.plugin" in nested_pytest_bytes, "test plugin-load configuration drift")
    require(parse_utc(receipt.get("observed_start_utc"), "tests.observed_start_utc") <= parse_utc(receipt.get("observed_end_utc"), "tests.observed_end_utc"), "test observation end precedes start")
    checkout = receipt.get("isolated_target_checkout")
    require(isinstance(checkout, dict), "test target checkout proof missing")
    checkout_path = verify_clean_target_checkout(checkout)
    runtime = receipt.get("runtime")
    require(isinstance(runtime, dict) and set(runtime) == {"python", "pytest", "target_import"}, "test runtime capture population drift")
    for row in runtime.values():
        verify_execution(row, stdout_text=True, stderr_text=True)
        require(row["exit_code"] == 0, "test runtime capture failed")
    require(runtime["python"]["cwd"] == runtime["pytest"]["cwd"] == checkout_path, "test runtime version capture used wrong cwd")
    require(runtime["python"]["argv"] == ["/usr/bin/python3.12", "-S", "--version"] and execution_stdout(runtime["python"], "test python version").strip() == "Python 3.12.3", "test Python identity drift")
    require(runtime["pytest"]["argv"] == ["/usr/bin/python3.12", "-S", "-m", "pytest", "--version"] and execution_stdout(runtime["pytest"], "test pytest version").strip() == "pytest 9.1.1", "test pytest identity drift")
    require(runtime["target_import"]["argv"] == ["/usr/bin/python3.12", "-S", "-c", TARGET_IMPORT_PROBE] and runtime["target_import"]["cwd"] == str(Path(checkout_path) / "ipfs_datasets_py"), "target package import probe argv/cwd drift")
    import_probe = strict_json_bytes(retained_stream_bytes(runtime["target_import"]["stdout"], "target import probe"), "target import probe")
    require(import_probe == {"module_file": str(Path(checkout_path) / "ipfs_datasets_py/ipfs_datasets_py/__init__.py")}, "test harness did not resolve ipfs_datasets_py from the clean target checkout")
    collections = receipt.get("collections")
    require(isinstance(collections, list) and len(collections) == 2, "test receipt must contain two independent collection executions")
    expected_collections = (
        ("focused_34_collection", FOCUSED_COLLECT_ARGV, 34),
        ("supplementary_3_collection", SUPPLEMENTARY_COLLECT_ARGV, 3),
    )
    for row, (role, argv, count) in zip(collections, expected_collections, strict=True):
        verify_execution(row, stdout_text=True, stderr_text=True)
        require(row.get("role") == role and row.get("argv") == argv and row.get("cwd") == checkout_path and row.get("exit_code") == 0, f"test collection execution drift: {role}")
        node_ids = collected_node_ids(execution_stdout(row, role), count, role, checkout_path)
        require(len(node_ids) == count and row.get("collected") == count and row.get("node_ids") == node_ids, f"test exact collected population drift: {role}")
        require(row.get("node_id_set") == {"count": count, **identity("".join(f"{node_id}\n" for node_id in node_ids).encode())}, f"test collected node-id identity drift: {role}")
    executions = receipt.get("executions")
    require(isinstance(executions, list) and len(executions) == 2, "test receipt must contain two distinct executions")
    expected = (("focused_34", FOCUSED_TEST_ARGV, 34), ("supplementary_3", SUPPLEMENTARY_TEST_ARGV, 3))
    for index, (row, (role, argv, count)) in enumerate(zip(executions, expected, strict=True)):
        verify_execution(row, stdout_text=True, stderr_text=True)
        require(row.get("role") == role and row.get("argv") == argv and row.get("cwd") == checkout_path, f"test execution identity drift: {role}")
        require(row.get("exit_code") == 0 and row.get("collected") == count, f"test execution did not bind exact collected population: {role}")
        outcomes = strict_passing_outcomes(execution_stdout(row, role), count, role, checkout_path, argv[5:])
        require({key: row.get(key) for key in outcomes} == outcomes, f"test execution outcome population drift: {role}")
        require(row.get("collection_node_id_set") == collections[index]["node_id_set"], f"test execution/collection link drift: {role}")
    require(parse_utc(collections[0]["ended_at_utc"], "focused collection end") <= parse_utc(executions[0]["started_at_utc"], "focused execution start") <= parse_utc(executions[0]["ended_at_utc"], "focused execution end") <= parse_utc(collections[1]["started_at_utc"], "supplementary collection start") <= parse_utc(collections[1]["ended_at_utc"], "supplementary collection end") <= parse_utc(executions[1]["started_at_utc"], "supplementary execution start"), "test collection/execution chronology drift")
    verify_capture_interval(receipt, checkout, list(runtime.values()) + collections + executions, "tests")
    return {
        "focused_collected": 34,
        "focused_passed": 34,
        "supplementary_collected": 3,
        "supplementary_passed": 3,
        "toolchain_integrity_status": receipt["toolchain"]["integrity_status"],
        "toolchain_record_mismatches": receipt["toolchain"]["record_mismatches"],
        "test_results_authority": "observed_behavior_only",
    }


def expected_release_rows() -> list[dict[str, str]]:
    releases = read_json(DATASETS / "data/ir_learning/corpora/successor-v1/source_releases.json")["releases"]
    return [
        {
            "release_id": row["id"],
            "revision": row["revision"],
            "url": row["citation"]["url"],
            "expected_sha256": "sha256:" + row["citation"]["response_sha256"],
        }
        for row in releases
    ]


def verify_network_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    verify_capture_startup(receipt, "network")
    require(receipt.get("schema") == "proof-grounded-ir-learning/integrated-network-capture/v2", "network receipt schema drift")
    require(receipt.get("task_id") == "PGIR-211" and receipt.get("target") == target_identity(), "network receipt target drift")
    require(receipt.get("sources") == expected_sources(), "network receipt source binding drift")
    require(parse_utc(receipt.get("observed_start_utc"), "network.observed_start_utc") <= parse_utc(receipt.get("observed_end_utc"), "network.observed_end_utc"), "network observation end precedes start")
    require(receipt.get("network_execution_required") is True and receipt.get("offline_replay_permitted") is False, "network receipt permits offline completion")
    checkout = receipt.get("isolated_target_checkout")
    require(isinstance(checkout, dict), "network target checkout proof missing")
    checkout_path = verify_clean_target_checkout(checkout)
    rows = receipt.get("responses")
    expected = expected_release_rows()
    require(isinstance(rows, list) and len(rows) == len(expected) == 21, "network receipt exact 21-response population drift")
    for row, frozen in zip(rows, expected, strict=True):
        execution = row.get("execution")
        require(isinstance(execution, dict), f"network execution missing: {frozen['release_id']}")
        verify_execution(execution, stdout_text=True, stderr_text=True)
        argv = ["/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail-with-body", "--header", "Accept-Encoding: identity", "--write-out", "\n%{http_code}", frozen["url"]]
        require(execution.get("argv") == argv and execution.get("cwd") == checkout_path, f"network argv/cwd drift: {frozen['release_id']}")
        require(execution.get("exit_code") == 0 and execution["stderr"]["size_bytes"] == 0, f"network request failed: {frozen['release_id']}")
        require({key: row.get(key) for key in frozen} == frozen, f"network release identity drift: {frozen['release_id']}")
        combined = retained_stream_bytes(execution["stdout"], f"network response/status {frozen['release_id']}")
        require(combined.endswith(b"\n200"), f"network retained HTTP status drift: {frozen['release_id']}")
        body = combined[:-4]
        require(row.get("body") == {"utf8": body.decode("utf-8"), **identity(body)}, f"network retained response-body identity drift: {frozen['release_id']}")
        require(identity(body)["sha256"] == frozen["expected_sha256"], f"network response hash drift: {frozen['release_id']}")
        parsed = strict_json_bytes(body, f"network response {frozen['release_id']}")
        require(row.get("canonical_json_identity") == identity(canonical(parsed)), f"network canonical JSON identity drift: {frozen['release_id']}")
        require(row.get("observed_revision") == parsed.get("sha") == frozen["revision"] and row.get("http_status") == 200, f"network observed revision/status drift: {frozen['release_id']}")
    verify_capture_interval(receipt, checkout, [row["execution"] for row in rows], "network")
    require(receipt.get("response_count") == 21 and receipt.get("all_exact_revision_hashes_matched") is True, "network capture summary drift")
    return {"response_count": 21, "exact_revision_hashes_matched": 21}


def parse_ls_remote(data: bytes, label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in data.decode("utf-8").splitlines():
        fields = line.split("\t")
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{40}", fields[0]) is not None and fields[1].startswith("refs/"), f"invalid remote-ref row in {label}")
        rows.append({"oid": fields[0], "ref": fields[1]})
    require(rows == sorted(rows, key=lambda row: row["ref"]) and len({row["ref"] for row in rows}) == len(rows), f"remote-ref set is not sorted and unique: {label}")
    return rows


def normalized_ref_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    return b"".join(f"{row['oid']}\t{row['ref']}\n".encode("utf-8") for row in rows)


def verify_portability_repository(row: Mapping[str, Any], *, name: str, url: str, candidates: Sequence[tuple[str, Sequence[str]]], missing: Sequence[str]) -> dict[str, Any]:
    require(row.get("repository") == name and row.get("remote_url") == url, f"portability repository identity drift: {name}")
    require(row.get("isolated_bare") is True and row.get("temp_repo_fresh") is True and row.get("removed_after_capture") is True, f"portability fetch was not fresh/bare/ephemeral: {name}")
    bare_path = row.get("bare_repo_path")
    require(isinstance(bare_path, str) and Path(bare_path).is_absolute(), f"portability bare repo path drift: {name}")
    bare_parent = row.get("bare_repo_parent")
    require(isinstance(bare_parent, str) and Path(bare_parent).is_absolute() and str(Path(bare_path).parent) == bare_parent, f"portability bare repo parent drift: {name}")
    precreation, init, bare_check, empty_before, removal = (row.get(key) for key in ("precreation", "init", "bare_check", "empty_refs_before_fetch", "removal"))
    require(all(isinstance(value, dict) for value in (precreation, init, bare_check, empty_before, removal)), f"portability lifecycle execution missing: {name}")
    for execution in (precreation, init, bare_check, empty_before, removal):
        verify_execution(execution, stdout_text=True, stderr_text=True)
        require(execution["exit_code"] == 0, f"portability lifecycle execution failed: {name}")
    require(precreation["argv"] == ["/usr/bin/test", "!", "-e", bare_path] and precreation["cwd"] == bare_parent, f"portability precreation absence proof drift: {name}")
    require(init["argv"] == ["/usr/bin/git", "init", "--bare", bare_path] and init["cwd"] == bare_parent, f"portability init argv/cwd drift: {name}")
    require(bare_check["argv"] == ["/usr/bin/git", "rev-parse", "--is-bare-repository"] and bare_check["cwd"] == bare_path and execution_stdout(bare_check, f"{name}.bare") == "true\n", f"portability bare-repository proof drift: {name}")
    require(empty_before["argv"] == ["/usr/bin/git", "for-each-ref", "--format=%(objectname)%09%(refname)"] and empty_before["cwd"] == bare_path and execution_stdout(empty_before, f"{name}.empty") == "", f"portability repo was not empty before fetch: {name}")
    require(removal["argv"] == ["/usr/bin/rm", "-r", "--", bare_path] and removal["cwd"] == bare_parent and removal["exit_code"] == 0, f"portability removal proof drift: {name}")
    pre, fetch, post, fetched, rev_list = (row.get(key) for key in ("pre_ls_remote", "fetch", "post_ls_remote", "fetched_refs", "rev_list_all"))
    require(all(isinstance(value, dict) for value in (pre, fetch, post, fetched, rev_list)), f"portability executions missing: {name}")
    for execution, retain_stdout in ((pre, True), (fetch, True), (post, True), (fetched, True), (rev_list, True)):
        verify_execution(execution, stdout_text=retain_stdout, stderr_text=True)
        require(execution["exit_code"] == 0, f"portability execution failed: {name}")
        require(execution["cwd"] == bare_path, f"portability execution did not use isolated bare repo: {name}")
    ls_argv = ["/usr/bin/git", "ls-remote", "--refs", "--heads", "--tags", url]
    require(pre["argv"] == post["argv"] == ls_argv, f"portability ls-remote argv drift: {name}")
    require(fetch["argv"] == ["/usr/bin/git", "fetch", "--no-write-fetch-head", url, "+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*"], f"portability fetch argv drift: {name}")
    require(fetched["argv"] == ["/usr/bin/git", "for-each-ref", "--format=%(objectname)%09%(refname)", "refs/remotes/origin", "refs/tags"], f"portability fetched-ref argv drift: {name}")
    require(rev_list["argv"] == ["/usr/bin/git", "rev-list", "--all"], f"portability reachability argv drift: {name}")
    pre_rows = parse_ls_remote(pre["stdout"]["utf8"].encode(), f"{name}.pre")
    post_rows = parse_ls_remote(post["stdout"]["utf8"].encode(), f"{name}.post")
    fetched_rows_raw = parse_ls_remote(fetched["stdout"]["utf8"].encode(), f"{name}.fetched")
    fetched_rows = [
        {"oid": item["oid"], "ref": item["ref"].replace("refs/remotes/origin/", "refs/heads/", 1)}
        for item in fetched_rows_raw
    ]
    fetched_rows.sort(key=lambda item: item["ref"])
    require(pre_rows == post_rows == fetched_rows, f"portability remote ref drift or incomplete fetch: {name}")
    ref_bytes = normalized_ref_bytes(pre_rows)
    require(row.get("normalized_ref_set") == {"count": len(pre_rows), **identity(ref_bytes)}, f"portability normalized ref-set identity drift: {name}")
    require(row.get("pre_equals_post_equals_fetched") is True, f"portability equality verdict drift: {name}")
    candidate_rows = row.get("candidates")
    require(isinstance(candidate_rows, list) and len(candidate_rows) == len(candidates), f"portability candidate population drift: {name}")
    missing_set = set(missing)
    remote_reachable = []
    local_repository = ROOT if name == "outer" else DATASETS
    for observed, (oid, roles) in zip(candidate_rows, candidates, strict=True):
        require(observed.get("oid") == oid and observed.get("source_roles") == list(roles), f"portability candidate identity/role drift: {name}:{oid}")
        require(observed.get("local_object_type") == git(local_repository, "cat-file", "-t", oid) == "commit", f"portability local candidate is not a commit: {name}:{oid}")
        reachable = oid not in missing_set
        require(observed.get("remote_reachable") is reachable, f"portability reachability drift: {name}:{oid}")
        object_check = observed.get("object_check")
        require(isinstance(object_check, dict), f"portability candidate object check missing: {name}:{oid}")
        verify_execution(object_check, stdout_text=True, stderr_text=True)
        require(object_check["cwd"] == bare_path, f"portability object check cwd drift: {name}:{oid}")
        require(object_check.get("argv") == ["/usr/bin/git", "cat-file", "-e", f"{oid}^{{commit}}"] and (object_check.get("exit_code") == 0) is reachable, f"portability candidate object presence drift: {name}:{oid}")
        containment = observed.get("containment_execution")
        require(isinstance(containment, dict) is reachable, f"portability containment execution population drift: {name}:{oid}")
        if reachable:
            verify_execution(containment, stdout_text=True, stderr_text=True)
            require(containment["cwd"] == bare_path, f"portability containment cwd drift: {name}:{oid}")
            require(containment.get("argv") == ["/usr/bin/git", "for-each-ref", "--contains", oid, "--format=%(refname)"] and containment.get("exit_code") == 0, f"portability containment argv/exit drift: {name}:{oid}")
        refs = observed.get("containing_refs")
        require(isinstance(refs, list) and refs == sorted(refs) and len(refs) == len(set(refs)), f"portability containing-ref set drift: {name}:{oid}")
        ref_data = "".join(f"{ref}\n" for ref in refs).encode()
        require(observed.get("containing_ref_set") == {"count": len(refs), **identity(ref_data)}, f"portability containing-ref identity drift: {name}:{oid}")
        require(observed.get("containing_ref_count") == len(refs), f"portability containing-ref count drift: {name}:{oid}")
        require(observed.get("witness_ref") == (refs[0] if refs else None), f"portability witness drift: {name}:{oid}")
        require(bool(refs) is reachable, f"portability containing-ref/reachability disagreement: {name}:{oid}")
        if reachable:
            require(containment["stdout"]["utf8"].splitlines() == refs, f"portability retained containment bytes drift: {name}:{oid}")
        if reachable:
            remote_reachable.append(oid)
    rev_oids = retained_stream_bytes(rev_list["stdout"], f"{name}.rev-list").decode("ascii").splitlines()
    require(all(re.fullmatch(r"[0-9a-f]{40}", oid) for oid in rev_oids) and len(rev_oids) == len(set(rev_oids)), f"portability rev-list output drift: {name}")
    derived_reachable = [oid for oid, _roles in candidates if oid in set(rev_oids)]
    require(derived_reachable == remote_reachable, f"portability rev-list/candidate derivation drift: {name}")
    require(row.get("reachable_candidate_oids") == remote_reachable, f"portability reachable candidate set drift: {name}")
    require(row.get("missing_candidate_oids") == list(missing), f"portability exact missing set drift: {name}")
    require(rev_list.get("reachable_candidate_oids") == remote_reachable and rev_list.get("commit_count") == len(rev_oids), f"portability rev-list derivation drift: {name}")
    executions = [precreation, init, bare_check, empty_before, pre, fetch, post, fetched, rev_list]
    executions.extend(candidate["object_check"] for candidate in candidate_rows)
    executions.extend(candidate["containment_execution"] for candidate in candidate_rows if candidate["containment_execution"] is not None)
    starts = [parse_utc(execution["started_at_utc"], f"{name}.start") for execution in executions]
    ends = [parse_utc(execution["ended_at_utc"], f"{name}.end") for execution in executions]
    observed_start = parse_utc(row.get("observed_start_utc"), f"{name}.observed_start")
    observed_end = parse_utc(row.get("observed_end_utc"), f"{name}.observed_end")
    removal_start = parse_utc(removal["started_at_utc"], f"{name}.removal.start")
    require(observed_start <= min(starts) and max(ends) <= removal_start <= parse_utc(removal["ended_at_utc"], f"{name}.removal.end") <= observed_end, f"portability lifecycle observation interval drift: {name}")
    require(parse_utc(precreation["ended_at_utc"], f"{name}.precreation.end") <= parse_utc(init["started_at_utc"], f"{name}.init.start") <= parse_utc(init["ended_at_utc"], f"{name}.init.end") <= parse_utc(bare_check["started_at_utc"], f"{name}.bare.start") <= parse_utc(empty_before["started_at_utc"], f"{name}.empty.start") <= parse_utc(pre["started_at_utc"], f"{name}.pre.start") <= parse_utc(fetch["started_at_utc"], f"{name}.fetch.start") <= parse_utc(post["started_at_utc"], f"{name}.post.start"), f"portability absence/init/pre/fetch/post chronology drift: {name}")
    return {"candidate_count": len(candidates), "missing_count": len(missing), "ref_count": len(pre_rows)}


def verify_portability_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    verify_capture_startup(receipt, "portability")
    require(receipt.get("schema") == "proof-grounded-ir-learning/integrated-portability-capture/v2", "portability receipt schema drift")
    require(receipt.get("task_id") == "PGIR-211" and receipt.get("target") == target_identity(), "portability target drift")
    require(receipt.get("sources") == expected_sources(), "portability source binding drift")
    observed_start = parse_utc(receipt.get("observed_start_utc"), "portability.observed_start_utc")
    observed_end = parse_utc(receipt.get("observed_end_utc"), "portability.observed_end_utc")
    require(observed_start <= observed_end, "portability observation end precedes start")
    repositories = receipt.get("repositories")
    require(isinstance(repositories, dict) and set(repositories) == {"outer", "nested"}, "portability repository population drift")
    outer = verify_portability_repository(repositories["outer"], name="outer", url=OUTER_REMOTE, candidates=OUTER_CANDIDATES, missing=OUTER_MISSING)
    nested = verify_portability_repository(repositories["nested"], name="nested", url=NESTED_REMOTE, candidates=NESTED_CANDIDATES, missing=NESTED_MISSING)
    repository_starts = [parse_utc(repositories[name]["observed_start_utc"], f"portability.{name}.observed_start") for name in ("outer", "nested")]
    repository_ends = [parse_utc(repositories[name]["observed_end_utc"], f"portability.{name}.observed_end") for name in ("outer", "nested")]
    require(observed_start <= min(repository_starts) and max(repository_ends) <= observed_end, "portability repository observations fall outside receipt interval")
    no_go = portability_no_go_claim(
        repositories["outer"]["missing_candidate_oids"],
        repositories["nested"]["missing_candidate_oids"],
    )
    require(
        {key: receipt.get(key) for key in no_go} == no_go,
        "portability no-go/missing-set/recursive-checkout disposition drift",
    )
    return {"status": "portability_no_go", "outer": outer, "nested": nested}


def verify_component_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    verify_capture_startup(receipt, "component")
    require(receipt.get("schema") == "proof-grounded-ir-learning/integrated-component-verification/v2", "component receipt schema drift")
    require(receipt.get("task_id") == "PGIR-211" and receipt.get("target") == target_identity(), "component target drift")
    require(receipt.get("sources") == expected_sources(), "component source binding drift")
    require(receipt.get("execution_source_location") == "prospective PGIR-211 verifier/evidence from SOURCE_ROOT; all repository inputs resolved through --target-root pointing at the fresh clean immutable target checkout", "component execution/source-location claim drift")
    require(receipt.get("controlled_environment") == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": SAFE_PYTHONPATH,
    }, "component execution environment-control drift")
    require(receipt.get("unset_environment") == ["PYTHONHOME", "PYTEST_ADDOPTS"], "component unset-environment boundary drift")
    require(receipt.get("acceptance_artifact_bound") is False and "integrated_acceptance.json" not in receipt.get("component_inputs", {}), "component receipt is circular")
    expected_inputs = {
        name: {"path": name, **identity((DIR / name).read_bytes())}
        for name in ("README.md", "capture_evidence.py", "historical_closure_receipt.json", "network_receipt.json", "portability_receipt.json", "test_receipt.json")
    }
    require(receipt.get("component_inputs") == expected_inputs, "component input closure drift")
    checkout = receipt.get("isolated_target_checkout")
    require(isinstance(checkout, dict), "component target checkout proof missing")
    checkout_path = verify_clean_target_checkout(checkout)
    execution = receipt.get("execution")
    require(isinstance(execution, dict), "component execution missing")
    verify_execution(execution, stdout_text=True, stderr_text=True)
    argv = execution.get("argv")
    require(isinstance(argv, list) and len(argv) == 6 and argv[:2] == ["/usr/bin/python3.12", "-S"] and argv[3] == "--components-pre-acceptance" and argv[4:] == ["--target-root", checkout_path] and execution.get("cwd") == checkout_path, "component execution argv/cwd drift")
    captured_verifier_path = receipt.get("execution_source_absolute_path")
    require(isinstance(captured_verifier_path, str) and Path(captured_verifier_path).is_absolute() and argv[2] == captured_verifier_path and Path(captured_verifier_path).as_posix().endswith("/" + VERIFIER_RELATIVE), "component capture-time verifier path drift")
    require(execution.get("exit_code") == 0 and execution["stderr"]["size_bytes"] == 0, "component execution failed")
    output = strict_json_bytes(retained_stream_bytes(execution["stdout"], "component stdout"), "component stdout")
    require(output.get("verified") is False and output.get("component_verified") is True and output.get("pgir_205_execution_authorized") is False, "component execution verdict drift")
    require(receipt.get("component_verified") is True and receipt.get("pgir_205_execution_authorized") is False, "component receipt verdict drift")
    verify_capture_interval(receipt, checkout, [execution], "component")
    return {"component_verified": True}


def component_replay() -> dict[str, Any]:
    historical = verify_historical_receipt(read_json(HISTORICAL))
    tests = verify_test_receipt(read_json(DIR / "test_receipt.json"))
    network = verify_network_receipt(read_json(DIR / "network_receipt.json"))
    portability = verify_portability_receipt(read_json(DIR / "portability_receipt.json"))
    return {"historical": historical, "tests": tests, "network_capture": network, "portability": portability}


def verify_static_acceptance() -> dict[str, Any]:
    acceptance = read_json(ACCEPTANCE)
    verify_acceptance(acceptance)
    components = component_replay()
    component = verify_component_receipt(read_json(DIR / "component_verification_receipt.json"))
    require(acceptance.get("component_results") == {**components, "component_verification": component}, "acceptance component-result closure drift")
    no_go = portability_no_go_claim(OUTER_MISSING, NESTED_MISSING)
    summary_keys = ("status", "missing_outer_commits", "missing_nested_commits", "pgir_205_execution_authorized")
    require(acceptance.get("portability_no_go") == {key: no_go[key] for key in summary_keys}, "acceptance portability no-go drift")
    return {**components, "component_verification": component, "acceptance_cid": acceptance["acceptance_cid"]}


def live_network_replay() -> dict[str, Any]:
    """Perform the required active 21-request replay; never use the receipt as a substitute."""
    rows = expected_release_rows()
    matched = 0
    for row in rows:
        argv = [
            "/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail-with-body",
            "--header", "Accept-Encoding: identity", "--write-out", "\n%{http_code}", row["url"],
        ]
        try:
            process = subprocess.run(argv, capture_output=True, timeout=45, env=BASE_SUBPROCESS_ENVIRONMENT, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvidenceError(f"live citation request failed for {row['release_id']}: {exc}") from exc
        require(process.returncode == 0, f"live citation request failed for {row['release_id']}: {process.stderr.decode(errors='replace').strip()}")
        require(process.stdout.endswith(b"\n200"), f"live citation status drift for {row['release_id']}")
        body = process.stdout[:-4]
        require(identity(body)["sha256"] == row["expected_sha256"], f"live citation response hash drift: {row['release_id']}")
        document = strict_json_bytes(body, f"live citation {row['release_id']}")
        require(document.get("sha") == row["revision"], f"live citation revision drift: {row['release_id']}")
        matched += 1
    require(matched == 21, "live citation population incomplete")
    return {"requested": 21, "matched": matched, "transport": "curl --disable HTTPS exact-revision GET under exact minimal environment", "receipt_replay_used_as_substitute": False}


def emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    startup_environment_identity(SOURCE_ROOT / "scripts")
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--network", action="store_true", help="run complete static proof plus 21 live HTTPS requests")
    modes.add_argument("--component", action="store_true", help="run diagnostic static component verification only")
    modes.add_argument("--components-pre-acceptance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (args.network or args.component or args.components_pre_acceptance):
        emit({"verified": False, "component_verified": False, "pgir_205_execution_authorized": False, "error": "live network replay is required; offline completion fails closed"})
        return 1
    try:
        if args.components_pre_acceptance:
            require(args.target_root is not None, "pre-acceptance component verification requires an explicit clean target root")
            configure_target_root(args.target_root)
            require(git(ROOT, "rev-parse", "HEAD") == TARGET and git(ROOT, "rev-parse", "HEAD^{tree}") == TARGET_TREE and gitlink(TARGET) == CURRENT, "component target root is not exact")
            require(git(ROOT, "status", "--porcelain", "--untracked-files=all") == "" and git(DATASETS, "status", "--porcelain", "--untracked-files=all") == "", "component target root is not clean")
            result = component_replay()
            emit({"verified": False, "component_verified": True, "pgir_205_execution_authorized": False, "decision": "permanent_no_go", "target_root_verified": target_identity(), "components": result})
            return 0
        result = verify_static_acceptance()
        if args.component:
            emit({"verified": False, "component_verified": True, "pgir_205_execution_authorized": False, "decision": "permanent_no_go", "acceptance_cid": result["acceptance_cid"]})
            return 0
        network = live_network_replay()
        emit({"verified": True, "component_verified": True, "pgir_205_execution_authorized": False, "completion_authoritative": False, "decision": "permanent_no_go", "acceptance_cid": result["acceptance_cid"], "live_network": network})
        return 0
    except (EvidenceError, KeyError, IndexError, OSError, TypeError, ValueError) as exc:
        emit({"verified": False, "component_verified": False, "pgir_205_execution_authorized": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
