"""PCCE-056 immutable installability and example acceptance gate.

The gate deliberately accepts a completed ``NO-GO`` record when required
installation evidence is unavailable.  It never turns an unavailable input,
an unrun check, or a predecessor's isolated smoke test into release
qualification.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

ACCELERATOR_ROOT = Path(__file__).resolve().parents[2]
BOARD_NAMESPACE = "proof-carrying-context-engine-v0.1"
TASK_RECEIPT_SCHEMA = "lift_coding.proof-carrying-context-engine.task-receipt@1"
QUALIFICATION_SCHEMA = "lift_coding.proof-carrying-context-engine.installability-qualification@1"
ENVIRONMENT_ID = "cpython312-linux-aarch64"
STATIC_TASK_CID = "baguqeeraghkszmidk3bdxsaagnp62palact7ux77cav6ymeollxougppm3aa"
LIVE_TASK_AUTHORITY = {
    "dependency_task_cids": [
        "baguqeera2d56ardt4hdce4b4bgmbkwkhf4nyjbeonf2famwf6hr7ydgivsya",
        "baguqeera2fvluv2phd4wttmcq62o7ya5dijdoe25dwl2elgipdacyg5w7lqq",
    ],
    "revision": 1,
    "task_cid": "baguqeerawocxhxqnvpqggoyuwaclxboph2jta457ri2z3dcnhwt3ufzpdwzq",
}
PROFILE_ORDER = ("core", "verification", "codex", "local-model", "evaluation")
UNSAFE_REQUIREMENT_CLASSES = (
    "editable",
    "local-path",
    "mutable-vcs",
    "unadmitted-direct-url",
)
EMPTY_UNSAFE_REQUIREMENTS = {name: [] for name in UNSAFE_REQUIREMENT_CLASSES}

PREDECESSOR_RECEIPT_SHA256 = {
    "PCCE-045": "640a904ca8837f59d98b5ff71915dd141bd143bf8d52d391abd4f5fe2dcaae7b",
    "PCCE-050": "409fe8a8fdde7f9fc44e7f1d85fb5f9e1e41a16d5de406d628d0a3cce75c43a8",
    "PCCE-051": "d916679ef25a05c615bf77528a171f0968d3d6b0fa757b20ac28b370166ef5b3",
    "PCCE-052": "2800a7059bb7544051d7424086e87b84bb1b97044133648b971c89108340f729",
    "PCCE-053": "7d95896b256251650250e07e5b62698da3496e6daa8e8874f715a2dd82b41819",
    "PCCE-054": "1c5469861f1a90a707c425cbebfaf285aa2ca9344243c7a7145917caae8b1a99",
    "PCCE-055": "a4991d99c007577672abc41c1f08ef27ca87c0d78f7c8d8207e4dd49f96329ad",
    "PCCE-057": "b46382f208cc06ab80aaced4c5a63ecc72f98dca3eb3bd0a75445eed858cedb1",
}
PREDECESSOR_ARTIFACT_IDENTITIES = {
    "PCCE-045": "urn:pcce:task-receipt:PCCE-045:bounded-self-hosting-harness",
    "PCCE-050": "urn:pcce:task-receipt:PCCE-050:datasets-proof-context-package",
    "PCCE-051": "urn:pcce:task-receipt:PCCE-051:kit-proof-context-package",
    "PCCE-052": "urn:pcce:task-receipt:PCCE-052:accelerator-runtime-profiles",
    "PCCE-053": "bafkreidnqdikfqa54om765g4ng7gimjdugdfcar7ydzssai7vmrbbtmaxy",
    "PCCE-054": "bafkreihkjzndxuy5iuhf5fxrmsdy4yhil6le36zzriyqunhkun6kj4fene",
    "PCCE-055": "urn:pcce:task-receipt:PCCE-055:governed-example-walkthrough",
    "PCCE-057": "bafkreifxd75vhhblmfeujlukld7pziwxjwcbmhbbvzq3qpcbzei54mv3em",
}

ENVIRONMENT_FILES = {
    "artifact_hashes.json": {
        "sha256": "b5b38995520aedd3392a205173182dcb07bc43361a5825b53639b985cb460ade",
        "cid_v1_raw": "bafkreifvwoezkuqk5xjtskrakfzrqlola66egnq2las3knrzxgc4wrqk3y",
    },
    "dependency_locks.json": {
        "sha256": "cc3f9268f25a2d06eed0f7a9ede7e227297d8937047be353896c7beca63bc35f",
        "cid_v1_raw": "bafkreigmh6jgr4s2fudo5uhxvhw6pyrhff6ysnyepprvhclmppwkmo6dl4",
    },
    "manifest.json": {
        "sha256": "6d80d0a2c01de399ff74dc69be643123a18651023fc0f329011fab2210cd80be",
        "cid_v1_raw": "bafkreidnqdikfqa54om765g4ng7gimjdugdfcar7ydzssai7vmrbbtmaxy",
    },
    "sbom.spdx.json": {
        "sha256": "39dc1ed3baeec0bec841ba575b1181bd0aa1c8bc6b9e6e2379707de5d5fc75ca",
        "cid_v1_raw": "bafkreibz3qpnhoxoyc7mqqn2k5nrdan5bkq4rpdltzxcg6lqpxs5l7dvzi",
    },
}

SOURCE_COMMITS = {
    "ipfs_accelerate_py": "8b3b08ad3b9705b7981080e5414c6b3aecb06afe",
    "ipfs_datasets_py": "6eb5803548b070a55c81d4631f13996d5c28137b",
    "ipfs_kit_py": "ec95e129bb39a3a18562c6b82ad5f3af3ff1230f",
    "mcp_plus_plus": "0ed2b23d13371a6cae25e5f328a10152e5d1da11",
}

ARCHIVE_IDENTITIES = {
    ("ipfs-datasets-py", "wheel"): {
        "filename": "ipfs_datasets_py-0.2.0-cp312-cp312-linux_aarch64.whl",
        "version": "0.2.0",
        "sha256": "7cd7898808ac5d5db3d587c0f25a4e520858eeb00b6dbe9534cec198aebf5d38",
        "cid_v1_raw": "bafkreid426eyqcfmlvo3hvmhydzfutssbbmo5malnw7jkngoygmk5p25ha",
        "size": 44053405,
        "source_commit": SOURCE_COMMITS["ipfs_datasets_py"],
        "bytes_available": True,
        "bytes_verified": True,
        "cid_binding_status": "bytes-verified",
    },
    ("ipfs-datasets-py", "sdist"): {
        "filename": "ipfs_datasets_py-0.2.0.tar.gz",
        "version": "0.2.0",
        "sha256": "f118b2ccd2b8a7f7262a95981ae05d8432d26b4c4bf3a29338e073c60786b26f",
        "cid_v1_raw": "bafkreihrdczmzuvyu73smkuvtanoaxmegljgwtcl6orjgohaopdapbvsn4",
        "size": 41102914,
        "source_commit": SOURCE_COMMITS["ipfs_datasets_py"],
        "bytes_available": True,
        "bytes_verified": True,
        "cid_binding_status": "bytes-verified",
    },
    ("ipfs-kit-py", "wheel"): {
        "filename": "ipfs_kit_py-0.3.0-py3-none-any.whl",
        "version": "0.3.0",
        "sha256": "f100c569681cfc534c5705b51e89bc2734bfa637200b7f30ebe8dab170bc3be3",
        "cid_v1_raw": "bafkreihradcws2a47rjuyvyfwupitpbhgs72mnzabn7tb27i3kyxbpb34m",
        "size": 7318209,
        "source_commit": SOURCE_COMMITS["ipfs_kit_py"],
        "bytes_available": True,
        "bytes_verified": True,
        "cid_binding_status": "bytes-verified",
    },
    ("ipfs-kit-py", "sdist"): {
        "filename": "ipfs_kit_py-0.3.0.tar.gz",
        "version": "0.3.0",
        "sha256": "8db7299f2cc144814d6b1b01a8476ba2daa67830856513f2863c6fac4af3ed15",
        "cid_v1_raw": "bafkreienw4uz6lgbisau22y3agueo25c3kthqmefmuj7fbr4n6wev47ncu",
        "size": None,
        "source_commit": SOURCE_COMMITS["ipfs_kit_py"],
        "bytes_available": False,
        "bytes_verified": False,
        "cid_binding_status": "identity-derived-bytes-unavailable",
    },
    ("ipfs-accelerate-py", "wheel"): {
        "filename": "ipfs_accelerate_py-0.0.45-py3-none-any.whl",
        "version": "0.0.45",
        "sha256": "ddc10e3e9f484639ce679c26542689df0e873f0a2214df632ff5a9da99c73c65",
        "cid_v1_raw": "bafkreig5yehd5h2iiy444z44ezkcnco7b2dt6crcctpwgl7vvhnjtrz4mu",
        "size": 17826043,
        "source_commit": SOURCE_COMMITS["ipfs_accelerate_py"],
        "bytes_available": True,
        "bytes_verified": True,
        "cid_binding_status": "bytes-verified",
    },
    ("ipfs-accelerate-py", "sdist"): {
        "filename": "ipfs_accelerate_py-0.0.45.tar.gz",
        "version": "0.0.45",
        "sha256": "7f3883460ec5b7980e3944acd0e67833c460837e8ab30647b12adc1fb465608d",
        "cid_v1_raw": "bafkreid7hcbumdwfw6ma4okevtiom6btyrqig7ukwmdepmjk3qp3izlaru",
        "size": 20579108,
        "source_commit": SOURCE_COMMITS["ipfs_accelerate_py"],
        "bytes_available": True,
        "bytes_verified": True,
        "cid_binding_status": "bytes-verified",
    },
    ("mcp-plus-plus-contracts", "wheel"): {
        "filename": "mcp_plus_plus_contracts-0.1.0-py3-none-any.whl",
        "version": "0.1.0",
        "sha256": "d88cc9ba3562f0b592ddfe234810a596c42a4398e32d11e246ea5ebf831a8c9c",
        "cid_v1_raw": "bafkreigyrte3unlc6c2zfxp6enebbjmwyqvehghdfui6erxkl27yggumtq",
        "size": 18516,
        "source_commit": SOURCE_COMMITS["mcp_plus_plus"],
        "bytes_available": True,
        "bytes_verified": True,
        "cid_binding_status": "bytes-verified",
    },
    ("mcp-plus-plus-contracts", "sdist"): {
        "filename": "mcp_plus_plus_contracts-0.1.0.tar.gz",
        "version": "0.1.0",
        "sha256": "4f2c9be7307cc8b55f2f75e24e732b60829a179d6354090306f03ee368966b4e",
        "cid_v1_raw": "bafkreicpfsn6omd4zc2v6l3v4jhhgk3aqknbphldkqeqgbxqh3rwrftljy",
        "size": 20373,
        "source_commit": SOURCE_COMMITS["mcp_plus_plus"],
        "bytes_available": True,
        "bytes_verified": True,
        "cid_binding_status": "bytes-verified",
    },
}

LOCK_IDENTITIES = {
    "core": {
        "lock_sha256": "daf48e32318af4b07cc732b42ab8fe81e52862da3d251146acac45efd514f119",
        "lock_cid_v1_raw": "bafkreig26shdemmk6syhzrzswqvlr7ub4uugfwr5euiunlfmixx5kfhrde",
        "resolver_sha256": "e9e471da36b4ddc29d874a0a5561cfeb651c4e7e501f927aeb6aec79adc8fa43",
        "resolver_cid_v1_raw": "bafkreihj4ry5unvu3xbj3b2kbjkwdt7lmuoe47sqd6jhv23k5r423sh2im",
        "distribution_count": 80,
        "native_build_status": "not-required-by-profile",
    },
    "verification": {
        "lock_sha256": "3e85dbe837b4c6535b32fc69afeca7143936d63a9fed0b47bc101a4aa8ff36c3",
        "lock_cid_v1_raw": "bafkreib6qxn6qn5uyzjvwmx4ngx6zjyuhe3nmou75ufuppaqdjfkr7zwym",
        "resolver_sha256": "0ddc6427a613ddd198ff2994f16d4f43349cd9dbb77082eebbf6dcd008be3595",
        "resolver_cid_v1_raw": "bafkreian3rscpjqt3xizr7zjstyw2t2dgsontw5xocbo5o7w3tiarprvsu",
        "distribution_count": 84,
        "native_build_status": "not-required-by-profile",
    },
    "codex": {
        "lock_sha256": "5f95721c8b5f57d874803811ff9dad8f0b47a406c684ead25b00786ebf5862ec",
        "lock_cid_v1_raw": "bafkreic7svzbzc27k7mhjabych7z3lmpbnd2ibwgqtvnewyapbxl6wdc5q",
        "resolver_sha256": "e4e5ba494b4170924812684c3f1c1e720711ae8d694bd96447ff8675190e27ac",
        "resolver_cid_v1_raw": "bafkreihe4w5ess2bocjeqetijq7ryhtsa4i25dljjpmwir77qz2rsdrhvq",
        "distribution_count": 80,
        "native_build_status": "not-required-by-profile",
    },
    "local-model": {
        "lock_sha256": "332079812d0347d7aade896e59011913cbb754681e3f35c2ce7cf320cb41a904",
        "lock_cid_v1_raw": "bafkreibteb4yclidi7l2vxujnzmqcgitzo3vi2a6h424ftt46mqmwqnjaq",
        "resolver_sha256": "4845cd20ffc09ad4afaa4928e04a33cec379c0bb69e87daeefbf8741be74987d",
        "resolver_cid_v1_raw": "bafkreiciixgsb76atlkk7ksjfdqeum6oyn44bo3j5b625357q5a345eypu",
        "distribution_count": 99,
        "native_build_status": "no-go-native-sdist-only",
    },
    "evaluation": {
        "lock_sha256": "87209db6618f8f9317352e35aa799eee4263bbc0e16ebb1004e7bfd238228618",
        "lock_cid_v1_raw": "bafkreieheco3mympr6jronjogwvhthxoijr3xqhbn25rabhhx7jdqiugda",
        "resolver_sha256": "79675f8c9b51568a023480e3cfa08c05772273b8eedbcb6b61033704242facaf",
        "resolver_cid_v1_raw": "bafkreidzm5pyzg2rk2faenea4ph2bdafo4rhhoho3pfwwyidg4ccil5mv4",
        "distribution_count": 107,
        "native_build_status": "no-go-native-sdist-only",
    },
}


class EvidenceError(AssertionError):
    """Raised when immutable gate evidence is absent or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw_cid_v1(sha256: str) -> str:
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", sha256)), "invalid SHA-256")
    payload = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    return "b" + base64.b32encode(payload).decode("ascii").lower().rstrip("=")


def _identity(raw: bytes) -> dict[str, Any]:
    digest = _sha256(raw)
    return {
        "sha256": digest,
        "cid_v1_raw": _raw_cid_v1(digest),
        "size": len(raw),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _pinned_object(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"missing regular {label}: {path}")
    raw = path.read_bytes()
    actual = _sha256(raw)
    _require(
        actual == expected_sha256,
        f"{label} is not the frozen identity: expected {expected_sha256}, got {actual}",
    )
    return _decode_object(raw, label), raw


def _verify_raw_cid(cid: Any, sha256: Any, label: str) -> None:
    _require(isinstance(cid, str), f"{label} CID must be a string")
    _require(isinstance(sha256, str), f"{label} SHA-256 must be a string")
    _require(cid == _raw_cid_v1(sha256), f"{label} raw CID does not bind its SHA-256")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def _receipt_path(root: Path, task_id: str) -> Path:
    return root / "artifacts" / "proof_carrying_context_engine" / "receipts" / f"{task_id}.json"


def _validate_predecessor_receipts(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    receipts: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for task_id, expected_sha256 in PREDECESSOR_RECEIPT_SHA256.items():
        path = _receipt_path(root, task_id)
        receipt, raw = _pinned_object(path, expected_sha256, f"{task_id} receipt")
        _require_equal(receipt.get("schema"), TASK_RECEIPT_SCHEMA, f"{task_id} schema")
        _require_equal(receipt.get("task_id"), task_id, f"{task_id} task id")
        _require_equal(receipt.get("objective_id"), "PCCE-G500", f"{task_id} objective")
        _require_equal(receipt.get("status"), "completed", f"{task_id} status")
        _require_equal(
            receipt.get("board_namespace"), BOARD_NAMESPACE, f"{task_id} board namespace"
        )
        expected_artifact = PREDECESSOR_ARTIFACT_IDENTITIES[task_id]
        _require_equal(
            receipt.get("artifact_identity"), expected_artifact, f"{task_id} artifact identity"
        )
        identity = _identity(raw)
        receipts[task_id] = receipt
        bindings[task_id] = {
            "artifact_identity": expected_artifact,
            "cid_v1_raw": identity["cid_v1_raw"],
            "path": f"artifacts/proof_carrying_context_engine/receipts/{task_id}.json",
            "sha256": identity["sha256"],
            "size": identity["size"],
        }
    _validate_predecessor_semantics(receipts)
    return receipts, bindings


def _evidence(receipts: dict[str, dict[str, Any]], task_id: str) -> dict[str, Any]:
    evidence = receipts[task_id].get("evidence")
    _require(isinstance(evidence, dict), f"{task_id} evidence must be an object")
    return evidence


def _require_cid(value: Any, label: str) -> None:
    _require(
        isinstance(value, str) and bool(re.fullmatch(r"b[a-z2-7]{58,127}", value)),
        f"{label} must be a CIDv1 base32 string",
    )


def _validate_predecessor_semantics(receipts: dict[str, dict[str, Any]]) -> None:
    harness = _evidence(receipts, "PCCE-045")
    _require_equal(
        harness.get("surface"),
        "ipfs_accelerate_py.agent_supervisor.self_hosting.SelfHostingQualificationHarness",
        "PCCE-045 harness surface",
    )
    _require_equal(
        harness.get("authority"),
        {
            "canonical_branch_mutation": False,
            "qualification": False,
            "self_approval": False,
        },
        "PCCE-045 authority",
    )

    runtime = _evidence(receipts, "PCCE-052")
    _require_equal(
        runtime.get("nested_source", {}).get("implementation_commit"),
        SOURCE_COMMITS["ipfs_accelerate_py"],
        "PCCE-052 implementation commit",
    )
    _require_equal(runtime.get("runtime_authority_widened"), False, "PCCE-052 authority")
    transcript = runtime.get("clean_environment_transcript", {})
    _require_equal(transcript.get("source_tree_on_import_path"), False, "PCCE-052 source path")
    _require_equal(transcript.get("harness_import"), True, "PCCE-052 harness import")
    _require_equal(transcript.get("console_help"), True, "PCCE-052 CLI smoke")

    environment = _evidence(receipts, "PCCE-053")
    _require_equal(
        environment.get("nested_source", {}).get("implementation_commit"),
        "0d2acde7fd0356e1cd944c615b5980cd2e568cc7",
        "PCCE-053 implementation commit",
    )
    _require_equal(environment.get("semantic_surrogates"), [], "PCCE-053 surrogates")
    _require_equal(environment.get("artifact_count"), 8, "PCCE-053 artifact count")
    _require_equal(environment.get("lock_count"), 5, "PCCE-053 lock count")
    _require_equal(
        environment.get("artifact_byte_availability_status"),
        "partial-one-admitted-sdist-unavailable",
        "PCCE-053 byte availability",
    )
    _require_equal(
        environment.get("artifact_clean_install_status"),
        "no-go-sdist-builds-not-qualified",
        "PCCE-053 clean-install status",
    )

    clean = _evidence(receipts, "PCCE-054")
    _require_equal(
        clean.get("nested_source", {}).get("implementation_commit"),
        "4953f09d79dc30149ac44034bfa03c38c0732a63",
        "PCCE-054 implementation commit",
    )
    matrix = clean.get("deterministic_evidence", {}).get("explicit_artifact_root_matrix", {})
    _require_equal(matrix.get("artifact_bytes_verified"), 7, "PCCE-054 verified bytes")
    _require_equal(
        matrix.get("identity_derived_unavailable_artifacts"),
        1,
        "PCCE-054 unavailable artifact count",
    )
    _require_equal(matrix.get("qualified_install_profile_count"), 0, "PCCE-054 qualified profiles")
    _require_equal(matrix.get("no_go_profile_count"), 5, "PCCE-054 no-go profiles")
    _require_equal(matrix.get("require_qualified_exit_code"), 5, "PCCE-054 fail-closed exit")
    _verify_raw_cid(matrix.get("cid_v1_raw"), matrix.get("sha256"), "PCCE-054 matrix")
    _require_equal(
        receipts["PCCE-054"].get("artifact_identity"),
        matrix.get("cid_v1_raw"),
        "PCCE-054 receipt/matrix binding",
    )
    profiles = clean.get("profiles")
    _require(isinstance(profiles, dict), "PCCE-054 profiles must be an object")
    _require_equal(set(profiles), set(PROFILE_ORDER), "PCCE-054 profile names")
    for profile in PROFILE_ORDER:
        _require_equal(profiles[profile].get("result"), "NO-GO", f"PCCE-054 {profile}")
    container = clean.get("container_no_go", {})
    _require_equal(container.get("images_created"), [], "PCCE-054 container images")
    _require_equal(container.get("runtime_user"), "65532:65532", "PCCE-054 runtime user")
    for profile in ("core", "verification"):
        result = container.get("profiles", {}).get(profile, {})
        _require_equal(result.get("build_exit_code"), 1, f"PCCE-054 {profile} build")
        _require_equal(result.get("harness_exit_code"), 5, f"PCCE-054 {profile} harness")
        _require_equal(result.get("image_created"), False, f"PCCE-054 {profile} image")
        _require_equal(
            result.get("result"),
            "explicit-no-go-source-archives-not-qualified",
            f"PCCE-054 {profile} result",
        )
        _verify_raw_cid(
            result.get("cid_v1_raw"), result.get("log_sha256"), f"PCCE-054 {profile} log"
        )
    workflow = clean.get("validation", {}).get("workflow_execution", {})
    _require_equal(workflow.get("dispatched"), False, "PCCE-054 workflow dispatch")
    _require_equal(workflow.get("result"), "not-run-no-success-claimed", "PCCE-054 workflow result")

    example = _evidence(receipts, "PCCE-055")
    walk = example.get("walkthrough", {})
    _require_equal(
        walk.get("schema"),
        "ipfs-accelerate.proof-context.v0.1/example-walkthrough@1",
        "PCCE-055 walkthrough schema",
    )
    transcript_sha = walk.get("transcript_sha256")
    _require(
        isinstance(transcript_sha, str) and transcript_sha.startswith("sha256:"),
        "PCCE-055 transcript SHA-256 is malformed",
    )
    _require_cid(walk.get("transcript_cid"), "PCCE-055 lifecycle transcript")
    _require_equal(example.get("bad_patch", {}).get("status"), "rejected", "PCCE-055 bad patch")
    _require_equal(example.get("acceptance", {}).get("status"), "succeeded", "PCCE-055 acceptance")
    _require_equal(example.get("seal", {}).get("status"), "succeeded", "PCCE-055 seal")
    _require_equal(example.get("seal", {}).get("provenance"), "live", "PCCE-055 provenance")
    _require_cid(example.get("seal", {}).get("seal_cid"), "PCCE-055 seal")

    contracts = _evidence(receipts, "PCCE-057")
    bundle = contracts.get("contract_bundle", {})
    _verify_raw_cid(bundle.get("cid"), bundle.get("sha256"), "PCCE-057 contract bundle")
    _require_equal(bundle.get("runtime_authority"), False, "PCCE-057 runtime authority")
    schemas = contracts.get("schema_cids")
    _require(isinstance(schemas, dict), "PCCE-057 schema CIDs must be an object")
    _require_equal(len(schemas), 17, "PCCE-057 schema count")
    for name, cid in schemas.items():
        _require_cid(cid, f"PCCE-057 schema {name}")
    vectors = contracts.get("canonical_vectors", {})
    _verify_raw_cid(vectors.get("cid"), vectors.get("sha256"), "PCCE-057 vectors")
    clean_contract = contracts.get("clean_environment_transcript", {})
    _require_equal(clean_contract.get("schema_count"), 17, "PCCE-057 installed schema count")
    _require_equal(clean_contract.get("source_tree_on_import_path"), False, "PCCE-057 source path")
    _require_equal(
        clean_contract.get("network_or_installer_effects_at_import"),
        False,
        "PCCE-057 import effects",
    )


def _environment_path(root: Path, name: str) -> Path:
    return root / "artifacts" / "proof_carrying_context_engine" / "environment" / name


def _load_environment_files(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    objects: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for name, expected in ENVIRONMENT_FILES.items():
        value, raw = _pinned_object(_environment_path(root, name), expected["sha256"], name)
        identity = _identity(raw)
        _require_equal(identity["cid_v1_raw"], expected["cid_v1_raw"], f"{name} CID")
        objects[name] = value
        bindings[name] = {
            "cid_v1_raw": identity["cid_v1_raw"],
            "path": f"artifacts/proof_carrying_context_engine/environment/{name}",
            "sha256": identity["sha256"],
            "size": identity["size"],
        }
    return objects, bindings


def _pinned_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"missing regular {label}: {path}")
    raw = path.read_bytes()
    actual = _sha256(raw)
    _require(
        actual == expected_sha256,
        f"{label} is not the frozen identity: expected {expected_sha256}, got {actual}",
    )
    return raw


def _archive_projection(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "bytes_available",
        "bytes_verified",
        "cid_binding_status",
        "cid_v1_raw",
        "distribution",
        "filename",
        "kind",
        "sha256",
        "size",
        "source_commit",
        "version",
    )
    projected = {name: record.get(name) for name in fields}
    if not record.get("bytes_available"):
        projected["unavailability_evidence"] = copy.deepcopy(record.get("unavailability_evidence"))
        projected["unavailability_reason"] = record.get("unavailability_reason")
    return projected


def _validate_archives(
    artifact_hashes: dict[str, Any], receipts: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    _require_equal(
        artifact_hashes.get("schema"),
        "lift_coding.proof-carrying-context-engine.artifact-hashes@1",
        "artifact manifest schema",
    )
    _require_equal(artifact_hashes.get("source_commits"), SOURCE_COMMITS, "artifact source commits")
    _require_equal(artifact_hashes.get("semantic_surrogates"), [], "artifact surrogates")
    _require_equal(
        artifact_hashes.get("artifact_byte_availability_status"),
        "partial-one-admitted-sdist-unavailable",
        "artifact byte availability",
    )
    _require_equal(
        artifact_hashes.get("artifact_clean_install_status"),
        "no-go-sdist-builds-not-qualified",
        "artifact clean-install status",
    )
    _require_equal(
        artifact_hashes.get("resolution_status"),
        "supported-hash-bound",
        "artifact resolution status",
    )

    records = artifact_hashes.get("artifacts")
    _require(isinstance(records, list), "artifact records must be an array")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in records:
        _require(isinstance(item, dict), "each artifact record must be an object")
        key = (item.get("distribution"), item.get("kind"))
        _require(key not in by_key, f"duplicate artifact record: {key}")
        by_key[key] = item
    _require_equal(set(by_key), set(ARCHIVE_IDENTITIES), "artifact population")

    for key, expected in ARCHIVE_IDENTITIES.items():
        actual = by_key[key]
        for field, expected_value in expected.items():
            _require_equal(actual.get(field), expected_value, f"artifact {key} {field}")
        _verify_raw_cid(actual.get("cid_v1_raw"), actual.get("sha256"), f"artifact {key}")

    missing = by_key[("ipfs-kit-py", "sdist")]
    unavailable = missing.get("unavailability_evidence", {})
    _require_equal(
        unavailable.get("bounded_provider_replay_candidate_sha256"),
        "66a52a69252858dc96947aa6d26af14c48e6ed211bcd4170f25ced0ffab516cf",
        "rejected kit sdist replay identity",
    )
    _require_equal(
        unavailable.get("bounded_provider_replay_candidate_size"),
        6893895,
        "rejected kit sdist replay size",
    )
    _require_equal(
        unavailable.get("disposition"), "rejected-hash-mismatch", "kit replay disposition"
    )
    _require(
        isinstance(missing.get("unavailability_reason"), str),
        "missing kit sdist requires an unavailability reason",
    )

    package_receipts = {
        "ipfs-datasets-py": "PCCE-050",
        "ipfs-kit-py": "PCCE-051",
        "ipfs-accelerate-py": "PCCE-052",
        "mcp-plus-plus-contracts": "PCCE-057",
    }
    for distribution, task_id in package_receipts.items():
        receipt_artifacts = _evidence(receipts, task_id).get("artifacts", {})
        for kind in ("wheel", "sdist"):
            receipt_item = receipt_artifacts.get(kind, {})
            record = by_key[(distribution, kind)]
            _require_equal(
                receipt_item.get("filename"), record.get("filename"), f"{task_id} {kind} filename"
            )
            _require_equal(
                receipt_item.get("sha256"), record.get("sha256"), f"{task_id} {kind} SHA-256"
            )
            if receipt_item.get("cid") is not None:
                _require_equal(
                    receipt_item.get("cid"), record.get("cid_v1_raw"), f"{task_id} {kind} CID"
                )

    return [_archive_projection(by_key[key]) for key in sorted(by_key)]


def _validate_inactive_vcs_ledger(
    ledger: dict[str, Any], label: str, *, require_policy_status: bool = True
) -> None:
    if require_policy_status:
        _require_equal(ledger.get("policy_status"), "passed", f"{label} policy status")
    selected = ledger.get("selected_unsafe_requirements_by_class")
    _require_equal(selected, EMPTY_UNSAFE_REQUIREMENTS, f"{label} unsafe requirements")
    _require_equal(
        ledger.get("selected_unsafe_vcs_direct_editable_path_requirements"),
        [],
        f"{label} flattened unsafe requirements",
    )
    inactive = ledger.get("inactive_mutable_vcs_core_metadata")
    _require(isinstance(inactive, list), f"{label} inactive metadata must be an array")
    _require_equal(len(inactive), 3, f"{label} inactive mutable VCS count")
    for item in inactive:
        _require_equal(
            item.get("selection_status"),
            "inactive-unrequested-extra-target-absent",
            f"{label} inactive VCS classification",
        )
        _require_equal(item.get("required_extra_requested"), False, f"{label} extra")
        _require_equal(item.get("target_distribution_selected"), False, f"{label} target")


def _lock_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_clean_install_status": record.get("artifact_clean_install_status"),
        "distribution_count": record.get("distribution_count"),
        "lock": {
            "cid_v1_raw": record.get("cid_v1_raw"),
            "path": f"external/ipfs_accelerate/{record.get('path')}",
            "sha256": record.get("sha256"),
        },
        "native_build_status": record.get("native_build_status"),
        "resolution_status": record.get("resolution_status"),
        "resolver_receipt": {
            "cid_v1_raw": record.get("resolver_receipt_cid_v1_raw"),
            "path": f"external/ipfs_accelerate/{record.get('resolver_receipt_path')}",
            "sha256": record.get("resolver_receipt_sha256"),
        },
        "selected_source_distributions": copy.deepcopy(record.get("selected_source_distributions")),
    }


def _validate_locks(
    root: Path, dependency_locks: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _require_equal(
        dependency_locks.get("schema"),
        "lift_coding.proof-carrying-context-engine.dependency-locks@1",
        "dependency-lock schema",
    )
    _require_equal(dependency_locks.get("environment_id"), ENVIRONMENT_ID, "lock environment")
    _require_equal(
        dependency_locks.get("resolution_status"),
        "supported-hash-bound",
        "lock resolution status",
    )
    _require_equal(
        dependency_locks.get("artifact_clean_install_status"),
        "no-go-sdist-builds-not-qualified",
        "lock clean-install status",
    )
    _require_equal(
        dependency_locks.get("artifact_byte_availability_status"),
        "partial-one-admitted-sdist-unavailable",
        "lock artifact availability",
    )
    records = dependency_locks.get("locks")
    _require(isinstance(records, list), "locks must be an array")
    by_profile = {item.get("profile"): item for item in records if isinstance(item, dict)}
    _require_equal(set(by_profile), set(PROFILE_ORDER), "lock profiles")

    projected: dict[str, dict[str, Any]] = {}
    hygiene_profiles: dict[str, Any] = {}
    accelerator_root = root / "external" / "ipfs_accelerate"
    for profile in PROFILE_ORDER:
        record = by_profile[profile]
        expected = LOCK_IDENTITIES[profile]
        expected_path = f"packaging/proof_context/locks/{ENVIRONMENT_ID}/{profile}.txt"
        expected_resolver_path = (
            f"packaging/proof_context/locks/{ENVIRONMENT_ID}/{profile}.resolver.json"
        )
        _require_equal(record.get("path"), expected_path, f"{profile} lock path")
        _require_equal(
            record.get("resolver_receipt_path"), expected_resolver_path, f"{profile} resolver path"
        )
        for source_field, expected_field in (
            ("sha256", "lock_sha256"),
            ("cid_v1_raw", "lock_cid_v1_raw"),
            ("resolver_receipt_sha256", "resolver_sha256"),
            ("resolver_receipt_cid_v1_raw", "resolver_cid_v1_raw"),
            ("distribution_count", "distribution_count"),
            ("native_build_status", "native_build_status"),
        ):
            _require_equal(
                record.get(source_field), expected[expected_field], f"{profile} {source_field}"
            )
        _require_equal(
            record.get("resolution_status"), "supported-hash-bound", f"{profile} resolution"
        )
        _require_equal(
            record.get("artifact_clean_install_status"),
            "no-go-sdist-builds-not-qualified",
            f"{profile} clean-install status",
        )
        _verify_raw_cid(record.get("cid_v1_raw"), record.get("sha256"), f"{profile} lock")
        _verify_raw_cid(
            record.get("resolver_receipt_cid_v1_raw"),
            record.get("resolver_receipt_sha256"),
            f"{profile} resolver receipt",
        )

        lock_raw = _pinned_bytes(
            accelerator_root / expected_path, expected["lock_sha256"], f"{profile} lock"
        )
        lock_text = lock_raw.decode("utf-8")
        _require("--hash=sha256:" in lock_text, f"{profile} lock is not hash-bound")
        for forbidden in ("--editable", "\n-e ", "file://", " @ git+", "../"):
            _require(forbidden not in lock_text, f"{profile} lock selects forbidden {forbidden!r}")

        resolver, _ = _pinned_object(
            accelerator_root / expected_resolver_path,
            expected["resolver_sha256"],
            f"{profile} resolver receipt",
        )
        _require_equal(resolver.get("profile"), profile, f"{profile} resolver profile")
        _require_equal(
            resolver.get("resolution_status"), "supported-hash-bound", f"{profile} resolver status"
        )
        ledger = resolver.get("requirement_risk_ledger")
        _require(isinstance(ledger, dict), f"{profile} risk ledger must be an object")
        _validate_inactive_vcs_ledger(ledger, f"{profile} resolver")
        projected[profile] = _lock_projection(record)
        hygiene_profiles[profile] = {
            "inactive_mutable_vcs_declaration_count": 3,
            "selected_unsafe_requirements_by_class": copy.deepcopy(EMPTY_UNSAFE_REQUIREMENTS),
        }

    aggregate = dependency_locks.get("requirement_source_risk")
    _require(isinstance(aggregate, dict), "aggregate requirement risk must be an object")
    _require_equal(aggregate.get("policy_status"), "passed", "aggregate risk status")
    for profile in PROFILE_ORDER:
        ledger = aggregate.get("profiles", {}).get(profile)
        _require(isinstance(ledger, dict), f"aggregate {profile} risk ledger is missing")
        _validate_inactive_vcs_ledger(ledger, f"aggregate {profile}", require_policy_status=False)

    hygiene = {
        "editable_dependencies": [],
        "inactive_mutable_vcs_declaration_count_per_profile": 3,
        "mutable_vcs_dependencies": [],
        "policy_status": "passed",
        "profiles": hygiene_profiles,
        "selected_sibling_or_local_path_requirements": [],
        "unadmitted_direct_url_dependencies": [],
    }
    return projected, hygiene


def _validate_environment_bindings(
    objects: dict[str, dict[str, Any]], receipts: dict[str, dict[str, Any]]
) -> None:
    artifact_hashes = objects["artifact_hashes.json"]
    dependency_locks = objects["dependency_locks.json"]
    manifest = objects["manifest.json"]
    environment_receipt = _evidence(receipts, "PCCE-053")

    _require_equal(
        manifest.get("schema"),
        "lift_coding.proof-carrying-context-engine.environment-manifest@1",
        "environment manifest schema",
    )
    _require_equal(manifest.get("environment_id"), ENVIRONMENT_ID, "manifest environment")
    _require_equal(
        manifest.get("package_source_commits"), SOURCE_COMMITS, "manifest source commits"
    )
    for name in ("artifact_hashes.json", "dependency_locks.json", "sbom.spdx.json"):
        expected = ENVIRONMENT_FILES[name]
        _require_equal(manifest.get("evidence", {}).get(name), expected["sha256"], f"{name} SHA")
        _require_equal(
            manifest.get("evidence_cid_v1_raw", {}).get(name),
            expected["cid_v1_raw"],
            f"{name} CID",
        )
    _require_equal(
        environment_receipt.get("output_sha256"),
        {name: value["sha256"] for name, value in ENVIRONMENT_FILES.items()},
        "PCCE-053 output SHA-256 map",
    )
    _require_equal(
        environment_receipt.get("output_cid_v1_raw"),
        {name: value["cid_v1_raw"] for name, value in ENVIRONMENT_FILES.items()},
        "PCCE-053 output CID map",
    )
    _require_equal(
        receipts["PCCE-053"].get("artifact_identity"),
        ENVIRONMENT_FILES["manifest.json"]["cid_v1_raw"],
        "PCCE-053 manifest identity",
    )

    for source in (artifact_hashes, dependency_locks, manifest):
        _require_equal(
            source.get("artifact_byte_availability_status"),
            "partial-one-admitted-sdist-unavailable",
            "environment byte availability",
        )
        _require_equal(
            source.get("artifact_clean_install_status"),
            "no-go-sdist-builds-not-qualified",
            "environment clean-install status",
        )
        _require_equal(
            source.get("resolution_status"),
            "supported-hash-bound",
            "environment resolution status",
        )
    supported = manifest.get("supported_environments")
    _require(
        isinstance(supported, list) and len(supported) == 1, "one environment must be supported"
    )
    _require_equal(
        supported[0].get("support_scope"),
        "hash-bound-dependency-resolution-only",
        "supported environment scope",
    )
    _require_equal(
        supported[0].get("resolution_status"),
        "supported-hash-bound",
        "supported environment resolution",
    )
    _require_equal(manifest.get("indexes", {}).get("additional"), [], "extra indexes")


def _validate_sbom(sbom: dict[str, Any]) -> dict[str, Any]:
    _require_equal(sbom.get("spdxVersion"), "SPDX-2.3", "SBOM SPDX version")
    _require_equal(sbom.get("dataLicense"), "CC0-1.0", "SBOM data license")
    packages = sbom.get("packages")
    relationships = sbom.get("relationships")
    _require(isinstance(packages, list), "SBOM packages must be an array")
    _require(isinstance(relationships, list), "SBOM relationships must be an array")
    _require_equal(len(packages), 107, "SBOM package count")
    _require_equal(len(relationships), 213, "SBOM relationship count")
    direct_names = {
        "ipfs-accelerate-py",
        "ipfs-datasets-py",
        "ipfs-kit-py",
        "mcp-plus-plus-contracts",
    }
    direct = {item.get("name"): item for item in packages if item.get("name") in direct_names}
    _require_equal(set(direct), direct_names, "SBOM direct package population")
    for name in direct_names:
        checksums = direct[name].get("checksums")
        _require(isinstance(checksums, list) and len(checksums) == 1, f"SBOM {name} checksum")
        _require_equal(checksums[0].get("algorithm"), "SHA256", f"SBOM {name} algorithm")
        expected = ARCHIVE_IDENTITIES[(name, "wheel")]["sha256"]
        _require_equal(checksums[0].get("checksumValue"), expected, f"SBOM {name} identity")
    return {
        "data_license": sbom.get("dataLicense"),
        "direct_packages": sorted(direct_names),
        "document_namespace": sbom.get("documentNamespace"),
        "package_count": len(packages),
        "relationship_count": len(relationships),
        "spdx_version": sbom.get("spdxVersion"),
    }


def _validate_hash_gates(
    dependency_locks: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        "codex": {"exit_code": 0, "outcome": "hash-gate-resolution-passed"},
        "core": {"exit_code": 0, "outcome": "hash-gate-resolution-passed"},
        "evaluation": {
            "exit_code": 2,
            "outcome": "no-go-native-sdist-build-backend-unavailable",
        },
        "local-model": {
            "exit_code": 2,
            "outcome": "no-go-native-sdist-build-backend-unavailable",
        },
        "verification": {"exit_code": 0, "outcome": "hash-gate-resolution-passed"},
    }
    first = dependency_locks.get("hash_gate_validation", {})
    second = manifest.get("hash_gate_validation", {})
    _require_equal(first, second, "manifest/lock hash-gate parity")
    profiles = first.get("profiles")
    _require(isinstance(profiles, dict), "hash-gate profiles must be an object")
    for profile, result in expected.items():
        _require_equal(profiles.get(profile, {}).get("exit_code"), result["exit_code"], profile)
        _require_equal(profiles.get(profile, {}).get("outcome"), result["outcome"], profile)
    _require_equal(first.get("selected_archive_identity_count"), 107, "hash-gate archive count")
    return {
        profile: {
            "exit_code": profiles[profile]["exit_code"],
            "outcome": profiles[profile]["outcome"],
        }
        for profile in PROFILE_ORDER
    }


def _validate_task_contract(root: Path) -> dict[str, Any]:
    path = root / "artifacts" / "proof_carrying_context_engine" / "control" / "task_board.json"
    _require(path.is_file(), "immutable task board is missing")
    board = _decode_object(path.read_bytes(), "task board")
    tasks = board.get("tasks")
    _require(isinstance(tasks, list), "task board tasks must be an array")
    matches = [task for task in tasks if task.get("task_id") == "PCCE-056"]
    _require_equal(len(matches), 1, "PCCE-056 task population")
    task = matches[0]
    _require_equal(task.get("canonical_task_cid"), STATIC_TASK_CID, "static PCCE-056 CID")
    _require_equal(task.get("dependencies"), ["PCCE-054", "PCCE-055"], "PCCE-056 dependencies")
    _require_equal(
        task.get("outputs"),
        [
            "external/ipfs_accelerate/test/proof_context/test_installability_gate.py",
            "artifacts/proof_carrying_context_engine/installation/qualification.json",
            "artifacts/proof_carrying_context_engine/receipts/PCCE-056.json",
        ],
        "PCCE-056 outputs",
    )
    metadata = task.get("metadata")
    _require(isinstance(metadata, dict), "PCCE-056 metadata must be an object")
    _require_equal(
        metadata.get("prohibited effects"),
        "Repair packaging/example code in the gate; use source imports; waive failed profiles; "
        "represent an unavailable required check as passed.",
        "PCCE-056 prohibited effects",
    )
    return {
        "dependencies": copy.deepcopy(task["dependencies"]),
        "static_canonical_task_cid": task["canonical_task_cid"],
        "title": task["title"],
    }


def _clean_install_projection(
    receipts: dict[str, dict[str, Any]], locks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    clean = _evidence(receipts, "PCCE-054")
    profiles = clean["profiles"]
    for profile in PROFILE_ORDER:
        _require_equal(
            profiles[profile].get("selected_source_archives"),
            locks[profile]["selected_source_distributions"],
            f"PCCE-054/{profile} source archive parity",
        )
    matrix = clean["deterministic_evidence"]["explicit_artifact_root_matrix"]
    return {
        "artifact_root_matrix": copy.deepcopy(matrix),
        "checks_not_reached": {
            "cli_smoke": "not-run-blocked-by-unqualified-source-archive-builds",
            "installed_distribution_hashes": (
                "not-run-blocked-by-unqualified-source-archive-builds"
            ),
            "package_imports": "not-run-blocked-by-unqualified-source-archive-builds",
            "schema_resource_byte_parity": ("not-run-blocked-by-unqualified-source-archive-builds"),
            "vector_resource_byte_parity": ("not-run-blocked-by-unqualified-source-archive-builds"),
        },
        "container": copy.deepcopy(clean["container_no_go"]),
        "profiles": copy.deepcopy(profiles),
        "qualified_install_profile_count": 0,
        "source_layout_isolation": "passed-isolated-python-I-probe",
        "workflow_execution": copy.deepcopy(clean["validation"]["workflow_execution"]),
    }


def _contract_resource_projection(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    contracts = _evidence(receipts, "PCCE-057")
    return {
        "canonical_vectors": copy.deepcopy(contracts["canonical_vectors"]),
        "clean_install_schema_byte_parity": (
            "not-run-blocked-by-unqualified-source-archive-builds"
        ),
        "clean_install_vector_byte_parity": (
            "not-run-blocked-by-unqualified-source-archive-builds"
        ),
        "contract_bundle": copy.deepcopy(contracts["contract_bundle"]),
        "packaged_resource_smoke": copy.deepcopy(contracts["clean_environment_transcript"]),
        "required_clean_install_resource_parity_satisfied": False,
        "schema_cids": copy.deepcopy(contracts["schema_cids"]),
        "source_to_packaged_resource_parity": "passed-by-PCCE-057",
    }


def _example_projection(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    example = _evidence(receipts, "PCCE-055")
    return {
        "acceptance": copy.deepcopy(example["acceptance"]),
        "bad_patch": copy.deepcopy(example["bad_patch"]),
        "execution_receipts": copy.deepcopy(example["execution_receipts"]),
        "independent_example_workflow_status": "passed-by-PCCE-055",
        "nested_repository": copy.deepcopy(example["nested_repository"]),
        "proof_reuse": copy.deepcopy(example["proof_reuse"]),
        "qualified_clean_environment_replay": (
            "not-run-blocked-by-unqualified-source-archive-builds"
        ),
        "seal": copy.deepcopy(example["seal"]),
        "selected_tests": copy.deepcopy(example["selected_tests"]),
        "walkthrough": copy.deepcopy(example["walkthrough"]),
    }


def _self_hosting_projection(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    harness = _evidence(receipts, "PCCE-045")
    runtime = _evidence(receipts, "PCCE-052")
    return {
        "clean_install_matrix_import": ("not-run-blocked-by-unqualified-source-archive-builds"),
        "harness_surface": harness["surface"],
        "packaged_wheel_no_deps_import_smoke": True,
        "qualification_authority": False,
        "runtime_clean_environment_transcript": copy.deepcopy(
            runtime["clean_environment_transcript"]
        ),
    }


def _load_facts(root: Path) -> dict[str, Any]:
    contract = _validate_task_contract(root)
    receipts, receipt_bindings = _validate_predecessor_receipts(root)
    objects, environment_bindings = _load_environment_files(root)
    _validate_environment_bindings(objects, receipts)
    archives = _validate_archives(objects["artifact_hashes.json"], receipts)
    locks, hygiene = _validate_locks(root, objects["dependency_locks.json"])
    sbom = _validate_sbom(objects["sbom.spdx.json"])
    hash_gates = _validate_hash_gates(objects["dependency_locks.json"], objects["manifest.json"])
    hygiene["source_layout_import_isolation"] = "passed-isolated-python-I-probe"
    environment = {
        "artifact_byte_availability_status": ("partial-one-admitted-sdist-unavailable"),
        "artifact_clean_install_status": "no-go-sdist-builds-not-qualified",
        "artifacts": archives,
        "environment_id": ENVIRONMENT_ID,
        "files": environment_bindings,
        "hash_gate_profiles": hash_gates,
        "identity_only_artifacts": [
            item for item in archives if item["cid_binding_status"] != "bytes-verified"
        ],
        "locks": locks,
        "resolution_status": "supported-hash-bound",
        "sbom": sbom,
        "semantic_surrogates": [],
        "source_commits": copy.deepcopy(SOURCE_COMMITS),
    }
    return {
        "clean_install": _clean_install_projection(receipts, locks),
        "contract_resources": _contract_resource_projection(receipts),
        "dependency_hygiene": hygiene,
        "environment": environment,
        "example": _example_projection(receipts),
        "packaged_self_hosting": _self_hosting_projection(receipts),
        "predecessor_receipts": receipt_bindings,
        "task_contract": contract,
    }


def _expected_qualification(facts: dict[str, Any]) -> dict[str, Any]:
    missing_artifact = facts["environment"]["identity_only_artifacts"][0]
    clean_profiles = facts["clean_install"]["profiles"]
    container_profiles = facts["clean_install"]["container"]["profiles"]
    return {
        "blocking_findings": [
            {
                "code": "required-artifact-bytes-unavailable",
                "disposition": "no substitute or waiver accepted",
                "evidence": copy.deepcopy(missing_artifact),
                "status": "unavailable",
            },
            {
                "code": "supported-clean-install-profiles-unqualified",
                "disposition": "all supported profiles remain release-blocking NO-GO",
                "profiles": {
                    profile: {
                        "result": clean_profiles[profile]["result"],
                        "selected_source_archives": copy.deepcopy(
                            clean_profiles[profile]["selected_source_archives"]
                        ),
                    }
                    for profile in PROFILE_ORDER
                },
                "status": "verification-failed-closed",
            },
            {
                "code": "container-profiles-unqualified",
                "disposition": "no image or container qualification claimed",
                "profiles": {
                    profile: {
                        "cid_v1_raw": container_profiles[profile]["cid_v1_raw"],
                        "image_created": container_profiles[profile]["image_created"],
                        "log_sha256": container_profiles[profile]["log_sha256"],
                        "result": container_profiles[profile]["result"],
                    }
                    for profile in ("core", "verification")
                },
                "status": "verification-failed-closed",
            },
        ],
        "board_namespace": BOARD_NAMESPACE,
        "completion_disposition": "completed-with-explicit-no-go",
        "decision": "NO-GO",
        "downstream": {
            "benchmark_and_security_evidence_design_allowed": True,
            "benchmark_or_security_release_claims_allowed": False,
            "blocked_release_epics": ["PCCE-G600", "PCCE-G700"],
            "release_qualification_allowed": False,
        },
        "evidence": {
            "clean_install": copy.deepcopy(facts["clean_install"]),
            "contract_resources": copy.deepcopy(facts["contract_resources"]),
            "dependency_hygiene": copy.deepcopy(facts["dependency_hygiene"]),
            "environment": copy.deepcopy(facts["environment"]),
            "example": copy.deepcopy(facts["example"]),
            "packaged_self_hosting": copy.deepcopy(facts["packaged_self_hosting"]),
            "predecessor_receipts": copy.deepcopy(facts["predecessor_receipts"]),
        },
        "limitations": [
            (
                "The exact admitted ipfs-kit-py 0.3.0 sdist bytes are unavailable; "
                "the rejected replay is not a semantic surrogate."
            ),
            (
                "Zero of five supported profiles completed a clean immutable-artifact "
                "installation because every lock selects source archives whose builds are "
                "not qualified."
            ),
            (
                "Core and verification container builds failed closed before environment "
                "creation; no image exists."
            ),
            (
                "Clean-install package imports, CLI smoke, installed-distribution hashes, "
                "and schema/vector byte parity were not reached and are not passed."
            ),
            "The clean-install GitHub Actions workflow was not dispatched.",
            (
                "The example independently passed with a live seal, but was not replayed "
                "inside a qualified clean installation."
            ),
        ],
        "objective_id": "PCCE-G500",
        "obligations": [
            {"name": "immutable-predecessor-receipts", "status": "passed"},
            {"name": "environment-file-identities-and-raw-cids", "status": "passed"},
            {"name": "all-required-artifact-bytes", "status": "failed"},
            {"name": "hash-bound-profile-resolution", "status": "passed-with-declared-no-go"},
            {"name": "selected-dependency-hygiene", "status": "passed"},
            {"name": "sbom-completeness", "status": "passed"},
            {"name": "packaged-self-hosting-wheel-smoke", "status": "passed"},
            {"name": "supported-profile-clean-install", "status": "failed"},
            {"name": "clean-install-schema-vector-parity", "status": "not-run-blocked"},
            {"name": "optional-container-profile", "status": "failed"},
            {"name": "independent-example-workflow-and-seal", "status": "passed"},
            {"name": "combined-clean-install-and-example-qualification", "status": "failed"},
        ],
        "release_qualified": False,
        "schema": QUALIFICATION_SCHEMA,
        "subject": "proof-carrying-context-engine-v0.1-installability-and-example-gate",
        "task_authority": {
            "live_quack_duckdb_projection": copy.deepcopy(LIVE_TASK_AUTHORITY),
            "static_board": copy.deepcopy(facts["task_contract"]),
        },
        "task_id": "PCCE-056",
        "waivers": [],
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(completed.returncode == 0, f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def _nested_source(root: Path) -> dict[str, Any]:
    accelerator = root / "external" / "ipfs_accelerate"
    commit = _git(accelerator, "rev-parse", "HEAD^{commit}")
    tree = _git(accelerator, "rev-parse", "HEAD^{tree}")
    blob = _git(
        accelerator,
        "rev-parse",
        "HEAD:test/proof_context/test_installability_gate.py",
    )
    for label, value in (("commit", commit), ("tree", tree), ("test blob", blob)):
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", value)), f"invalid nested {label}")
    return {
        "implementation_commit": commit,
        "repository": "external/ipfs_accelerate",
        "repository_tree": tree,
        "test_blob": blob,
    }


def _expected_receipt(
    facts: dict[str, Any], qualification_raw: bytes, nested_source: dict[str, Any]
) -> dict[str, Any]:
    qualification_identity = _identity(qualification_raw)
    return {
        "artifact_identity": qualification_identity["cid_v1_raw"],
        "artifact_identity_kind": "CIDv1(raw,sha2-256) of qualification.json",
        "board_namespace": BOARD_NAMESPACE,
        "completion_mode": "supervised-acceptance-gate-explicit-no-go",
        "evidence": {
            "blocking_finding_codes": [
                "required-artifact-bytes-unavailable",
                "supported-clean-install-profiles-unqualified",
                "container-profiles-unqualified",
            ],
            "decision": "NO-GO",
            "nested_source": copy.deepcopy(nested_source),
            "predecessor_receipt_cids": {
                task_id: binding["cid_v1_raw"]
                for task_id, binding in facts["predecessor_receipts"].items()
            },
            "qualification": {
                "cid_v1_raw": qualification_identity["cid_v1_raw"],
                "path": "artifacts/proof_carrying_context_engine/installation/qualification.json",
                "schema": QUALIFICATION_SCHEMA,
                "sha256": qualification_identity["sha256"],
                "size": qualification_identity["size"],
            },
            "release_qualified": False,
            "task_authority": copy.deepcopy(LIVE_TASK_AUTHORITY),
            "validation": [
                (
                    "python -m pytest -q "
                    "external/ipfs_accelerate/test/proof_context/"
                    "test_installability_gate.py "
                    "external/ipfs_accelerate/test/proof_context/"
                    "test_example_repository.py"
                )
            ],
            "waivers": [],
        },
        "objective_id": "PCCE-G500",
        "rollback": (
            "Publish the retained NO-GO, block benchmark/security release claims, and "
            "reopen only PCCE-051/PCCE-053/PCCE-054 as appropriate; do not repair "
            "predecessor evidence from the gate worktree."
        ),
        "schema": TASK_RECEIPT_SCHEMA,
        "status": "completed",
        "task_id": "PCCE-056",
    }


def _difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: expected type {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            return f"{path}: missing keys {missing}, extra keys {extra}"
        for key in sorted(expected):
            difference = _difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            difference = _difference(expected_item, actual_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def _validate_qualification_document(actual: dict[str, Any], facts: dict[str, Any]) -> None:
    expected = _expected_qualification(facts)
    difference = _difference(expected, actual)
    _require(difference is None, f"qualification mismatch: {difference}")


def validate_repository_gate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    facts = _load_facts(root)
    qualification_path = (
        root / "artifacts" / "proof_carrying_context_engine" / "installation" / "qualification.json"
    )
    _require(
        qualification_path.is_file() and not qualification_path.is_symlink(),
        "qualification.json is missing or not a regular file",
    )
    qualification_raw = qualification_path.read_bytes()
    qualification = _decode_object(qualification_raw, "qualification.json")
    _require_equal(
        qualification_raw,
        _canonical_json_bytes(qualification),
        "qualification canonical encoding",
    )
    _validate_qualification_document(qualification, facts)

    receipt_path = _receipt_path(root, "PCCE-056")
    _require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "PCCE-056 receipt is missing or not a regular file",
    )
    receipt_raw = receipt_path.read_bytes()
    receipt = _decode_object(receipt_raw, "PCCE-056 receipt")
    _require_equal(receipt_raw, _canonical_json_bytes(receipt), "PCCE-056 canonical encoding")
    expected_receipt = _expected_receipt(facts, qualification_raw, _nested_source(root))
    difference = _difference(expected_receipt, receipt)
    _require(difference is None, f"PCCE-056 receipt mismatch: {difference}")
    return {
        "decision": qualification["decision"],
        "qualification": _identity(qualification_raw),
        "receipt": _identity(receipt_raw),
        "release_qualified": qualification["release_qualified"],
    }


def _discover_outer_root() -> Path | None:
    candidate = ACCELERATOR_ROOT.parent.parent
    embedded = candidate / "external" / "ipfs_accelerate"
    if embedded.is_dir() and embedded.resolve() == ACCELERATOR_ROOT:
        return candidate
    return None


def _synthetic_facts() -> dict[str, Any]:
    profiles = {
        profile: {
            "result": "NO-GO",
            "selected_source_archives": ["selected-source.tar.gz"],
        }
        for profile in PROFILE_ORDER
    }
    container_profiles = {
        profile: {
            "cid_v1_raw": _raw_cid_v1("1" * 64),
            "image_created": False,
            "log_sha256": "1" * 64,
            "result": "explicit-no-go-source-archives-not-qualified",
        }
        for profile in ("core", "verification")
    }
    return {
        "clean_install": {
            "container": {"profiles": container_profiles},
            "profiles": profiles,
        },
        "contract_resources": {"required_clean_install_resource_parity_satisfied": False},
        "dependency_hygiene": {"policy_status": "passed"},
        "environment": {
            "identity_only_artifacts": [
                {
                    "bytes_available": False,
                    "bytes_verified": False,
                    "cid_binding_status": "identity-derived-bytes-unavailable",
                    "filename": "ipfs_kit_py-0.3.0.tar.gz",
                    "sha256": "8db7299f2cc144814d6b1b01a8476ba2daa67830856513f2863c6fac4af3ed15",
                }
            ]
        },
        "example": {"independent_example_workflow_status": "passed-by-PCCE-055"},
        "packaged_self_hosting": {"packaged_wheel_no_deps_import_smoke": True},
        "predecessor_receipts": {"PCCE-054": {"cid_v1_raw": _raw_cid_v1("2" * 64)}},
        "task_contract": {
            "dependencies": ["PCCE-054", "PCCE-055"],
            "static_canonical_task_cid": STATIC_TASK_CID,
            "title": "Seal installability and example acceptance gate",
        },
    }


def test_raw_cid_binds_exact_sha256() -> None:
    digest = _sha256(b"PCCE-056\n")
    cid = _raw_cid_v1(digest)
    _verify_raw_cid(cid, digest, "fixture")
    with pytest.raises(EvidenceError, match="does not bind"):
        _verify_raw_cid(cid, "0" * 64, "fixture")


def test_truthful_no_go_fixture_is_accepted() -> None:
    facts = _synthetic_facts()
    qualification = _expected_qualification(facts)
    _validate_qualification_document(qualification, facts)
    assert qualification["decision"] == "NO-GO"
    assert qualification["release_qualified"] is False
    assert qualification["waivers"] == []


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("decision",), "GO"),
        (("release_qualified",), True),
        (("waivers",), [{"reason": "treat unavailable as passed"}]),
        (("downstream", "benchmark_or_security_release_claims_allowed"), True),
        (
            (
                "evidence",
                "environment",
                "identity_only_artifacts",
                0,
                "bytes_verified",
            ),
            True,
        ),
        (("evidence", "clean_install", "profiles", "core", "result"), "GO"),
        (
            (
                "evidence",
                "contract_resources",
                "required_clean_install_resource_parity_satisfied",
            ),
            True,
        ),
    ],
)
def test_dishonest_or_waived_no_go_is_rejected(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    facts = _synthetic_facts()
    qualification = _expected_qualification(facts)
    cursor: Any = qualification
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = replacement
    with pytest.raises(EvidenceError, match="qualification mismatch"):
        _validate_qualification_document(qualification, facts)


def test_missing_qualification_evidence_is_rejected() -> None:
    facts = _synthetic_facts()
    qualification = _expected_qualification(facts)
    del qualification["evidence"]["predecessor_receipts"]
    with pytest.raises(EvidenceError, match="missing keys"):
        _validate_qualification_document(qualification, facts)


def test_self_consistent_substitute_is_not_a_frozen_identity(tmp_path: Path) -> None:
    original = _canonical_json_bytes({"frozen": True})
    expected_sha256 = _sha256(original)
    substitute = _canonical_json_bytes({"frozen": False})
    claimed = _identity(substitute)
    _verify_raw_cid(claimed["cid_v1_raw"], claimed["sha256"], "substitute")
    path = tmp_path / "evidence.json"
    path.write_bytes(substitute)
    with pytest.raises(EvidenceError, match="not the frozen identity"):
        _pinned_object(path, expected_sha256, "fixture evidence")


def test_repository_installability_qualification() -> None:
    root = _discover_outer_root()
    if root is None:
        pytest.skip("nested implementation worktree is not embedded in an outer checkout")
    result = validate_repository_gate(root)
    assert result["decision"] == "NO-GO"
    assert result["release_qualified"] is False


def _main() -> int:
    parser = argparse.ArgumentParser(description="Render the exact PCCE-056 outer artifact")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--render", choices=("qualification", "receipt"), default="qualification")
    arguments = parser.parse_args()
    facts = _load_facts(arguments.root.resolve())
    qualification = _expected_qualification(facts)
    if arguments.render == "qualification":
        value = qualification
    else:
        value = _expected_receipt(
            facts,
            _canonical_json_bytes(qualification),
            _nested_source(arguments.root.resolve()),
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
