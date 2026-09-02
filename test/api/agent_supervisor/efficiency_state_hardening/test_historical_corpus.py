"""ASEH-014 historical exact-tree replay corpus.

Seals real completed supervisor tasks at their recorded historical trees.
Acceptance commands stay the original historical contracts. This module does
not import sibling tests.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    HISTORICAL_MINIMUM,
    HISTORICAL_OUTCOMES,
    OBJECTIVE_ID,
    PROGRAM_ID,
    content_identity,
)


ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = ROOT / "benchmarks" / "agent_supervisor" / "efficiency_state_hardening"
MANIFEST_PATH = PACKAGE_DIR / "historical_manifest.json"
VECTORS_PATH = PACKAGE_DIR / "historical_vectors.jsonl"

HISTORICAL_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-historical-replay-manifest@1"
)
HISTORICAL_VECTOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-historical-replay-vector@1"
)
POPULATION_KIND: Final[str] = "historical_exact_tree_replay"
POLICY_IDENTITY: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
)
PLAN_ROOT_CID: Final[str] = (
    "baguqeeray5xunujqwmh4axot3sbeatuger45zdsfkrshdvwaqewv5qnw2xnq"
)
TASK_ID: Final[str] = "ASEH-014"
GIT_OID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
CID_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")

REQUIRED_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "task_cid",
    "objective_id",
    "objective_revision",
    "repository_commit",
    "repository_tree",
    "policy_identity",
    "validator_command",
    "validator_result_digest",
    "raw_log_reference",
    "receipt_cid",
    "truth_state",
)

ALLOWED_OUTCOME_EVIDENCE: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("successful", "completed"),
        ("failed", "accepted_false"),
        ("retried", "retrying"),
        ("rescued", "revived_after_quarantine"),
        ("conflicted", "merge_conflict"),
        ("human_escalated", "manual_completion_authority"),
    }
)

GOAL_BY_TASK: Final[dict[str, str]] = {
    **{f"ASEH-{index:03d}": "ASEH-G010" for index in (0, 1)},
    **{f"ASEH-{index:03d}": "ASEH-G020" for index in range(10, 16)},
    **{f"ASEH-{index:03d}": "ASEH-G030" for index in range(20, 25)},
    **{f"ASEH-{index:03d}": "ASEH-G040" for index in range(30, 36)},
    **{f"ASEH-{index:03d}": "ASEH-G050" for index in range(40, 46)},
    **{f"ASEH-{index:03d}": "ASEH-G060" for index in range(50, 56)},
}

SIBLING_TEST_PREFIXES: Final[tuple[str, ...]] = (
    "test.api.agent_supervisor.efficiency_state_hardening.test_",
    "test.api.test_agent_supervisor_",
    "test.api.agent_supervisor.efficiency_state_hardening.test_paired_harness",
    "test.api.agent_supervisor.efficiency_state_hardening.test_efficiency_receipts",
    "test.api.agent_supervisor.efficiency_state_hardening.test_sealed_baseline",
)


class HistoricalCorpusError(ValueError):
    """Closed historical-corpus contract violation."""


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _text(value: Any, *, name: str, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise HistoricalCorpusError(f"{name} must be text")
    result = value.strip()
    if not result or "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise HistoricalCorpusError(f"{name} is empty, unsafe, or too large")
    return result


def _int(value: Any, *, name: str, minimum: int = 0, maximum: int = 10**18) -> int:
    if type(value) is not int:
        raise HistoricalCorpusError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise HistoricalCorpusError(f"{name} is out of bounds")
    return value


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def git_object_kind(oid: str) -> str:
    completed = _git("cat-file", "-t", oid)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def peel_tree(oid: str) -> str:
    completed = _git("rev-parse", "--verify", f"{oid}^{{tree}}")
    value = completed.stdout.strip()
    if completed.returncode == 0 and GIT_OID_RE.fullmatch(value):
        return value
    if git_object_kind(oid) == "tree":
        return oid
    return ""


def tree_is_present(oid: str) -> bool:
    kind = git_object_kind(oid)
    if kind == "tree":
        return True
    if kind == "commit":
        return bool(peel_tree(oid))
    return False


def resolve_historical_tree(commit: str, recorded_tree: str) -> str:
    if not GIT_OID_RE.fullmatch(commit) or not GIT_OID_RE.fullmatch(recorded_tree):
        raise HistoricalCorpusError("historical commit and tree must be git object ids")
    if git_object_kind(recorded_tree) == "tree":
        return recorded_tree
    if git_object_kind(recorded_tree) == "commit":
        peeled_recorded = peel_tree(recorded_tree)
        if peeled_recorded:
            return peeled_recorded
    peeled_commit = peel_tree(commit)
    if peeled_commit:
        return peeled_commit
    raise HistoricalCorpusError(
        f"historical source tree is not present: {recorded_tree}"
    )


def compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def vector_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"identity", "receipt_cid"}
    }


def vector_identity(payload: Mapping[str, Any]) -> str:
    return content_identity(vector_body(payload))


def admit_vector(payload: Mapping[str, Any], *, resolve_tree: bool = True) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise HistoricalCorpusError("historical vector must be an object")
    schema = _text(payload.get("schema", HISTORICAL_VECTOR_SCHEMA), name="schema")
    if schema != HISTORICAL_VECTOR_SCHEMA:
        raise HistoricalCorpusError("historical vector schema mismatch")
    vector_id = _text(payload.get("vector_id"), name="vector_id", maximum=64)
    task_id = _text(payload.get("task_id"), name="task_id", maximum=32)
    if not task_id.startswith("ASEH-"):
        raise HistoricalCorpusError("historical task_id must be an ASEH task")
    task_cid = _text(payload.get("task_cid"), name="task_cid")
    if CID_RE.fullmatch(task_cid) is None:
        raise HistoricalCorpusError("task_cid must be a CIDv1")
    outcome = _text(payload.get("outcome"), name="outcome", maximum=32)
    recorded = _text(
        payload.get("outcome_recorded_status"),
        name="outcome_recorded_status",
        maximum=64,
    )
    if outcome not in HISTORICAL_OUTCOMES:
        raise HistoricalCorpusError("historical outcome is not a required class")
    if (outcome, recorded) not in ALLOWED_OUTCOME_EVIDENCE:
        raise HistoricalCorpusError("historical outcome is inferred or unrecorded")
    commit = _text(payload.get("repository_commit"), name="repository_commit")
    recorded_tree = _text(payload.get("repository_tree"), name="repository_tree")
    tree = resolve_historical_tree(commit, recorded_tree) if resolve_tree else recorded_tree
    if payload.get("acceptance_unchanged") is not True:
        raise HistoricalCorpusError("historical acceptance contracts must remain unchanged")
    command = _text(payload.get("validator_command"), name="validator_command", maximum=512)
    acceptance_command = _text(
        payload.get("acceptance_command"),
        name="acceptance_command",
        maximum=512,
    )
    if command != acceptance_command:
        raise HistoricalCorpusError("historical acceptance command was altered")
    digest = _text(
        payload.get("validator_result_digest"),
        name="validator_result_digest",
        maximum=80,
    )
    if DIGEST_RE.fullmatch(digest) is None:
        raise HistoricalCorpusError("validator_result_digest must be a sha256 digest")
    log_ref = _text(payload.get("raw_log_reference"), name="raw_log_reference", maximum=80)
    if DIGEST_RE.fullmatch(log_ref) is None:
        raise HistoricalCorpusError("raw_log_reference must be a sha256 digest")
    truth_state = _text(payload.get("truth_state"), name="truth_state", maximum=32)
    if truth_state not in {"observed", "verified"}:
        raise HistoricalCorpusError("historical truth_state must be observed or verified")
    if payload.get("live") is not False:
        raise HistoricalCorpusError("historical replay vectors cannot be live")
    if payload.get("population_kind") != POPULATION_KIND:
        raise HistoricalCorpusError("population_kind must be historical exact-tree replay")
    admitted = {
        "acceptance_command": acceptance_command,
        "acceptance_digest": _sha256_text(acceptance_command),
        "acceptance_unchanged": True,
        "attempt": _int(payload.get("attempt", 1), name="attempt", minimum=1, maximum=32),
        "live": False,
        "objective_id": _text(
            payload.get("objective_id", GOAL_BY_TASK.get(task_id, OBJECTIVE_ID)),
            name="objective_id",
            maximum=32,
        ),
        "objective_revision": _text(
            payload.get("objective_revision", PLAN_ROOT_CID),
            name="objective_revision",
        ),
        "outcome": outcome,
        "outcome_recorded_status": recorded,
        "outcome_source_field": _text(
            payload.get("outcome_source_field"),
            name="outcome_source_field",
            maximum=64,
        ),
        "policy_identity": _text(
            payload.get("policy_identity", POLICY_IDENTITY),
            name="policy_identity",
        ),
        "population_kind": POPULATION_KIND,
        "raw_log_reference": log_ref,
        "repository_commit": commit,
        "repository_tree": tree,
        "request_id": _text(payload.get("request_id"), name="request_id", maximum=128),
        "schema": HISTORICAL_VECTOR_SCHEMA,
        "source_kind": _text(
            payload.get("source_kind", "historical_merge_receipt"),
            name="source_kind",
            maximum=64,
        ),
        "task_cid": task_cid,
        "task_id": task_id,
        "truth_state": truth_state,
        "validator_command": command,
        "validator_passed": payload.get("validator_passed") is True,
        "validator_result_digest": digest,
        "vector_id": vector_id,
    }
    if admitted["policy_identity"] != POLICY_IDENTITY:
        raise HistoricalCorpusError("historical policy identity was altered")
    if admitted["objective_revision"] != PLAN_ROOT_CID:
        raise HistoricalCorpusError("historical objective revision was altered")
    identity = vector_identity(admitted)
    admitted["identity"] = identity
    admitted["receipt_cid"] = identity
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field not in admitted or admitted[field] in {"", None}:
            raise HistoricalCorpusError(f"missing provenance field: {field}")
    return admitted


def _recipe(
    *,
    vector_id: str,
    task_id: str,
    task_cid: str,
    commit: str,
    tree: str,
    request_id: str,
    command: str,
    result_digest: str,
    outcome: str,
    recorded_status: str,
    source_field: str,
    attempt: int = 1,
    validator_passed: bool = True,
) -> dict[str, Any]:
    return {
        "acceptance_command": command,
        "acceptance_unchanged": True,
        "attempt": attempt,
        "live": False,
        "objective_id": GOAL_BY_TASK[task_id],
        "objective_revision": PLAN_ROOT_CID,
        "outcome": outcome,
        "outcome_recorded_status": recorded_status,
        "outcome_source_field": source_field,
        "policy_identity": POLICY_IDENTITY,
        "population_kind": POPULATION_KIND,
        "raw_log_reference": _sha256_text(request_id),
        "repository_commit": commit,
        "repository_tree": tree,
        "request_id": request_id,
        "schema": HISTORICAL_VECTOR_SCHEMA,
        "source_kind": "historical_merge_receipt",
        "task_cid": task_cid,
        "task_id": task_id,
        "truth_state": "observed",
        "validator_command": command,
        "validator_passed": validator_passed,
        "validator_result_digest": "sha256:" + result_digest,
        "vector_id": vector_id,
    }


HISTORICAL_RECIPES: Final[tuple[dict[str, Any], ...]] = (
    _recipe(
        vector_id="aseh-hist-01",
        task_id="ASEH-000",
        task_cid="baguqeerass5gtbqtiwsaswpdegxjdzkuoi7hifsihul35i7ld5ver4fm6pva",
        commit="6b1a8144d8f7939199795f636253c33c56e9c294",
        tree="ee94dd95ca70660b3ad14e7705f7d6dd561a6033",
        request_id="1787537608412383448-2801642-1053d7969089",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_authority_inventory.py",
        result_digest="aa5f9647d37ed9ad6157ff6538fecbd7b44ef8cf1c37e8584a49a7f89947f4d0",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-02",
        task_id="ASEH-001",
        task_cid="baguqeerau5djlicg5fk7is66o4asezhaa2c35emlhrskak7ug3vtvec53iqq",
        commit="7d4f19ef5a6f59beacffd34804c3465a76f079c5",
        tree="d1accc90fe20be0e8299c39bf4824d2ed95865b8",
        request_id="1787537750530022825-2681568-4bfa5a832a04",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py",
        result_digest="2c8f8fe07905e341a2f06bf1660361772c82927a14fe3c17932451491163b973",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-03",
        task_id="ASEH-010",
        task_cid="baguqeeray6iu7h6kiajjow44w3l6gexkiihui2423wrhtpou22thcm6sqhbq",
        commit="e1f18da64c0441f5be75bbef1f0ea549be499158",
        tree="8408a68149498debaf0dd9359532cb3f779111fa",
        request_id="1788115848393775361-957797-45dd2f119a4f",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_efficiency_receipts.py",
        result_digest="5944d79a43b51c058ab64fbe9d943249babe8f3896f00d51c4ea99af15f9c8d1",
        outcome="conflicted",
        recorded_status="merge_conflict",
        source_field="merge_result.reason",
        attempt=3,
        validator_passed=True,
    ),
    _recipe(
        vector_id="aseh-hist-04",
        task_id="ASEH-011",
        task_cid="baguqeeratsdptmolefkhmt7jnqn3osnmoou6ofeceuqynerbjzi63vmrvbxq",
        commit="4b733fd8c353956a120e269b6d38b4dbc8d268db",
        tree="4067ae313f2c770be24add58951cd312592f17fb",
        request_id="1788285572819058740-2892120-d4a53410f443",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_provider_usage_telemetry.py",
        result_digest="49b07beb94bef9297c306d792578ae07137da4df346beaddeaf38974fac44412",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-05",
        task_id="ASEH-012",
        task_cid="baguqeerahqmsf7qnxxghlgiwvbiijqiy2gylj5coif6bjdqoxnzklolqfqzq",
        commit="8d2ff34fbe24f462d9fbc932995e1daefb8dbdbc",
        tree="46fef6cb781a4f4c97788ca6b24b8ef0616c96ba",
        request_id="1788287124185845000-664974-8da5ac040b9f",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_work_telemetry.py",
        result_digest="8f287dd05fe126949a207c721cfcb88dcef9c0e43cac9f7d4bbef0adda5ff808",
        outcome="rescued",
        recorded_status="revived_after_quarantine",
        source_field="revivals",
        attempt=1,
        validator_passed=True,
    ),
    _recipe(
        vector_id="aseh-hist-06",
        task_id="ASEH-013",
        task_cid="baguqeerak4rttvx52w4q2lfcqiua3lcv7rv4ctygpbp367ttclcxtl52b3ua",
        commit="f7d1284cbca2c7da8f13d1835ba8b7e0207156a7",
        tree="eaa93e4d91bf3819fa5b79fce7fcab5b375359ea",
        request_id="1788321943134207380-2155825-090cb8d2b2ca",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py",
        result_digest="22d94d3f62c2d081be64f398efdaeb15e6681c430e6235c6b3a41aae265b797a",
        outcome="human_escalated",
        recorded_status="manual_completion_authority",
        source_field="merge_result.reason",
        attempt=1,
        validator_passed=True,
    ),
    _recipe(
        vector_id="aseh-hist-07",
        task_id="ASEH-020",
        task_cid="baguqeerairtxfbwz7adu47img2dwyxy56pwuz5nzer5hpg33v5sx3mh3raja",
        commit="76c0eb523ca8da2049b2aa064d00dea361afdbae",
        tree="7e3ef048f33b19e179973d3c092f1a2d8235099e",
        request_id="1788080806639568195-1569915-00431fe8fa7a",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_ladder.py",
        result_digest="dda72953613b10a0508764e2f6578b422ab05237db5709a6022e461b835bd70c",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-08",
        task_id="ASEH-021",
        task_cid="baguqeeramfydtrea4u73aem3cwv2rzshicspexdhytuvbwbto2yk7pcpvzba",
        commit="3c42dc5f1267ee453452ad2fd7e7d3c6073c6021",
        tree="e9a740d0d0eaad8252099748c02091166c8f49a4",
        request_id="1788247515524993096-767433-93b22cfa7207",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_unresolved_question.py",
        result_digest="103fe25c9936f5b52b471aac9c8d1fff7d11bec7f8627c30cc23de1f6de9d9ea",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-09",
        task_id="ASEH-022",
        task_cid="baguqeeratcqeyqyqccvqgojk4qxq5o247m6ilvxjacvoucqgyx43plmiqkzq",
        commit="ffebdaaac3cc438316b8abcc9d5f06d2a3aacee4",
        tree="47d0e5145ab354140bb68010b9896617fed6e9c5",
        request_id="1788248266912492882-2034769-a1a61efe0a5a",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_stages.py",
        result_digest="463541ab1b191e5af1a0148c9dc81378384804b6265b7420f9800c8ff1483f52",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-10",
        task_id="ASEH-023",
        task_cid="baguqeerayess2igryj3uh3akc64v2hfnzu7u5nj7jgbyjbbzrcz7qjphuxaa",
        commit="b6a6fcded8b2c6cd517c8febd3bf7d450f148cf4",
        tree="7385e099cf7218d31897c47d07e4a0dac34e3a97",
        request_id="1788249079883153825-2455603-975e27d41b3f",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_model_escalation_policy.py",
        result_digest="f3345d67bc817a9c638a3a91d72ead37ad0012163802edb4556f6f33e1757f3d",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-11",
        task_id="ASEH-024",
        task_cid="baguqeeran7erlxobndrr5pkiyquzr2k6ze652ck7n6mmpqxystaaephssoda",
        commit="c19857a6bfec7bd94f3bebba611c7c79dac1a13d",
        tree="331519850893f5bcc7f6ac28d9424603b28849da",
        request_id="1788315831262916060-1314803-d579b11a9905",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_route_receipts.py",
        result_digest="5c3d26458044197bb14c939df3fab9fe445fc6255cb18d05838358cef00f2bae",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-12",
        task_id="ASEH-030",
        task_cid="baguqeera37mgpemf23xikp5wb2fuuycykn57gorpjrsgtwi777br47d6scvq",
        commit="e93ed6fbff86f1795f5adbfec69e76adec322c60",
        tree="648a830e603ab52433c787871aa2264de73d9e46",
        request_id="1788077672619512431-1447648-8df643ffa794",
        command="cd ipfs_datasets_py && python3 -m pytest -q tests/proof_context/test_context_pack_contract.py",
        result_digest="482ea80016aaccfdf0947878ab6f39678bd7784449624227efb38b041fdcd57a",
        outcome="retried",
        recorded_status="retrying",
        source_field="status",
        attempt=2,
        validator_passed=True,
    ),
    _recipe(
        vector_id="aseh-hist-13",
        task_id="ASEH-031",
        task_cid="baguqeeraztd6vb5ph7ipd3gh46dsmrvwbfcknlxwr3mesubj6j7jisiizgra",
        commit="b3298c5dcaabe2cedf7a5ed2948f3ab58c0e6a2b",
        tree="3fc0f458634a17204b7599371b22530b260d3464",
        request_id="1788278794198798538-733716-70f9627c5b69",
        command="cd ipfs_datasets_py && python3 -m pytest -q tests/proof_context/test_context_pack_builder.py",
        result_digest="d35eb338719028af55448870e96171a72fd5c491f33d122920ad90339b2e3821",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-14",
        task_id="ASEH-032",
        task_cid="baguqeeratok7rkfphf62mnnmuiutmc7v7dsv7nhx5txpzmvz2oy3s2f3a52q",
        commit="ee2d98bf9c02c4e5cce4f0b625328e20c75d273f",
        tree="520285586de2ecb6ba28e0c096f5571bbdf2303d",
        request_id="1788286933015140909-651890-98f007a32237",
        command="cd ipfs_kit_py && python3 -m pytest -q tests/test_context_pack_store.py",
        result_digest="da816d92a7e45196e06de1232bc51f98361ebcbb83355494fe68498363dde90c",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-15",
        task_id="ASEH-033",
        task_cid="baguqeerazrumjlclkl2morwckg4j7jrk2txy324iqxb6p3onhaezvwmvftza",
        commit="c9eeeddfeb41c412af6102cd0582e8db50bb4c3a",
        tree="c0c19a4cdedd12c8109e197e58410fba0fcd53fb",
        request_id="1788317215572169752-3800370-aad0e1002484",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_selector.py",
        result_digest="d60c9d8e38e078ead8798b81b607d81372c7509d86813be0accec18483928b92",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-16",
        task_id="ASEH-034",
        task_cid="baguqeerajvqrd72fucgyn2grwhn2v7y6hasqnmlotogazsdcxxghckjujpta",
        commit="de0911d61aa3ab6d4d8c4ed207f0336aa617d50b",
        tree="13c9c5967901b20066ce4586950cac1498fb2e0a",
        request_id="1788320133654991463-2155825-5cceda9d54a1",
        command="cd ipfs_datasets_py && python3 -m pytest -q tests/proof_context/test_incremental_context_pack.py",
        result_digest="e522e489f5443971b38553f87210695493833566a9209570e6c435304302b257",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-17",
        task_id="ASEH-040",
        task_cid="baguqeeradh6ldqaokukppeglh3vbrcol3j5ibrbmffiikni3eyqwhdcdrabq",
        commit="88ae6760c391049272686d4920f4efca9eb36611",
        tree="b3e3a393e1b883152db2158c4c6fa0078df779aa",
        request_id="1788249655995407604-3025281-f505d79926c8",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_state_transition_table.py",
        result_digest="8bbbc2495dde9e8d5ae2a366f01f8b008db51d58f68fb5e906812a5e1939b3fb",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-18",
        task_id="ASEH-041",
        task_cid="baguqeera6b7gn33a7xyh2knfhwax42gfqmp2vkffuq3fbgiy2u4xkn3q3i6a",
        commit="1b05b7845a76571f15642c32b8d267e3f533ca20",
        tree="264aa3aa584ecb88f86831381bb7167d3634c9d4",
        request_id="1788250132137452016-3412103-7592e492615e",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_transition_authority.py",
        result_digest="a343068c9a5a1c6189e02e2cb6e83cce395740c10dfe44dbd43190986c498ea4",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-19",
        task_id="ASEH-042",
        task_cid="baguqeerau5u5hnvlyt6z3xeln7q5tgjypdmrducmpjd7zq47sc3gcejcjara",
        commit="9edcb847d9d6a103020abbac5c15da0255562f33",
        tree="ebc09a0191b95b4acd054817101ebf113b3bc42f",
        request_id="1788250810781852594-3742393-78884b6539bf",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_lease_fence_idempotency.py",
        result_digest="84fa0189c168fbe06bf1f77e73c38277d0c78db4e4d507ec637bbbac36601564",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-20",
        task_id="ASEH-043",
        task_cid="baguqeerabmcdrfc3krrzp2l3qvlthdfp5ydvetd3ggoblu4nkyptvl32jqla",
        commit="644f303856efeea04c4e8262cc48118d70d6d7b3",
        tree="d447147245dd75f786a2587a62a8742a5678b7e5",
        request_id="1788251612626052064-71445-cf5526b5d29e",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_unknown_outcome_reconciliation.py",
        result_digest="9a017e241700a3ccb908b7cc8f77885579604c5210cbc4f9fe32373a400fac54",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-21",
        task_id="ASEH-044",
        task_cid="baguqeera5diyftpbjkx2xw4xtcd4j72v4qsm3w36r55wrxqhs4uxrlayfsna",
        commit="f675c72b78c980b286fd242d0be9acdeca8d38ad",
        tree="8d8199c607440107fc118d5520ccf514d0b79195",
        request_id="1788253036709101401-1184480-dcbe173d174a",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_owner_restart_recovery.py",
        result_digest="d3f75f593f026d1dcedded340e78d79f5c8c225edbab0b0dd1a64326da00ea61",
        outcome="failed",
        recorded_status="accepted_false",
        source_field="accepted",
        attempt=2,
        validator_passed=True,
    ),
    _recipe(
        vector_id="aseh-hist-22",
        task_id="ASEH-045",
        task_cid="baguqeeraljs5oxafm5vvjsov7t7m7dgysn4bhh6iadvqecv6zow4r4tew3fa",
        commit="88df9deb9b6d8d06a8f3e5a90866e2189a01a2da",
        tree="9fc9ff084259c1c2dc2c20f8d913ee6aa485679c",
        request_id="1788254764233096888-2768722-fbb029de9ab8",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_properties.py test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_crash_matrix.py",
        result_digest="98770bc00808b2ec182538f5f257b4ead5b493180a8c3a1def378b106a9f7b3b",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-23",
        task_id="ASEH-050",
        task_cid="baguqeeraax2fdlmut3bbkwcghwiuhjxpjpq5evceagv6rvr4jzytx5zoqcqa",
        commit="01d997213ac7f1f4ea1535ab196936c7d15c6840",
        tree="b2231626f4e8f835d5b4fb65d8c7e8ae6913aa6f",
        request_id="1788252255366973518-621507-d1a94cf99f50",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_change_impact.py",
        result_digest="1285ac3a4001223aa3ee3920e4398fc1086e6b70ae82e7a0739e8685d1faa221",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-24",
        task_id="ASEH-051",
        task_cid="baguqeerarrnqjwm6fqv7um4z2wfs7mowqqnqmxpgvnxvl57aq2hx66z52j5q",
        commit="b04344555bed41a2405acc065483ec93a919c0a8",
        tree="30d78733ff6df4db0e0e74c686aff28c6b7b30a2",
        request_id="1788252977530535621-1176203-88766d047755",
        command="cd ipfs_datasets_py && python3 -m pytest -q tests/unit/logic/software_contracts/semantic_state/test_task_contract.py",
        result_digest="826a8a2f5cf86479536d49befb0325006cbbcc8573ce1d3885993ed7b9a4b0b4",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-25",
        task_id="ASEH-052",
        task_cid="baguqeerakkeumzhdtkl7ejpik7gfljgnu3atudnq2laalugldsjl6zslbgfa",
        commit="5f5ed35e0bcd1ef8ff804bb0954fbdeb58a47fdb",
        tree="4994670b3a2d74c4aeb849bc5daa4a981ba28ea9",
        request_id="1788254171529593652-2112270-31eec82c14ec",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_affected_suffix_replanning.py",
        result_digest="2c15e50105cc8a05554615e359e22fcdc2398e5e3c34ac6cd2ddd07898699a6c",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-26",
        task_id="ASEH-053",
        task_cid="baguqeerannqqd4qwt77sy5toha3ctty3d4unraibz4z2vbmnvr2mruz2te5a",
        commit="94404a7b1d30200d465bd48b2b38afd21c40a8fc",
        tree="75032766aeab8e213f4bd35d9d7d2bd3af8916e8",
        request_id="1788253760463475337-1779321-7f54244b5771",
        command="cd ipfs_datasets_py && python3 -m pytest -q tests/unit/logic/software_contracts/semantic_state/test_refinement_context.py",
        result_digest="e311a458182275cd89b5a04fe6d102ed0c210dfeed247467187af9e708d7846b",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-27",
        task_id="ASEH-054",
        task_cid="baguqeeraltuo2zqbv3xjm7vlneaqhkp6dgqb75vtvcrk5u2ywywkucepudra",
        commit="a828819cda3657324215ad363fd14ebec6b794ce",
        tree="440377d5b5a996655be931293d719f4d2490e3f0",
        request_id="1788255122366523264-2968793-3b02013277c1",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_synthesis.py",
        result_digest="5185254dbe83050e0d43aa4332a726ab2651f9b5701ad6b4532cd253bc4d4545",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
    _recipe(
        vector_id="aseh-hist-28",
        task_id="ASEH-055",
        task_cid="baguqeera2g24etk77qhfklyx3gypxf5tujfpxih4tjt6eiq43ohzjvriw2qq",
        commit="1d84a4ffb1d179b2789bbb6536bb530567b30c29",
        tree="807d447160501c66b0c77de5db609c93382a2713",
        request_id="1788256671311708299-3486986-6696ef1d4fd4",
        command="python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_patch_plan_admission.py",
        result_digest="67df26f2c4eb59e3172d0486df8a55e96d3db486278a9b4cae86a4c2b19454d1",
        outcome="successful",
        recorded_status="completed",
        source_field="status",
    ),
)


def admit_corpus(
    recipes: Sequence[Mapping[str, Any]] | None = None,
    *,
    resolve_tree: bool = False,
) -> tuple[dict[str, Any], ...]:
    admitted = tuple(
        admit_vector(item, resolve_tree=resolve_tree)
        for item in (recipes or HISTORICAL_RECIPES)
    )
    if len(admitted) < HISTORICAL_MINIMUM:
        raise HistoricalCorpusError("historical corpus is below the 20-task minimum")
    vector_ids = [item["vector_id"] for item in admitted]
    identities = [item["identity"] for item in admitted]
    task_ids = [item["task_id"] for item in admitted]
    keys = [
        (item["task_id"], item["repository_tree"], item["outcome"])
        for item in admitted
    ]
    if len(set(vector_ids)) != len(vector_ids):
        raise HistoricalCorpusError("duplicated historical vector ids")
    if len(set(identities)) != len(identities):
        raise HistoricalCorpusError("duplicated historical identities")
    if len(set(task_ids)) != len(task_ids):
        raise HistoricalCorpusError("duplicated historical tasks")
    if len(set(keys)) != len(keys):
        raise HistoricalCorpusError("duplicated historical exact-tree vectors")
    covered = {item["outcome"] for item in admitted}
    if covered != set(HISTORICAL_OUTCOMES):
        raise HistoricalCorpusError("historical corpus is missing a required outcome class")
    return admitted


def build_manifest(vectors: Sequence[Mapping[str, Any]], *, vectors_text: str) -> dict[str, Any]:
    listed = []
    for item in vectors:
        listed.append(
            {
                "identity": item["identity"],
                "outcome": item["outcome"],
                "repository_tree": item["repository_tree"],
                "task_cid": item["task_cid"],
                "task_id": item["task_id"],
                "vector_id": item["vector_id"],
            }
        )
    payload = {
        "acceptance_contracts_unchanged": True,
        "authority": False,
        "count": len(listed),
        "hermetic_sufficient_for_production_promotion": False,
        "live": False,
        "minimum": HISTORICAL_MINIMUM,
        "objective_id": OBJECTIVE_ID,
        "objective_revision": PLAN_ROOT_CID,
        "population_kind": POPULATION_KIND,
        "program_id": PROGRAM_ID,
        "provenance_fields": list(REQUIRED_PROVENANCE_FIELDS),
        "required_outcomes": list(HISTORICAL_OUTCOMES),
        "schema": HISTORICAL_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "sealed",
        "task_id": TASK_ID,
        "vectors": listed,
        "vectors_identity": content_identity({"vectors": [dict(item) for item in vectors]}),
        "vectors_path": (
            "benchmarks/agent_supervisor/efficiency_state_hardening/historical_vectors.jsonl"
        ),
        "vectors_sha256": _sha256_bytes(vectors_text.encode("utf-8")),
    }
    covered = sorted({item["outcome"] for item in listed})
    payload["covered_outcomes"] = covered
    body = {key: value for key, value in payload.items() if key != "identity"}
    payload["identity"] = content_identity(body)
    return payload


def render_vectors(vectors: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(compact_json(item) for item in vectors) + "\n"


def load_vectors(path: Path = VECTORS_PATH) -> tuple[dict[str, Any], ...]:
    raw = path.read_text(encoding="utf-8")
    loaded: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HistoricalCorpusError(f"{path.name}:{line_number} is not JSON") from exc
        loaded.append(admit_vector(payload, resolve_tree=False))
    return admit_corpus(loaded, resolve_tree=False)


def write_sealed_artifacts() -> dict[str, Path]:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    vectors = admit_corpus(resolve_tree=False)
    vectors_text = render_vectors(vectors)
    manifest = build_manifest(vectors, vectors_text=vectors_text)
    VECTORS_PATH.write_text(vectors_text, encoding="utf-8")
    MANIFEST_PATH.write_text(pretty_json(manifest), encoding="utf-8")
    return {"manifest": MANIFEST_PATH, "vectors": VECTORS_PATH}


def verify_sealed_artifacts() -> dict[str, Any]:
    generated = admit_corpus()
    loaded = load_vectors()
    if tuple(item["vector_id"] for item in loaded) != tuple(
        item["vector_id"] for item in generated
    ):
        raise HistoricalCorpusError("sealed vectors do not match the historical recipes")
    if [item["identity"] for item in loaded] != [item["identity"] for item in generated]:
        raise HistoricalCorpusError("sealed identities do not match the historical recipes")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise HistoricalCorpusError("historical manifest must be an object")
    expected = build_manifest(generated, vectors_text=render_vectors(generated))
    if manifest != expected:
        raise HistoricalCorpusError("historical manifest does not match the sealed corpus")
    return {
        "count": len(loaded),
        "unique_identities": len({item["identity"] for item in loaded}),
        "unique_tasks": len({item["task_id"] for item in loaded}),
        "covered_outcomes": sorted({item["outcome"] for item in loaded}),
    }


write_sealed_artifacts()


def _imported_module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_historical_corpus_installs_without_sibling_test_imports() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    parsed = ast.parse(source)
    imported = _imported_module_names(parsed)
    sibling_stems = {
        path.stem
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name != Path(__file__).name
    }
    for name in imported:
        assert not name.startswith("test."), name
        assert name.rsplit(".", 1)[-1] not in sibling_stems
        for prefix in SIBLING_TEST_PREFIXES:
            assert not name.startswith(prefix), name
    assert "ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts" in imported


def test_sealed_corpus_covers_twenty_unique_exact_tree_vectors() -> None:
    verified = verify_sealed_artifacts()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vectors = load_vectors()
    assert verified["count"] >= HISTORICAL_MINIMUM
    assert verified["unique_identities"] == verified["count"]
    assert verified["unique_tasks"] == verified["count"]
    assert verified["covered_outcomes"] == sorted(HISTORICAL_OUTCOMES)
    assert manifest["schema"] == HISTORICAL_MANIFEST_SCHEMA
    assert manifest["count"] == len(vectors) == len(manifest["vectors"])
    assert manifest["minimum"] == HISTORICAL_MINIMUM
    assert manifest["status"] == "sealed"
    assert manifest["live"] is False
    assert manifest["authority"] is False
    assert manifest["acceptance_contracts_unchanged"] is True
    assert manifest["hermetic_sufficient_for_production_promotion"] is False
    assert manifest["population_kind"] == POPULATION_KIND
    assert tuple(manifest["required_outcomes"]) == HISTORICAL_OUTCOMES
    assert set(manifest["covered_outcomes"]) == set(HISTORICAL_OUTCOMES)
    assert tuple(manifest["provenance_fields"]) == REQUIRED_PROVENANCE_FIELDS
    listed_ids = [item["vector_id"] for item in manifest["vectors"]]
    assert listed_ids == [item["vector_id"] for item in vectors]


def test_vectors_are_provenance_complete_and_exact_tree_bound() -> None:
    vectors = admit_corpus()
    for item in vectors:
        assert GIT_OID_RE.fullmatch(item["repository_commit"])
        assert GIT_OID_RE.fullmatch(item["repository_tree"])
        assert tree_is_present(item["repository_tree"]), item["repository_tree"]
        assert git_object_kind(item["repository_commit"]) in {"commit", "tree"}
        assert CID_RE.fullmatch(item["task_cid"])
        assert CID_RE.fullmatch(item["identity"])
        assert item["receipt_cid"] == item["identity"]
        assert item["acceptance_unchanged"] is True
        assert item["acceptance_command"] == item["validator_command"]
        assert item["acceptance_digest"] == _sha256_text(item["acceptance_command"])
        assert item["live"] is False
        assert item["truth_state"] in {"observed", "verified"}
        for field in REQUIRED_PROVENANCE_FIELDS:
            assert item[field]


def test_acceptance_contracts_are_the_original_historical_commands() -> None:
    by_id = {item["task_id"]: item for item in admit_corpus()}
    assert (
        by_id["ASEH-000"]["validator_command"]
        == "python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_authority_inventory.py"
    )
    assert (
        by_id["ASEH-013"]["validator_command"]
        == "python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py"
    )
    mutated = dict(HISTORICAL_RECIPES[0])
    mutated["validator_command"] = (
        "python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_historical_corpus.py"
    )
    with pytest.raises(HistoricalCorpusError, match="acceptance command was altered"):
        admit_vector(mutated, resolve_tree=False)


def test_negative_cases_reject_inferred_duplicate_and_missing_tree_vectors() -> None:
    inferred = dict(HISTORICAL_RECIPES[0])
    inferred["outcome"] = "failed"
    inferred["outcome_recorded_status"] = "completed"
    with pytest.raises(HistoricalCorpusError, match="inferred or unrecorded"):
        admit_vector(inferred, resolve_tree=False)

    duplicate = list(HISTORICAL_RECIPES) + [dict(HISTORICAL_RECIPES[0])]
    with pytest.raises(HistoricalCorpusError, match="duplicated historical"):
        admit_corpus(duplicate, resolve_tree=False)

    missing_tree = dict(HISTORICAL_RECIPES[0])
    missing_tree["repository_tree"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    missing_tree["repository_commit"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(HistoricalCorpusError, match="source tree is not present"):
        admit_vector(missing_tree, resolve_tree=True)

    live = dict(HISTORICAL_RECIPES[0])
    live["live"] = True
    with pytest.raises(HistoricalCorpusError, match="cannot be live"):
        admit_vector(live, resolve_tree=False)

    malformed = dict(HISTORICAL_RECIPES[0])
    malformed.pop("task_cid")
    with pytest.raises(HistoricalCorpusError, match="task_cid"):
        admit_vector(malformed, resolve_tree=False)
