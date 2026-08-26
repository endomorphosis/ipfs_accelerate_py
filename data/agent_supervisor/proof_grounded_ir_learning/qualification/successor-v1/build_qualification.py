#!/usr/bin/env python3
"""Build the immutable PGIR-207 terminal successor qualification.

The builder consumes the exact completed PGIR-206 forest and its sealed typed
``not_run`` bundle.  It resolves the closed 16 acceptance criteria and 32
report sections, records a deterministic ``no_go``, and emits a following
board that cannot be scheduled without manual external evidence.

No-site standard-library Python is sufficient.  The builder is write-once and
does not train, evaluate, invoke a prover, mutate a pointer, access the network,
open hidden labels, publish, stage, or commit.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[4]
DATA_ROOT = PACKAGE_DIR.parents[1]
FREEZE_DIR = DATA_ROOT / "freeze" / "successor-v1"
EXPERIMENT_DIR = DATA_ROOT / "experiments" / "successor-v1"
DOCS_DIR = (
    REPOSITORY_ROOT
    / "docs"
    / "architecture"
    / "proof_grounded_ir_learning"
    / "successor-v1"
)
NESTED_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"

TASK_ID = "PGIR-207"
TASK_TITLE = "Re-qualify, publish or reject, and issue the following board"
TASK_KEY = "task/v1/efe4a7c323386342928737eaf8f18cad23f597f16cc27cd4acf32939066a3279"
TASK_CID = "baguqeera57skpqzdhbrufeuhg7vpr4mmvur7lf7rntbhzvfm6mutsbtkgj4q"
OBJECTIVE_ID = "PGIR-G110"
PARENT_GOAL = "PGIR-G110"
SUBGOAL = "final-decision-report-v2"
RESULT_IDENTITY = "RESULT(PGIR-207)"

# These forest values are patched exactly once after the current supervisor has
# merged and completed PGIR-206.  A placeholder is a hard build failure.
PGIR_206_IMPLEMENTATION = "0373c2968b9b46b43ee9f564051182df137c08e4"
PGIR_206_IMPLEMENTATION_PARENT = "c03180fbbd64b225a121fcc7be046bd0beab8918"
PGIR_206_MERGE_FIRST_PARENT = "c03180fbbd64b225a121fcc7be046bd0beab8918"
PGIR_206_MERGE = "efd5bc4107a595ba7b56632faa7d2f6c82f249d8"
PGIR_206_COMPLETION = "ef4a74b362386578858d368a8e7420238e4c897d"
PGIR_206_COMPLETION_TREE = "556e95765e90f00025eecefb8048a655956d70a3"
PGIR_206_COMPLETION_PARENT = "efd5bc4107a595ba7b56632faa7d2f6c82f249d8"

PGIR_206_RESULT_CID = "baguqeera7pwtswk2472r2yi2ijs5veamgx4hmy2nice2bnd3fvf72o3kgswq"
PGIR_206_MANIFEST_CID = "baguqeerao2izb4i6bv7ffg3flharhjlph35gpwl4rbekzrrwm3n25mibqxsq"
PGIR_205_RESULT_CID = "baguqeerarcuvejfqyjsfbqdtel67wu2lhjec3oh27enlqbxsvgyfbytsgd2q"
CAMPAIGN_INPUT_ROOT_CID = "baguqeerajvu2dvjjxe4l6dibujguiedwhdahseziqpz7xhhp724jlxhlxz4q"
PGIR_211_ACCEPTANCE_CID = "baguqeeram562re6snweb5nuinwprb4ehccvkin7kpylihktizlirsss7pllq"
NESTED_CURRENT = "2a06dfe8546cdde78ff6d101a94708be0e6bf6e6"
NESTED_CURRENT_TREE = "7169c2a67929044a02350bc26d0a51c853a4981b"

PGIR_206_RECEIPTS = {
    "admission": "baguqeeraqyqjwwkt3jmr2b2hnwpwafhwndre4fkd7gcck5rvwzcyfh22shia",
    "checkpoint": "baguqeeraa2bj4uvhwqlcqyo6iastvodab6ulvnycmu2emvyohhwdrm3dcmka",
    "evaluation": "baguqeerashtuscaphw2hpicra6klxt6a2dae64gqbuanvpmdzqzj5ljotrda",
    "proof": "baguqeeraa55wxtbh2s63j22qgytbnsfjpatnstg4vun4nqg5qlvu3fgcx36a",
    "reducer_cas": "baguqeera5x32cyuywisqviyfi7nbzfwnjdmtf5k4zzbbdkdywzikzswswrna",
    "resource": "baguqeerani6ro4d7naxn4mrb5rgzuiw77zq3hckyttbhgihqvtcj6oquziiq",
    "training": "baguqeeral5izctrs4pkjsiwmodfufnvgcsvmvz3fucarh5atlduzta7aaiaa",
}

HISTORICAL_TASK_CIDS = {
    "PGIR-072": "baguqeeraarlt745ftpwax4tdovajsxrgp5r72fkudlbs5kmg3q2cbqow6mpq",
    "PGIR-090": "baguqeeragt5xhqov2e5vna6pp2zmghhrqhq2v4qzl36a3aucytrcom2erkta",
    "PGIR-100": "baguqeerainaqx5w72m2epc7wsolk3elqihpcganr6d3lrsuzseldzkqngija",
    "PGIR-111": "baguqeerad2idzhwzfqlyxjh7u34bczkyutsvq7oxcmuaglupr436gqfo6kga",
}
HISTORICAL_EVIDENCE_CIDS = {
    "PGIR-072": "baguqeerazlsqaghk6m2c2ru3qbrsfprv7b5nbc5oa5lvfxz6d4b6qqgxpvra",
    "PGIR-090": "bafkreigwdei25h3eg2k2l6gp6ak5tbkbcfabi6vsoqzrzv6k6mpm73lhge",
    "PGIR-100": HISTORICAL_TASK_CIDS["PGIR-100"],
    "PGIR-111": "baguqeeraejs56hwzs3bqtgzoayrc2fxwgfnhcsxjthi4dh7gh64wptlkfhwa",
}

REASON_CODES = (
    "no_rights_admitted_training_rows",
    "corpus_not_materialized",
    "required_holdouts_insufficient",
    "tokenizer_not_admitted",
    "historical_semantic_baseline_not_currently_qualified",
    "integrated_evidence_does_not_authorize_execution",
    "portability_no_go",
    "pgir_206_execution_not_authorized",
    "pgir_206_typed_not_run",
    "no_candidate_checkpoint",
    "publication_not_authorized",
)

INSUFFICIENT_HOLDOUTS = (
    "compiler",
    "cross_reference",
    "domain",
    "exception",
    "length",
    "lineage",
    "notation",
    "premise",
    "proof_library",
    "publication",
    "rare_operator",
    "time",
    "type",
)

M2_GATES = (
    "lineage",
    "syntax",
    "type",
    "semantic",
    "proof",
    "calibration",
    "family",
    "jurisdiction",
    "source_span",
    "latency",
    "resource",
)

QUALIFIED_CLAIM = (
    "The qualified compiler/decompiler checkpoint was trained from "
    "content-addressed, lineage-safe JusticeDAO source and proof artifacts under "
    "the declared split, compiler, tokenizer, loss, curriculum, and supervisor "
    "configuration. It achieved the reported held-out token, latent-retrieval, "
    "structural, semantic, and proof metrics without exceeding the admitted "
    "regression thresholds. Learned outputs remain candidates until validated by "
    "the canonical parser, type system, translation contracts, and required proof "
    "or counterexample authorities."
)

ACCEPTANCE_CRITERIA = (
    ("F01", "current_input_child_evidence", "All PGIR-G010 through PGIR-G110 children have fresh current-input evidence."),
    ("F02", "no_source_lineage_leakage", "No source-lineage leakage across related derivatives."),
    ("F03", "one_canonical_bridge", "One canonical typed bridge is bound and used."),
    ("F04", "deterministic_baseline_measured", "A current-input deterministic baseline is measured and qualified."),
    ("F05", "proof_aware_contracts", "Proof-aware pair, loss, and evaluation contracts are sealed."),
    ("F06", "resumable_resource_aware_campaign", "A resumable resource-aware campaign exists and may execute only from sealed inputs."),
    ("F07", "deterministic_promotion_or_nogo", "Promotion is deterministic policy admission or a documented no-go."),
    ("F08", "authorized_append_only_publication", "Append-only qualified publication occurs only when independently authorized."),
    ("F09", "token_metrics_with_uncertainty", "Actual token metrics are reported with paired uncertainty."),
    ("F10", "latent_retrieval_metrics_with_uncertainty", "Actual latent and retrieval metrics are reported with paired uncertainty."),
    ("F11", "structural_metrics_with_uncertainty", "Actual structural metrics are reported with paired uncertainty."),
    ("F12", "semantic_metrics_with_uncertainty", "Actual semantic metrics are reported with paired uncertainty."),
    ("F13", "proof_metrics_with_uncertainty", "Actual proof metrics are reported with paired uncertainty."),
    ("F14", "calibration_ood_metrics_with_uncertainty", "Actual calibration and OOD metrics are reported with paired uncertainty."),
    ("F15", "latency_resource_metrics_with_uncertainty", "Actual latency and resource results are reported with paired uncertainty."),
    ("F16", "no_hidden_test_tune_admitted_publish_next_board", "Hidden tests are never used for tuning; only an admitted candidate may be published; the next content-addressed board is produced."),
)

REPORT_SECTIONS = (
    (1, "exact_source_revisions", "Exact source revisions"),
    (2, "justicedao_revisions", "Exact JusticeDAO repository revisions and configurations"),
    (3, "current_state_inventory", "Current-state inventory"),
    (4, "source_versus_derived", "Source versus derived record counts"),
    (5, "lineage_and_split", "Lineage and split design"),
    (6, "leakage_audit", "Leakage-audit results"),
    (7, "canonical_bridge", "Canonical bridge-IR design"),
    (8, "compiler_architecture", "Compiler architecture"),
    (9, "decompiler_architecture", "Decompiler architecture"),
    (10, "deterministic_baseline", "Deterministic baseline"),
    (11, "learned_model_architecture", "Learned-model architecture"),
    (12, "tokenizer_and_vocabulary", "Tokenizer and vocabulary"),
    (13, "loss_configuration", "Loss configuration"),
    (14, "training_curriculum", "Training curriculum"),
    (15, "hard_negative_generation", "Hard-negative generation"),
    (16, "lean_capable_results", "Lean-capable model results"),
    (17, "tactician_results", "Tactician results"),
    (18, "hammer_results", "Hammer results"),
    (19, "kernel_verification", "Kernel-verification results"),
    (20, "cross_entropy_metrics", "Cross-entropy metrics"),
    (21, "cosine_contrastive_metrics", "Cosine and contrastive metrics"),
    (22, "retrieval_metrics", "Retrieval metrics"),
    (23, "structural_metrics", "Structural metrics"),
    (24, "semantic_metrics", "Semantic metrics"),
    (25, "proof_metrics", "Proof metrics"),
    (26, "calibration_ood_metrics", "Calibration and OOD metrics"),
    (27, "resource_utilization", "Resource utilization"),
    (28, "multi_supervisor_scheduling", "Multi-supervisor scheduling results"),
    (29, "promotion_or_rejection", "Checkpoint promotion or rejection decision"),
    (30, "published_artifacts", "Published artifacts"),
    (31, "known_limitations", "Known limitations"),
    (32, "next_board_recommendation", "Exact recommendation for the next training and data-improvement board"),
)

SECTION_STATUS = {
    1: "resolved",
    2: "resolved",
    3: "resolved",
    4: "resolved",
    5: "resolved_with_no_go",
    6: "satisfied",
    7: "satisfied_with_limitations",
    8: "resolved",
    9: "resolved",
    10: "no_go",
    11: "not_run",
    12: "no_go",
    13: "resolved_unused",
    14: "not_run",
    15: "resolved_unused",
    16: "not_run",
    17: "not_run",
    18: "not_run",
    19: "not_run",
    20: "not_run",
    21: "not_run",
    22: "not_run",
    23: "not_run",
    24: "not_run",
    25: "not_run",
    26: "not_run",
    27: "not_run",
    28: "resolved",
    29: "no_go",
    30: "denied",
    31: "resolved",
    32: "resolved",
}

SECTION_SUMMARIES = {
    1: "The completed PGIR-206 recursive forest and unchanged nested source revision are pinned exactly.",
    2: "JDAO-PINSET-1 remains immutable; no JusticeDAO repository is admitted for learned training.",
    3: "Freeze, experiment, receipt, result, and dependency inventories are content addressed and complete.",
    4: "The 7,173 source rows remain quarantined; zero materialized or training-admitted rows were read.",
    5: "Lineage-safe split policy remains sealed while thirteen required holdouts are permanently insufficient.",
    6: "The successor leakage audit passed and hidden labels stayed unopened.",
    7: "The one canonical compiler/decompiler bridge remains bound with explicit limitations.",
    8: "The deterministic typed compiler remains the only admitted compiler architecture.",
    9: "The source-withheld deterministic decompiler remains the only admitted decompiler architecture.",
    10: "The historical deterministic baseline was retired and is not currently qualified for this campaign.",
    11: "R2-R6 declarations exist, but no learned architecture was instantiated and no weights were created.",
    12: "The tokenizer policy is permanently deterministic-only and authorizes no learned vocabulary.",
    13: "The closed loss configuration remained declared but was consumed by zero optimizer steps.",
    14: "All six arms and sixteen experiment keys were sealed; admission closed before any run.",
    15: "Hard-negative contracts remain sealed and unused; unknown or timeout outcomes cannot become labels.",
    16: "No Lean-capable proposal was invoked or admitted to a curriculum.",
    17: "No tactician proposal was invoked or admitted to a curriculum.",
    18: "No Hammer, ATP, or SMT proposal was invoked or admitted to a curriculum.",
    19: "Independent proof authority remains required; PGIR-206 invoked no prover or kernel checker.",
    20: "Cross-entropy has zero measured cells and unavailable paired uncertainty.",
    21: "Cosine and contrastive metrics have zero measured cells and unavailable paired uncertainty.",
    22: "Retrieval metrics have zero measured cells and unavailable paired uncertainty.",
    23: "Structural metrics have zero measured cells and unavailable paired uncertainty.",
    24: "Semantic metrics have zero measured cells; historical fixture scores are not substituted.",
    25: "Proof metrics have zero measured cells and no checked campaign proofs.",
    26: "Calibration and OOD metrics have zero measured cells and unavailable uncertainty.",
    27: "No resource lease was requested or acquired; measured cost is unavailable, not zero.",
    28: "The current supervisor completed the typed not-run lifecycle without granting a training lease.",
    29: "No checkpoint exists, every non-compensable gate remains binding, and the pointer is unchanged.",
    30: "No upload was attempted and no remote revision exists because qualification and authority are absent.",
    31: "The no-go reason set is complete and no missing evidence is suppressed or counted as success.",
    32: "The following board contains only blocked PGIR-212 and requires manual external evidence.",
}

GRAPH_DEPENDENCIES = {
    "PGIR-072": (),
    "PGIR-090": (),
    "PGIR-100": (),
    "PGIR-111": (),
    "PGIR-200": ("PGIR-111",),
    "PGIR-201": ("PGIR-200",),
    "PGIR-202": ("PGIR-201",),
    "PGIR-203": ("PGIR-202",),
    "PGIR-204": ("PGIR-202",),
    "PGIR-208": ("PGIR-200", "PGIR-201", "PGIR-202"),
    "PGIR-209": ("PGIR-202", "PGIR-204"),
    "PGIR-210": ("PGIR-204", "PGIR-208"),
    "PGIR-211": ("PGIR-209", "PGIR-210"),
    "PGIR-205": ("PGIR-200", "PGIR-201", "PGIR-202", "PGIR-203", "PGIR-204", "PGIR-208", "PGIR-209", "PGIR-210", "PGIR-211"),
    "PGIR-206": ("PGIR-205",),
    "PGIR-207": ("PGIR-072", "PGIR-090", "PGIR-100", "PGIR-206"),
}
DEPENDENCY_FIRST_ORDER = (
    "PGIR-072",
    "PGIR-090",
    "PGIR-100",
    "PGIR-111",
    "PGIR-200",
    "PGIR-201",
    "PGIR-202",
    "PGIR-203",
    "PGIR-204",
    "PGIR-208",
    "PGIR-209",
    "PGIR-210",
    "PGIR-211",
    "PGIR-205",
    "PGIR-206",
    "PGIR-207",
)

SEALED_PATHS = (
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/README.md",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/acceptance.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/build_qualification.py",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/decision.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/evaluation_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/proof_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/promotion_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/publication_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/recipe.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/report_sections.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/result_graph.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verify_qualification.py",
    "docs/architecture/proof_grounded_ir_learning/successor-v1/final_report.md",
    "docs/architecture/proof_grounded_ir_learning/successor-v1/next.todo.md",
)
FINAL_PATHS = (
    *SEALED_PATHS,
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/manifest.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verification_receipt.json",
    "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/result.json",
)


class QualificationBuildError(RuntimeError):
    """Raised when terminal qualification cannot be sealed exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationBuildError(message)


def task_binding() -> dict[str, Any]:
    return {
        "current_task_cid": TASK_CID,
        "current_task_key": TASK_KEY,
        "objective_id": OBJECTIVE_ID,
        "parent_goal": PARENT_GOAL,
        "subgoal": SUBGOAL,
        "task_id": TASK_ID,
        "title": TASK_TITLE,
    }


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise QualificationBuildError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise QualificationBuildError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rendered_bytes(value: Any) -> bytes:
    validate_value(value)
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def dag_json_cid(value: Any) -> str:
    digest = hashlib.sha256(canonical_bytes(value)).digest()
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + digest).decode("ascii").rstrip("=").lower()


def raw_cid(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode("ascii").rstrip("=").lower()


def add_identity(payload: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    result = dict(payload)
    projection = dict(result)
    result[f"{name}_sha256"] = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    result[f"{name}_cid"] = dag_json_cid(projection)
    return result


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON input: {path}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(QualificationBuildError(f"float {raw!r} in {path}")),
            parse_constant=lambda raw: (_ for _ in ()).throw(QualificationBuildError(f"constant {raw!r} in {path}")),
        )
    require(isinstance(value, dict), f"{path} must be a JSON object")
    validate_value(value)
    return value


def git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise QualificationBuildError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.decode("utf-8").strip()


def commit_record(commit: str) -> dict[str, Any]:
    return {
        "commit": commit,
        "parents": git("show", "-s", "--format=%P", commit).split(),
        "subject": git("show", "-s", "--format=%s", commit),
        "tree": git("show", "-s", "--format=%T", commit),
    }


def artifact_record(data: bytes) -> dict[str, Any]:
    return {
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def write_once(path: Path, data: bytes, *, check: bool) -> None:
    if path.exists():
        require(path.is_file() and not path.is_symlink(), f"output is not a regular file: {path}")
        require(path.read_bytes() == data, f"refusing different bytes at {path.relative_to(REPOSITORY_ROOT)}")
        return
    require(not check, f"missing generated artifact: {path.relative_to(REPOSITORY_ROOT)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            require(written > 0, f"short write for {path}")
            offset += written
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.close(fd)


def validate_forest(*, check: bool) -> list[dict[str, Any]]:
    pending = (
        PGIR_206_MERGE,
        PGIR_206_MERGE_FIRST_PARENT,
        PGIR_206_COMPLETION,
        PGIR_206_COMPLETION_TREE,
        PGIR_206_COMPLETION_PARENT,
    )
    require(not any("PENDING" in value for value in pending), "PGIR-206 completion forest is not pinned")
    head = git("rev-parse", "HEAD")
    if check:
        require(
            git("merge-base", PGIR_206_COMPLETION, head) == PGIR_206_COMPLETION,
            "HEAD does not descend from exact PGIR-206 completion",
        )
    else:
        require(head == PGIR_206_COMPLETION, "generation HEAD is not exact PGIR-206 completion")
    implementation = commit_record(PGIR_206_IMPLEMENTATION)
    merge = commit_record(PGIR_206_MERGE)
    completion = commit_record(PGIR_206_COMPLETION)
    require(
        implementation["parents"] == [PGIR_206_IMPLEMENTATION_PARENT],
        "PGIR-206 implementation parent drifted",
    )
    require(
        merge["parents"] == [PGIR_206_MERGE_FIRST_PARENT, PGIR_206_IMPLEMENTATION],
        "PGIR-206 merge parents drifted",
    )
    require(completion["parents"] == [PGIR_206_COMPLETION_PARENT], "PGIR-206 completion parent drifted")
    require(PGIR_206_COMPLETION_PARENT == PGIR_206_MERGE, "PGIR-206 completion does not directly follow merge")
    require(completion["tree"] == PGIR_206_COMPLETION_TREE, "PGIR-206 completion tree drifted")
    require(
        implementation["subject"] == "PGIR-206: Re-run R1-R6 on the superseding freeze",
        "PGIR-206 implementation subject drifted",
    )
    require(
        merge["subject"]
        == f"Merge commit '{PGIR_206_IMPLEMENTATION}' into agent/pgir-successor-current-supervisor-20260825",
        "PGIR-206 merge subject drifted",
    )
    require(completion["subject"] == "PGIR-206: mark todo completed", "PGIR-206 completion subject drifted")
    for ancestor, descendant in (
        (implementation["parents"][0], PGIR_206_IMPLEMENTATION),
        (PGIR_206_IMPLEMENTATION_PARENT, PGIR_206_MERGE_FIRST_PARENT),
        (PGIR_206_MERGE_FIRST_PARENT, PGIR_206_MERGE),
        (PGIR_206_IMPLEMENTATION, PGIR_206_MERGE),
        (PGIR_206_MERGE, PGIR_206_COMPLETION),
    ):
        require(git("merge-base", ancestor, descendant) == ancestor, f"ancestry drifted: {ancestor} -> {descendant}")
    return [
        {**implementation, "role": "implementation"},
        {**merge, "role": "merge"},
        {**completion, "role": "completion"},
    ]


def validate_recursive_repository() -> dict[str, Any]:
    require(NESTED_ROOT.is_dir() and not NESTED_ROOT.is_symlink(), "nested repository is absent")
    nested_head = git("rev-parse", "HEAD", cwd=NESTED_ROOT)
    nested_tree = git("rev-parse", "HEAD^{tree}", cwd=NESTED_ROOT)
    require(nested_head == NESTED_CURRENT, "nested revision drifted")
    require(nested_tree == NESTED_CURRENT_TREE, "nested tree drifted")
    require(git("status", "--porcelain=v1", "--untracked-files=all", cwd=NESTED_ROOT) == "", "nested worktree is dirty")
    require(git("rev-parse", f"{PGIR_206_COMPLETION}:ipfs_datasets_py") == NESTED_CURRENT, "completion gitlink drifted")
    return {
        "nested_commit": nested_head,
        "nested_tree": nested_tree,
        "outer_commit": PGIR_206_COMPLETION,
        "outer_tree": PGIR_206_COMPLETION_TREE,
    }


def load_inputs(*, check: bool) -> dict[str, Any]:
    forest = validate_forest(check=check)
    recursive = validate_recursive_repository()
    freeze = {
        "result": strict_json(FREEZE_DIR / "result.json"),
        "manifest": strict_json(FREEZE_DIR / "manifest.json"),
        "root": strict_json(FREEZE_DIR / "campaign_input_root.json"),
        "admission": strict_json(FREEZE_DIR / "plan_admission_receipt.json"),
        "verification": strict_json(FREEZE_DIR / "verification_receipt.json"),
        "integrated": strict_json(FREEZE_DIR / "integrated-acceptance" / "integrated_acceptance.json"),
    }
    experiment = {
        "result": strict_json(EXPERIMENT_DIR / "result.json"),
        "manifest": strict_json(EXPERIMENT_DIR / "manifest.json"),
        "campaign": strict_json(EXPERIMENT_DIR / "campaign.json"),
        "comparison": strict_json(EXPERIMENT_DIR / "comparison.json"),
        "heldouts": strict_json(EXPERIMENT_DIR / "heldouts.json"),
        "admission": strict_json(EXPERIMENT_DIR / "receipts" / "admission.json"),
        "checkpoint": strict_json(EXPERIMENT_DIR / "receipts" / "checkpoint.json"),
        "evaluation": strict_json(EXPERIMENT_DIR / "receipts" / "evaluation.json"),
        "proof": strict_json(EXPERIMENT_DIR / "receipts" / "proof.json"),
        "reducer_cas": strict_json(EXPERIMENT_DIR / "receipts" / "reducer_cas.json"),
        "resource": strict_json(EXPERIMENT_DIR / "receipts" / "resource.json"),
        "training": strict_json(EXPERIMENT_DIR / "receipts" / "training.json"),
    }
    require(freeze["result"]["result_cid"] == PGIR_205_RESULT_CID, "PGIR-205 result CID drifted")
    require(freeze["root"]["root_cid"] == CAMPAIGN_INPUT_ROOT_CID, "campaign root CID drifted")
    require(freeze["root"]["qualification"]["decision"] == "no_go", "freeze decision drifted")
    require(
        freeze["root"]["qualification"]["descendant_execution_authorized"] is False,
        "freeze unexpectedly authorizes execution",
    )
    require(freeze["integrated"]["acceptance_cid"] == PGIR_211_ACCEPTANCE_CID, "PGIR-211 acceptance drifted")
    require(freeze["integrated"]["pgir_205_execution_authorized"] is False, "integrated evidence unexpectedly authorizes execution")
    result = experiment["result"]
    require(result["result_cid"] == PGIR_206_RESULT_CID, "PGIR-206 result CID drifted")
    require(result["manifest_cid"] == PGIR_206_MANIFEST_CID, "PGIR-206 manifest CID drifted")
    require(result["result_identity"] == "RESULT(PGIR-206)", "PGIR-206 result identity drifted")
    require(result["decision"] == "no_go", "PGIR-206 decision drifted")
    require(result["execution_status"] == "not_run", "PGIR-206 execution status drifted")
    require(result["disposition"] == "typed_not_run", "PGIR-206 disposition drifted")
    require(result["execution_authorized"] is False, "PGIR-206 unexpectedly authorizes execution")
    require(result["checkpoint_count"] == 0, "PGIR-206 unexpectedly created checkpoints")
    require(result["measured_cell_count"] == 0 and result["metric_cell_count"] == 192, "PGIR-206 metric population drifted")
    require(result["task_binding"]["current_task_cid"] == "baguqeerafze3wxxiomo4rhuxaguuk35d4tj4lrp2jtuj6jcinxnokczuctca", "PGIR-206 task CID drifted")
    require(result["task_binding"]["current_task_key"] == "task/v1/2e49bb5ee8731dc89e9701a9456fa3e4d3c5c5fa4ce89f24486ddae50b3414c4", "PGIR-206 task key drifted")
    require(result["receipt_cids"] == PGIR_206_RECEIPTS, "PGIR-206 receipt set drifted")
    require(experiment["manifest"]["manifest_cid"] == PGIR_206_MANIFEST_CID, "experiment manifest identity drifted")
    require(experiment["manifest"]["artifact_count"] == 16, "experiment manifest artifact count drifted")
    require(experiment["campaign"]["execution_status"] == "not_run", "campaign unexpectedly ran")
    require(experiment["campaign"]["lease_eligible"] is False, "campaign unexpectedly lease eligible")
    require(experiment["comparison"]["no_winner"] is True, "experiment unexpectedly chose a winner")
    require(experiment["comparison"]["pair_count"] == 15, "comparison pair count drifted")
    require(experiment["heldouts"]["hidden_labels_opened"] is False, "hidden labels were opened")
    require(experiment["heldouts"]["failed_holdout_count"] == 13, "failed holdout count drifted")
    require(experiment["evaluation"]["receipt_cid"] == PGIR_206_RECEIPTS["evaluation"], "evaluation receipt drifted")
    require(experiment["evaluation"]["evaluation_invoked"] is False, "evaluation unexpectedly invoked")
    require(experiment["evaluation"]["hidden_test_access"] is False, "hidden test was accessed")
    require(experiment["proof"]["receipt_cid"] == PGIR_206_RECEIPTS["proof"], "proof receipt drifted")
    require(experiment["proof"]["proof_invoked"] is False, "proof provider unexpectedly invoked")
    require(experiment["training"]["training_started"] is False, "training unexpectedly started")
    require(experiment["checkpoint"]["weights_created"] is False, "weights unexpectedly created")
    require(experiment["resource"]["lease_requested"] is False, "resource lease unexpectedly requested")
    require(experiment["reducer_cas"]["compare_and_swap_attempted"] is False, "promotion CAS unexpectedly attempted")
    require(
        result["observed_effects"]
        == {
            "checkpoint_created": False,
            "evaluation_invoked": False,
            "gpu_probe_performed": False,
            "hidden_labels_opened": False,
            "network_accessed": False,
            "optimizer_steps": 0,
            "promotion_attempted": False,
            "proof_invoked": False,
            "reducer_cas_attempted": False,
            "resource_lease_acquired": False,
            "resource_lease_requested": False,
            "training_started": False,
            "weights_created": False,
        },
        "PGIR-206 observed effects drifted",
    )
    package_prefix = "data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1"
    for relative in git("ls-tree", "-r", "--name-only", PGIR_206_IMPLEMENTATION, "--", package_prefix).splitlines():
        require(git("rev-parse", f"{PGIR_206_IMPLEMENTATION}:{relative}") == git("hash-object", "--", relative), f"PGIR-206 worktree bytes drifted: {relative}")
    return {"experiment": experiment, "forest": forest, "freeze": freeze, "recursive": recursive}


def build_acceptance(inputs: Mapping[str, Any]) -> dict[str, Any]:
    experiment = inputs["experiment"]
    freeze = inputs["freeze"]
    bindings = freeze["root"]["bindings"]
    resolutions = {
        "F01": ("resolved_with_evidence", "The complete successor result graph is closed through verified PGIR-206 typed-not-run evidence.", [PGIR_206_RESULT_CID, PGIR_211_ACCEPTANCE_CID]),
        "F02": ("satisfied", "The successor leakage audit passed and hidden labels stayed sealed.", [bindings["split"]["binding_cid"], experiment["heldouts"]["heldout_cid"]]),
        "F03": ("satisfied_with_limitations", "One deterministic compiler/decompiler bridge is bound; its residual limits remain explicit.", [bindings["compiler"]["binding_cid"], bindings["decompiler"]["binding_cid"]]),
        "F04": ("no_go", "The historical semantic baseline is retired and no current-input measured baseline was admitted.", [freeze["root"]["gates"]["current_baseline"]["retirement_cid"]]),
        "F05": ("satisfied", "Proof-aware arm, metric, and receipt contracts are sealed even though execution was denied.", [experiment["result"]["arm_set_cid"], experiment["result"]["metrics_cid"]]),
        "F06": ("satisfied_execution_denied", "The campaign is resumable in contract only; the freeze and admission receipt denied every lease.", [experiment["campaign"]["campaign_cid"], PGIR_206_RECEIPTS["admission"]]),
        "F07": ("satisfied", "The deterministic reducer recorded no winner and no compare-and-swap attempt.", [experiment["comparison"]["comparison_cid"], PGIR_206_RECEIPTS["reducer_cas"]]),
        "F08": ("satisfied", "Append-only publication policy was honored: no upload was attempted without independent authority.", [HISTORICAL_EVIDENCE_CIDS["PGIR-090"]]),
        "F09": ("no_go", "Token metrics contain explicit not-run cells with zero denominators and unavailable uncertainty.", [PGIR_206_RECEIPTS["evaluation"]]),
        "F10": ("no_go", "Latent and retrieval metrics contain explicit not-run cells with unavailable uncertainty.", [PGIR_206_RECEIPTS["evaluation"]]),
        "F11": ("no_go", "Structural metrics contain explicit not-run cells with unavailable uncertainty.", [PGIR_206_RECEIPTS["evaluation"]]),
        "F12": ("no_go", "Semantic metrics contain explicit not-run cells; historical scores were not substituted.", [PGIR_206_RECEIPTS["evaluation"]]),
        "F13": ("no_go", "Proof metrics were not run and no proof authority was granted.", [PGIR_206_RECEIPTS["proof"]]),
        "F14": ("no_go", "Calibration and OOD metrics contain explicit not-run cells with unavailable uncertainty.", [PGIR_206_RECEIPTS["evaluation"]]),
        "F15": ("no_go", "Latency and resource metrics were not measured because no resource lease was requested.", [PGIR_206_RECEIPTS["resource"]]),
        "F16": ("satisfied", "Hidden tests stayed sealed, no candidate was published, and blocked PGIR-212 is the only following task.", [experiment["heldouts"]["heldout_cid"], PGIR_206_RESULT_CID]),
    }
    rows = []
    for criterion_id, slug, text in ACCEPTANCE_CRITERIA:
        status_value, evidence, evidence_cids = resolutions[criterion_id]
        rows.append(
            {
                "criterion_id": criterion_id,
                "evidence": evidence,
                "evidence_cids": evidence_cids,
                "gate_resolved": True,
                "promotion_gate_pass": False,
                "slug": slug,
                "status": status_value,
                "text": text,
            }
        )
    require([row["status"] for row in rows] == [
        "resolved_with_evidence", "satisfied", "satisfied_with_limitations", "no_go",
        "satisfied", "satisfied_execution_denied", "satisfied", "satisfied",
        "no_go", "no_go", "no_go", "no_go", "no_go", "no_go", "no_go", "satisfied",
    ], "acceptance status vector drifted")
    return add_identity(
        {
            "criteria": rows,
            "criterion_count": 16,
            "interface": "proof-grounded-ir-learning/final-acceptance/v2",
            "promotion_gate_pass_count": 0,
            "schema": "PGIRFinalAcceptanceCatalog@2",
            "task_binding": task_binding(),
        },
        name="acceptance",
    )


def unavailable_metric(family: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "confidence_interval": None,
        "denominator": 0,
        "family": family,
        "hidden_test_used": False,
        "measured_cell_count": 0,
        "missing_as_zero": False,
        "paired_uncertainty": None,
        "reason_code": "admission_closed",
        "source_evaluation_receipt_cid": inputs["experiment"]["evaluation"]["receipt_cid"],
        "status": "not_run",
        "value": None,
    }


def section_bindings(number: int, inputs: Mapping[str, Any]) -> dict[str, Any]:
    experiment = inputs["experiment"]
    freeze = inputs["freeze"]
    bindings = freeze["root"]["bindings"]
    if number == 1:
        return {"pgir_206_forest": inputs["forest"], "recursive_repository": inputs["recursive"]}
    if number == 2:
        return {"pinset_id": "JDAO-PINSET-1", "training_repositories_admitted": 0, "training_admitted_rows": 0}
    if number == 3:
        return {"freeze_manifest_cid": freeze["manifest"]["manifest_cid"], "pgir_206_manifest_cid": PGIR_206_MANIFEST_CID, "pgir_206_result_cid": PGIR_206_RESULT_CID}
    if number == 4:
        return {"derived_count": 38690, "materialized_source_rows": 0, "source_count": 7173, "training_admitted_rows": 0}
    if number == 5:
        return {"hidden_test_commitment": experiment["heldouts"]["hidden_test_commitment"], "insufficient_holdouts": list(INSUFFICIENT_HOLDOUTS), "split_binding_cid": experiment["heldouts"]["split_binding_cid"]}
    if number == 6:
        return {"hidden_labels_opened": False, "leakage_passed": True, "violations": 0}
    if number == 7:
        return {"compiler_binding_cid": bindings["compiler"]["binding_cid"], "decompiler_binding_cid": bindings["decompiler"]["binding_cid"]}
    if number == 8:
        return {"entrypoint": bindings["compiler"]["entrypoint"], "learned_stages": [], "symbolic_alias": bindings["compiler"]["symbolic_alias"]}
    if number == 9:
        return {"entrypoint": bindings["decompiler"]["entrypoint"], "symbolic_alias": bindings["decompiler"]["symbolic_alias"], "uses_model": False}
    if number == 10:
        return {"qualification": "retired_not_currently_qualified", "retirement_cid": freeze["root"]["gates"]["current_baseline"]["retirement_cid"]}
    if number == 11:
        return {"architectures_instantiated": [], "checkpoint_count": 0, "weights_created": False}
    if number == 12:
        return {"policy_cid": bindings["tokenizer_policy"]["policy_cid"], "status": bindings["tokenizer_policy"]["status"], "training_authorized": False}
    if number == 13:
        return {"identity": "IRLossConfiguration@1", "optimizer_steps": 0, "proof_in_gradient_path": False}
    if number == 14:
        return {"arm_count": 6, "campaign_cid": experiment["campaign"]["campaign_cid"], "execution_status": "not_run", "experiment_key_count": 16}
    if number == 15:
        return {"attempts": 0, "closed_authority_vocabulary": True, "proof_invoked": False}
    if number in {16, 17, 18}:
        return {"attempts_admitted_to_curriculum": 0, "provider_invoked": False, "role": "proposal_only"}
    if number == 19:
        return {"authority_granted": False, "checked_proof_count": 0, "source_proof_receipt_cid": PGIR_206_RECEIPTS["proof"]}
    metric_families = {20: "token", 21: "latent_retrieval", 22: "latent_retrieval", 23: "structural", 24: "semantic", 25: "proof", 26: "calibration_ood", 27: "latency_resource"}
    if number in metric_families:
        return unavailable_metric(metric_families[number], inputs)
    if number == 28:
        return {"execution_authorized": False, "pgir_206_forest": inputs["forest"], "resource_lease_acquired": False}
    if number == 29:
        return {"candidate_checkpoint": None, "decision": "no_go", "m2_gates": list(M2_GATES), "pointer_mutated": False}
    if number == 30:
        return {"remote_revision": None, "upload_attempted": False, "upload_authorized": False}
    if number == 31:
        return {"reason_codes": list(REASON_CODES)}
    if number == 32:
        return {"descendant_task_ids": [], "next_board_path": "docs/architecture/proof_grounded_ir_learning/successor-v1/next.todo.md", "next_task_ids": ["PGIR-212"]}
    raise QualificationBuildError(f"unknown report section {number}")


def section_evidence_cids(number: int, inputs: Mapping[str, Any]) -> list[str]:
    experiment = inputs["experiment"]
    root = inputs["freeze"]["root"]
    if number == 1:
        return [PGIR_206_RESULT_CID]
    if number == 2:
        return [CAMPAIGN_INPUT_ROOT_CID]
    if number == 3:
        return [PGIR_206_MANIFEST_CID, PGIR_206_RESULT_CID]
    if number == 4:
        return [root["bindings"]["rights"]["binding_cid"], root["bindings"]["corpus"]["binding_cid"]]
    if number in {5, 6}:
        return [experiment["heldouts"]["heldout_cid"]]
    if number in {7, 8}:
        return [root["bindings"]["compiler"]["binding_cid"]]
    if number == 9:
        return [root["bindings"]["decompiler"]["binding_cid"]]
    if number == 10:
        return [root["gates"]["current_baseline"]["retirement_cid"]]
    if number in {11, 12, 13, 14, 15}:
        return [experiment["campaign"]["campaign_cid"]]
    if number in {16, 17, 18, 19, 25}:
        return [PGIR_206_RECEIPTS["proof"]]
    if number in {20, 21, 22, 23, 24, 26}:
        return [PGIR_206_RECEIPTS["evaluation"]]
    if number == 27:
        return [PGIR_206_RECEIPTS["resource"]]
    if number == 28:
        return [PGIR_206_RESULT_CID]
    if number == 29:
        return [PGIR_206_RECEIPTS["reducer_cas"]]
    if number == 30:
        return [HISTORICAL_EVIDENCE_CIDS["PGIR-090"]]
    if number in {31, 32}:
        return [PGIR_206_RESULT_CID]
    raise QualificationBuildError(f"unknown report section evidence {number}")


def build_sections(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "bindings": section_bindings(number, inputs),
            "evidence_cids": section_evidence_cids(number, inputs),
            "number": number,
            "slug": slug,
            "status": SECTION_STATUS[number],
            "summary": SECTION_SUMMARIES[number],
            "title": title,
        }
        for number, slug, title in REPORT_SECTIONS
    ]
    require(len(rows) == 32 and [row["number"] for row in rows] == list(range(1, 33)), "report section order drifted")
    return add_identity(
        {
            "interface": "proof-grounded-ir-learning/final-report-sections/v2",
            "schema": "PGIRFinalReportSectionCatalog@2",
            "section_count": 32,
            "sections": rows,
            "task_binding": task_binding(),
        },
        name="sections",
    )


def build_evaluation_receipt(inputs: Mapping[str, Any]) -> dict[str, Any]:
    experiment = inputs["experiment"]
    source = experiment["evaluation"]
    require(source["metric_cell_count"] == 192 and source["measured_cell_count"] == 0, "evaluation counts drifted")
    require(source["target_attainment_claim_count"] == 0, "target attainment was claimed")
    return add_identity(
        {
            "authorizes_promotion": False,
            "best_test_selection": False,
            "campaign_cid": experiment["campaign"]["campaign_cid"],
            "checked_metric_cell_count": 192,
            "comparison_cid": experiment["comparison"]["comparison_cid"],
            "evaluation_invoked": False,
            "hidden_labels_opened": False,
            "hidden_test_access": False,
            "hidden_test_selection": False,
            "hidden_test_tuning": False,
            "measured_cell_count": 0,
            "metric_cell_count": 192,
            "metrics_cid": experiment["result"]["metrics_cid"],
            "missing_metric_as_zero": False,
            "pgir_206_result_cid": PGIR_206_RESULT_CID,
            "schema": "PGIRQualificationEvaluationReceipt@1",
            "source_receipt_cid": source["receipt_cid"],
            "status": "verified_typed_not_run",
            "target_attainment_claim_count": 0,
            "task_binding": task_binding(),
            "uncertainty_substituted": False,
            "verifier_interface": "pgir-terminal-qualification-evaluation/v1",
        },
        name="evaluation",
    )


def build_proof_receipt(inputs: Mapping[str, Any]) -> dict[str, Any]:
    experiment = inputs["experiment"]
    source = experiment["proof"]
    return add_identity(
        {
            "proof_attempt_count": 0,
            "authority_granted": False,
            "authorizes_promotion": False,
            "campaign_cid": experiment["campaign"]["campaign_cid"],
            "checked_proof_count": 0,
            "hidden_labels_opened": False,
            "kernel_check_count": 0,
            "nondifferentiable": True,
            "pgir_206_result_cid": PGIR_206_RESULT_CID,
            "proof_invoked": False,
            "proof_invocation_count": 0,
            "proof_results": [],
            "schema": "PGIRQualificationProofReceipt@1",
            "source_receipt_cid": source["receipt_cid"],
            "status": "verified_typed_not_run",
            "task_binding": task_binding(),
            "timeout_as_falsehood": False,
            "training_receipt_cid": PGIR_206_RECEIPTS["training"],
            "verifier_interface": "pgir-terminal-qualification-proof/v1",
        },
        name="proof",
    )


def build_decision(
    acceptance: Mapping[str, Any],
    sections: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    no_go_criteria = [row["criterion_id"] for row in acceptance["criteria"] if row["status"] == "no_go"]
    require(no_go_criteria == ["F04", "F09", "F10", "F11", "F12", "F13", "F14", "F15"], "no-go criteria drifted")
    return add_identity(
        {
            "acceptance_cid": acceptance["acceptance_cid"],
            "all_criteria_resolved": True,
            "all_sections_resolved": True,
            "candidate_checkpoint": None,
            "criterion_count": 16,
            "decision": "no_go",
            "disposition": "qualified_no_go",
            "evaluator_or_model_held_authority": False,
            "evaluation_cid": evaluation["evaluation_cid"],
            "execution_status": "not_run",
            "hidden_tests_opened": False,
            "human_approval": False,
            "interface": "proof-grounded-ir-learning/qualification-decision/v2",
            "no_go_criterion_ids": no_go_criteria,
            "pgir_206_result_cid": PGIR_206_RESULT_CID,
            "proof_cid": proof["proof_cid"],
            "publication_authorized": False,
            "qualified_claim_emitted": False,
            "reason_codes": list(REASON_CODES),
            "schema": "PGIRQualificationDecision@2",
            "section_count": 32,
            "sections_cid": sections["sections_cid"],
            "task_binding": task_binding(),
            "training_admitted_rows": 0,
        },
        name="decision",
    )


def build_promotion_receipt(decision: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    source = inputs["experiment"]["reducer_cas"]
    return add_identity(
        {
            "admitted_gates": [],
            "candidate_checkpoint": None,
            "cas_attempted": False,
            "decision": "no_go",
            "decision_cid": decision["decision_cid"],
            "human_approval": False,
            "interface": "proof-grounded-ir-learning/qualification-promotion/v2",
            "lease_key": "promotion-pointer",
            "m2_gates": list(M2_GATES),
            "observed_pointer_after": None,
            "observed_pointer_before": None,
            "pointer_mutated": False,
            "pointer_unchanged": True,
            "reason_codes": ["no_candidate_checkpoint", "qualification_gates_failed", "publication_not_authorized"],
            "required_gates": list(M2_GATES),
            "schema": "PGIRQualificationPromotionReceipt@2",
            "self_promotion": False,
            "source_reducer_cas_receipt_cid": source["receipt_cid"],
            "promotion_authorized": False,
            "task_binding": task_binding(),
        },
        name="promotion",
    )


def build_publication_receipt(decision: Mapping[str, Any]) -> dict[str, Any]:
    return add_identity(
        {
            "append_only": True,
            "decision": "denied",
            "decision_cid": decision["decision_cid"],
            "human_approval": False,
            "interface": "proof-grounded-ir-learning/qualification-publication/v2",
            "lease_key": "hf-publication:Publicus/proof-grounded-ir-learning",
            "local_package_qualified": False,
            "network_accessed": False,
            "publication_authorized": False,
            "reason_codes": ["qualification_gates_failed", "publication_not_authorized", "manual_external_authority_absent"],
            "remote_revision": None,
            "schema": "PGIRQualificationPublicationReceipt@2",
            "task_binding": task_binding(),
            "trust_remote_code": False,
            "upload_attempted": False,
            "upload_authorized": False,
        },
        name="publication",
    )


def graph_evidence(inputs: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, str]:
    root = inputs["freeze"]["root"]
    integrated = inputs["freeze"]["integrated"]
    return {
        **HISTORICAL_EVIDENCE_CIDS,
        "PGIR-200": root["bindings"]["rights"]["binding_cid"],
        "PGIR-201": root["bindings"]["corpus"]["binding_cid"],
        "PGIR-202": root["bindings"]["split"]["binding_cid"],
        "PGIR-203": root["bindings"]["tokenizer_policy"]["result_cid"],
        "PGIR-204": root["gates"]["current_baseline"]["retirement_cid"],
        "PGIR-208": integrated["predecessor_acceptance_cids"]["PGIR-208"],
        "PGIR-209": integrated["predecessor_acceptance_cids"]["PGIR-209"],
        "PGIR-210": integrated["predecessor_acceptance_cids"]["PGIR-210"],
        "PGIR-211": integrated["acceptance_cid"],
        "PGIR-205": PGIR_205_RESULT_CID,
        "PGIR-206": PGIR_206_RESULT_CID,
        "PGIR-207": decision["decision_cid"],
    }


def build_result_graph(inputs: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    evidence = graph_evidence(inputs, decision)
    dispositions = {
        "PGIR-072": "historical_completed",
        "PGIR-090": "historical_completed",
        "PGIR-100": "historical_completed",
        "PGIR-111": "historical_qualified_no_go",
        "PGIR-200": "permanent_no_go",
        "PGIR-201": "not_materialized",
        "PGIR-202": "permanent_no_go",
        "PGIR-203": "permanently_deterministic_only",
        "PGIR-204": "retired",
        "PGIR-208": "qualified_no_go",
        "PGIR-209": "qualified_no_go",
        "PGIR-210": "qualified_no_go",
        "PGIR-211": "qualified_no_go",
        "PGIR-205": "frozen_no_go",
        "PGIR-206": "typed_not_run",
        "PGIR-207": "qualified_no_go",
    }
    nodes = []
    edges = []
    seen: set[str] = set()
    for position, task_id in enumerate(DEPENDENCY_FIRST_ORDER):
        dependencies = list(GRAPH_DEPENDENCIES[task_id])
        require(all(item in seen for item in dependencies), f"dependency-first order drifted at {task_id}")
        nodes.append(
            {
                "depends_on": dependencies,
                "dependency_first_position": position,
                "disposition": dispositions[task_id],
                "evidence_cid": evidence[task_id],
                "historical_leaf": task_id in HISTORICAL_TASK_CIDS,
                "node_id": f"result:{task_id}",
                "result_identity": f"RESULT({task_id})",
                "task_cid": HISTORICAL_TASK_CIDS.get(task_id, TASK_CID if task_id == TASK_ID else None),
                "task_id": task_id,
            }
        )
        for dependency in dependencies:
            edges.append({"dependency": dependency, "dependent": task_id, "edge_id": f"{task_id}->{dependency}"})
        seen.add(task_id)
    require(len(nodes) == 16, "result graph node count drifted")
    require(len(edges) == 28, "result graph edge count drifted")
    require([node["task_id"] for node in nodes if node["historical_leaf"]] == ["PGIR-072", "PGIR-090", "PGIR-100", "PGIR-111"], "historical leaves drifted")
    return add_identity(
        {
            "dependency_first_order": list(DEPENDENCY_FIRST_ORDER),
            "edge_count": 28,
            "edges": edges,
            "historical_leaf_count": 4,
            "interface": "proof-grounded-ir-learning/complete-result-graph/v1",
            "node_count": 16,
            "nodes": nodes,
            "root_node_id": "result:PGIR-207",
            "schema": "PGIRCompleteResultGraph@1",
            "task_binding": task_binding(),
        },
        name="graph",
    )


def build_recipe(
    *,
    acceptance: Mapping[str, Any],
    sections: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    proof: Mapping[str, Any],
    decision: Mapping[str, Any],
    promotion: Mapping[str, Any],
    publication: Mapping[str, Any],
    graph: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    return add_identity(
        {
            "acceptance_cid": acceptance["acceptance_cid"],
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "decision_cid": decision["decision_cid"],
            "depends_on": ["PGIR-072", "PGIR-090", "PGIR-100", "PGIR-206"],
            "evaluation_cid": evaluation["evaluation_cid"],
            "hidden_test_selection": False,
            "interface": "proof-grounded-ir-learning/qualification-recipe/v2",
            "lease_policy": "LEASE-DEFAULT",
            "missing_metric_as_zero": False,
            "pgir_206_forest": inputs["forest"],
            "pgir_206_manifest_cid": PGIR_206_MANIFEST_CID,
            "pgir_206_result_cid": PGIR_206_RESULT_CID,
            "proof_cid": proof["proof_cid"],
            "promotion_cid": promotion["promotion_cid"],
            "publication_cid": publication["publication_cid"],
            "reason_codes": list(REASON_CODES),
            "resource_profile": "RP-CPU-M",
            "result_graph_cid": graph["graph_cid"],
            "schema": "PGIRQualificationRecipe@2",
            "sections_cid": sections["sections_cid"],
            "task_binding": task_binding(),
        },
        name="recipe",
    )


def render_final_report(
    *,
    acceptance: Mapping[str, Any],
    sections: Mapping[str, Any],
    decision: Mapping[str, Any],
    promotion: Mapping[str, Any],
    publication: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> str:
    lines = [
        "# Proof-Grounded IR Learning Fabric successor terminal report",
        "",
        "This report is `RESULT(PGIR-207)`. It applies the closed sixteen final",
        "criteria and thirty-two report sections to the exact typed `not_run`",
        "evidence from `RESULT(PGIR-206)`. Missing evidence is never inferred as a",
        "pass, zero, or measured value. The qualified-claim text is withheld.",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Decision CID: `{decision['decision_cid']}`",
        f"- PGIR-206 result CID: `{PGIR_206_RESULT_CID}`",
        f"- Complete result graph CID: `{graph['graph_cid']}`",
        f"- Promotion receipt CID: `{promotion['promotion_cid']}`",
        f"- Publication receipt CID: `{publication['publication_cid']}`",
        "- Qualified claim emitted: `false`",
        "- Following automated descendants: `0`",
        "",
        "## Final acceptance criteria",
        "",
    ]
    for row in acceptance["criteria"]:
        lines.extend(
            [
                f"### {row['criterion_id']} {row['text']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Evidence: {row['evidence']}",
                f"- Evidence CIDs: `{json.dumps(row['evidence_cids'], separators=(',', ':'))}`",
                "",
            ]
        )
    for row in sections["sections"]:
        lines.extend(
            [
                f"## {row['number']}. {row['title']}",
                "",
                row["summary"],
                "",
                f"- Section status: `{row['status']}`",
            ]
        )
        for key in sorted(row["bindings"]):
            rendered = json.dumps(row["bindings"][key], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            lines.append(f"- `{key}`: `{rendered}`")
        lines.append("")
    lines.extend(
        [
            "## Authorized closing claim",
            "",
            "The terminal qualification decision is `no_go`. No learned campaign",
            "ran, no metric was measured, no proof authority was granted, no",
            "checkpoint exists, no promotion pointer changed, and no upload was",
            "attempted. PGIR-212 is blocked on manual external evidence and is not",
            "schedulable. This integrity-preserving no-go is the complete result.",
            "",
            "Reason codes:",
            "",
        ]
    )
    lines.extend(f"- `{code}`" for code in REASON_CODES)
    lines.extend(["", "Never claim universal legal-semantic understanding.", ""])
    text = "\n".join(lines)
    require(QUALIFIED_CLAIM not in text, "qualified claim leaked into terminal report")
    require(sum(f"\n## {number}. " in text for number in range(1, 33)) == 32, "terminal report section population drifted")
    return text


def render_next_board(decision: Mapping[str, Any], graph: Mapping[str, Any]) -> str:
    text = "\n".join(
        [
            "# Proof-Grounded IR Learning Fabric Following Board",
            "",
            "This board is issued by `RESULT(PGIR-207)` after the terminal",
            f"`no_go` decision `{decision['decision_cid']}` and complete result graph",
            f"`{graph['graph_cid']}`. It contains no runnable automated work.",
            "",
            "No worker, scheduler, model, evaluator, or publication client may",
            "reinterpret this board as authority to train, prove, promote, publish,",
            "or open hidden evidence. A human operator must first admit new external",
            "rights, corpus, holdout, tokenizer, baseline, portability, execution,",
            "promotion, and publication evidence through a separately reviewed plan.",
            "",
            "## PGIR-212 Admit manual external evidence for any future campaign",
            "",
            "- Status: blocked",
            "- Completion: manual-external-evidence",
            "- Is schedulable: false",
            "- Priority: P0",
            "- Track: qualification",
            "- Parent goal: PGIR-G110",
            "- Subgoal: external-authority-admission",
            "- Owning repository: manual-external-authority",
            "- Owned paths: none",
            "- Objective: Admit independently reviewed external evidence before proposing any successor campaign or publication action.",
            "- Depends on: PGIR-207",
            "- Resource profile: MANUAL-EXTERNAL-EVIDENCE",
            "- Expected inputs: externally authorized rights, materialized corpus, complete holdouts, admitted tokenizer, current baseline, portability, execution, promotion, and publication evidence",
            "- Expected outputs: a separately reviewed plan revision; this board grants no output mutation",
            "- Allowed effects: none under this board",
            "- Prohibited effects: automated execution, training, proof invocation, hidden-test access, promotion, publication, descendant task creation",
            "- Acceptance criteria: a human-controlled external authority explicitly admits every previously failed gate and issues a new reviewed board",
            "- Required proof or evaluation evidence: manual external evidence only; RESULT(PGIR-207) cannot authorize its own supersession",
            "- Lease and checkpoint policy: no lease; no checkpoint",
            "- Rollback procedure: retain RESULT(PGIR-207) and reject the proposed supersession",
            "- Result identity: none until a separately reviewed plan creates a new task",
            "- Outputs: none",
            "- Validation: none; manual external evidence required",
            "- Bundle: none",
            "- Parallel lane: none",
            "- Predicted files: none",
            "- Conflict policy: no automated writer",
            "- Descendants: none",
            "",
        ]
    )
    require(text.count("\n## PGIR-") == 1, "following board must contain exactly one task")
    require("- Status: blocked" in text and "- Is schedulable: false" in text, "PGIR-212 is not terminally blocked")
    require("python " not in text and "pytest" not in text and "scripts/" not in text, "following board contains runnable validation")
    require(QUALIFIED_CLAIM not in text, "qualified claim leaked into following board")
    return text


def repository_path(relative: str) -> Path:
    path = REPOSITORY_ROOT / relative
    require(path.resolve(strict=False).is_relative_to(REPOSITORY_ROOT.resolve()), f"path escapes repository: {relative}")
    return path


def build_manifest(
    *,
    sealed_bytes: Mapping[str, bytes],
    decision: Mapping[str, Any],
    graph: Mapping[str, Any],
    recipe: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    require(tuple(sealed_bytes) == SEALED_PATHS, "sealed path order drifted")
    artifacts = {relative: artifact_record(sealed_bytes[relative]) for relative in SEALED_PATHS}
    return add_identity(
        {
            "artifact_count": 14,
            "artifacts": artifacts,
            "decision": "no_go",
            "decision_cid": decision["decision_cid"],
            "excluded_paths": list(FINAL_PATHS[-3:]),
            "immutability": "supersede_never_overwrite",
            "interface": "proof-grounded-ir-learning/qualification-manifest/v2",
            "pgir_206_forest": inputs["forest"],
            "pgir_206_manifest_cid": PGIR_206_MANIFEST_CID,
            "pgir_206_result_cid": PGIR_206_RESULT_CID,
            "recipe_cid": recipe["recipe_cid"],
            "result_graph_cid": graph["graph_cid"],
            "schema": "PGIRTerminalQualificationManifest@1",
            "sealed_path_order": list(SEALED_PATHS),
            "task_binding": task_binding(),
        },
        name="manifest",
    )


def build_verification_receipt(
    *,
    acceptance: Mapping[str, Any],
    sections: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    proof: Mapping[str, Any],
    decision: Mapping[str, Any],
    promotion: Mapping[str, Any],
    publication: Mapping[str, Any],
    graph: Mapping[str, Any],
    recipe: Mapping[str, Any],
    manifest: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        ("recursive-repository-forest", inputs["recursive"]["outer_commit"]),
        ("pgir-206-implementation-merge-completion", PGIR_206_COMPLETION),
        ("pgir-206-result", PGIR_206_RESULT_CID),
        ("pgir-206-manifest", PGIR_206_MANIFEST_CID),
        ("acceptance-16-resolved", acceptance["acceptance_cid"]),
        ("report-sections-32-resolved", sections["sections_cid"]),
        ("evaluation-typed-not-run", evaluation["evaluation_cid"]),
        ("proof-typed-not-run", proof["proof_cid"]),
        ("terminal-no-go-decision", decision["decision_cid"]),
        ("promotion-pointer-unchanged", promotion["promotion_cid"]),
        ("publication-not-attempted", publication["publication_cid"]),
        ("complete-result-graph-16x28", graph["graph_cid"]),
        ("qualification-recipe", recipe["recipe_cid"]),
        ("sealed-artifact-manifest-14", manifest["manifest_cid"]),
        ("following-board-pgir-212-blocked", "PGIR-212"),
    ]
    return add_identity(
        {
            "all_integrity_checks_passed": True,
            "authorizes_execution": False,
            "authorizes_promotion": False,
            "authorizes_publication": False,
            "checks": [
                {"check_id": check_id, "evidence": evidence, "status": "passed"}
                for check_id, evidence in checks
            ],
            "decision": "no_go",
            "decision_cid": decision["decision_cid"],
            "manifest_cid": manifest["manifest_cid"],
            "result_graph_cid": graph["graph_cid"],
            "schema": "PGIRQualificationVerificationReceipt@1",
            "task_binding": task_binding(),
            "verification_verdict": "verified_terminal_no_go",
            "verifier_interface": "pgir-terminal-qualification-independent-verifier/v1",
        },
        name="verification",
    )


def build_result(
    *,
    decision: Mapping[str, Any],
    graph: Mapping[str, Any],
    recipe: Mapping[str, Any],
    manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    return add_identity(
        {
            "automated_descendant_count": 0,
            "campaign_input_root_cid": CAMPAIGN_INPUT_ROOT_CID,
            "candidate_checkpoint": None,
            "completion_authoritative": False,
            "decision": "no_go",
            "decision_cid": decision["decision_cid"],
            "dependency_task_cids": {
                "PGIR-072": HISTORICAL_TASK_CIDS["PGIR-072"],
                "PGIR-090": HISTORICAL_TASK_CIDS["PGIR-090"],
                "PGIR-100": HISTORICAL_TASK_CIDS["PGIR-100"],
                "PGIR-206": "baguqeerafze3wxxiomo4rhuxaguuk35d4tj4lrp2jtuj6jcinxnokczuctca",
            },
            "disposition": "qualified_no_go",
            "execution_status": "not_run",
            "execution_authorized": False,
            "hidden_tests_opened": False,
            "manifest_cid": manifest["manifest_cid"],
            "next_task_ids": ["PGIR-212"],
            "objective_id": OBJECTIVE_ID,
            "objective_revision": TASK_CID,
            "parent_goal": PARENT_GOAL,
            "pgir_206_forest": inputs["forest"],
            "pgir_206_result_cid": PGIR_206_RESULT_CID,
            "promotion_authorized": False,
            "publication_authorized": False,
            "qualified_claim_emitted": False,
            "reason_codes": list(REASON_CODES),
            "recipe_cid": recipe["recipe_cid"],
            "recursive_repository": inputs["recursive"],
            "result_graph_cid": graph["graph_cid"],
            "result_identity": RESULT_IDENTITY,
            "rollback": "retain this immutable terminal qualification; only a separately reviewed manual plan may supersede it",
            "schema": "pgir-task-result@1",
            "source_tree_id": PGIR_206_COMPLETION_TREE,
            "subgoal": SUBGOAL,
            "task_cid": TASK_CID,
            "task_binding": task_binding(),
            "task_id": TASK_ID,
            "task_key": TASK_KEY,
            "training_admitted_rows": 0,
            "verification_cid": verification["verification_cid"],
        },
        name="result",
    )


def materialize(*, check: bool) -> dict[str, Any]:
    inputs = load_inputs(check=check)
    acceptance = build_acceptance(inputs)
    sections = build_sections(inputs)
    evaluation = build_evaluation_receipt(inputs)
    proof = build_proof_receipt(inputs)
    decision = build_decision(acceptance, sections, evaluation, proof)
    promotion = build_promotion_receipt(decision, inputs)
    publication = build_publication_receipt(decision)
    graph = build_result_graph(inputs, decision)
    recipe = build_recipe(
        acceptance=acceptance,
        sections=sections,
        evaluation=evaluation,
        proof=proof,
        decision=decision,
        promotion=promotion,
        publication=publication,
        graph=graph,
        inputs=inputs,
    )
    report = render_final_report(
        acceptance=acceptance,
        sections=sections,
        decision=decision,
        promotion=promotion,
        publication=publication,
        graph=graph,
    ).encode("utf-8")
    following_board = render_next_board(decision, graph).encode("utf-8")
    generated = {
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/acceptance.json": rendered_bytes(acceptance),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/decision.json": rendered_bytes(decision),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/evaluation_receipt.json": rendered_bytes(evaluation),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/proof_receipt.json": rendered_bytes(proof),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/promotion_receipt.json": rendered_bytes(promotion),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/publication_receipt.json": rendered_bytes(publication),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/recipe.json": rendered_bytes(recipe),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/report_sections.json": rendered_bytes(sections),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/result_graph.json": rendered_bytes(graph),
        "docs/architecture/proof_grounded_ir_learning/successor-v1/final_report.md": report,
        "docs/architecture/proof_grounded_ir_learning/successor-v1/next.todo.md": following_board,
    }
    source_paths = {
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/README.md",
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/build_qualification.py",
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verify_qualification.py",
    }
    sealed_bytes: dict[str, bytes] = {}
    for relative in SEALED_PATHS:
        if relative in generated:
            sealed_bytes[relative] = generated[relative]
            continue
        require(relative in source_paths, f"unclassified sealed path: {relative}")
        path = repository_path(relative)
        require(path.is_file() and not path.is_symlink(), f"missing sealed source: {relative}")
        mode = path.lstat().st_mode
        permission = stat.S_IMODE(mode)
        require(
            stat.S_ISREG(mode) and permission & 0o111 == 0 and permission & 0o002 == 0,
            f"unsafe sealed source mode: {relative}",
        )
        sealed_bytes[relative] = path.read_bytes()
    manifest = build_manifest(
        sealed_bytes=sealed_bytes,
        decision=decision,
        graph=graph,
        recipe=recipe,
        inputs=inputs,
    )
    verification = build_verification_receipt(
        acceptance=acceptance,
        sections=sections,
        evaluation=evaluation,
        proof=proof,
        decision=decision,
        promotion=promotion,
        publication=publication,
        graph=graph,
        recipe=recipe,
        manifest=manifest,
        inputs=inputs,
    )
    result = build_result(
        decision=decision,
        graph=graph,
        recipe=recipe,
        manifest=manifest,
        verification=verification,
        inputs=inputs,
    )
    final_bytes = {
        **generated,
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/manifest.json": rendered_bytes(manifest),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verification_receipt.json": rendered_bytes(verification),
        "data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/result.json": rendered_bytes(result),
    }
    require(set(final_bytes) | source_paths == set(FINAL_PATHS), "final artifact population drifted")
    for relative in FINAL_PATHS:
        if relative in source_paths:
            continue
        write_once(repository_path(relative), final_bytes[relative], check=check)
    return {
        "artifact_count": 17,
        "decision": decision["decision"],
        "manifest_cid": manifest["manifest_cid"],
        "next_task_id": "PGIR-212",
        "result_cid": result["result_cid"],
        "result_graph_cid": graph["graph_cid"],
        "task_id": TASK_ID,
        "verified_terminal_no_go": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="read-only replay; require all seventeen artifact bytes to be exact",
    )
    args = parser.parse_args(argv)
    try:
        summary = materialize(check=args.check)
    except (QualificationBuildError, OSError, UnicodeError, ValueError) as exc:
        print(f"PGIR-207 build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
