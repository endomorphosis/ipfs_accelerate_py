#!/usr/bin/env python3
"""Validate the sealed Tactician/Hammer logic-repair program board.

This validator is intentionally deterministic and side-effect free.  It checks
the finite goal/task DAG and the scheduler's safety policy; live checkout and
provider checks belong to the launcher's ``doctor`` command.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    RECONCILIATION_GUARDRAIL_SCHEMA,
    RECONCILIATION_RESOLUTION_SCHEMA,
    RETRY_BUDGET_REPAIR_SCHEMA,
    parse_task_file,
    retry_budget_repair_source,
)

PLAN_PATH = Path(
    "docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md"
)
OBJECTIVE_PATH = Path(
    "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md"
)
TODO_PATH = Path(
    "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.todo.md"
)
SCHEDULER_PATH = Path(
    "config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json"
)
VALIDATOR_PATH = Path("scripts/validate_tactician_hammer_logic_repair_board.py")
LAUNCHER_PATH = Path("scripts/tactician_hammer_logic_repair_supervisor.sh")
BOOTSTRAP_TEST_PATH = Path(
    "test/api/test_agent_supervisor_tactician_hammer_logic_repair_bootstrap.py"
)
RPR_TODO_PATH = Path(
    "docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md"
)

TASK_PREFIX = "LPR-"
BOARD_NAMESPACE = "agent-supervisor-tactician-hammer-logic-repair-v1"
TARGET_BRANCH = "agent/proof-gated-contract-repair"
DATASETS_TACTICIAN_ANCESTOR = "014b8ea69721d8e0f0cd15b36b83bc5e8bb6a29c"
DATASETS_TACTICIAN_INTERFACE = "ipfs_datasets_py.logic.tactician@1"
DATASETS_REQUIRED_PATHS = (
    "ipfs_datasets_py/logic/tactician/__init__.py",
    "ipfs_datasets_py/logic/tactician/models.py",
    "ipfs_datasets_py/logic/tactician/planner.py",
    "ipfs_datasets_py/logic/tactician/policy.py",
    "ipfs_datasets_py/logic/tactician/receipts.py",
    "ipfs_datasets_py/logic/tactician/adapters.py",
    "ipfs_datasets_py/logic/hammers/models.py",
    "ipfs_datasets_py/logic/hammers/policy.py",
    "ipfs_datasets_py/logic/hammers/portfolio.py",
    "ipfs_datasets_py/logic/hammers/reconstruction.py",
    "ipfs_datasets_py/logic/hammers/receipts.py",
    "ipfs_datasets_py/logic/hammers/proof_cache.py",
    "ipfs_datasets_py/logic/common/proof_cache.py",
    "ipfs_datasets_py/logic/proof_corpus/applicability.py",
    "ipfs_datasets_py/logic/proof_corpus/verifier.py",
    "ipfs_datasets_py/logic/intent_ir/graphrag/retrieval.py",
    "ipfs_datasets_py/knowledge_graphs/cypher/ast.py",
    "ipfs_datasets_py/knowledge_graphs/cypher/parser.py",
    "ipfs_datasets_py/embeddings/generation_engine.py",
    "tests/unit/logic/tactician/test_models.py",
    "tests/unit/logic/tactician/test_planner.py",
)
EXPECTED_TASK_IDS = tuple(f"LPR-{number:03d}" for number in range(43))
MAX_OPERATIONAL_RETRY_REPAIR_TASKS = len(EXPECTED_TASK_IDS) * 3
LEGACY_OPERATIONAL_REPAIR_TASK_IDS = frozenset({"LPR-043", "LPR-044"})
RECONCILIATION_REASONS_BY_KIND = {
    "dirty_backlogged_worktree": frozenset(
        {
            "content_not_in_target",
            "dirty_worktree",
            "empty_status_path",
            "unsupported_status",
        }
    ),
    "main_checkout_dirty": frozenset({"main_checkout_dirty"}),
    "preflight_merge_conflict": frozenset({"preflight_merge_conflict"}),
}
RECONCILIATION_GUARDRAIL_KINDS = frozenset(RECONCILIATION_REASONS_BY_KIND)
MAX_ACTIVE_OPERATIONAL_RECONCILIATION_TASKS = sum(
    len(reasons) for reasons in RECONCILIATION_REASONS_BY_KIND.values()
)
MAX_OPERATIONAL_RECONCILIATION_TASKS = len(EXPECTED_TASK_IDS) * 3
MAX_OPERATIONAL_REPAIR_TASKS = (
    MAX_OPERATIONAL_RETRY_REPAIR_TASKS
    + MAX_OPERATIONAL_RECONCILIATION_TASKS
)
EXPECTED_GOAL_IDS = (
    "LPR-G000",
    "LPR-G010",
    "LPR-G020",
    "LPR-G030",
    "LPR-G040",
    "LPR-G050",
    "LPR-G060",
    "LPR-G070",
    "LPR-G080",
    "LPR-G090",
    "LPR-G100",
    "LPR-G110",
)
EXPECTED_GOAL_TASK_IDS = {
    "LPR-G000": ("LPR-000",),
    "LPR-G010": tuple(f"LPR-{number:03d}" for number in range(1, 5)),
    "LPR-G020": tuple(f"LPR-{number:03d}" for number in range(5, 8)),
    "LPR-G030": tuple(f"LPR-{number:03d}" for number in range(8, 11)),
    "LPR-G040": tuple(f"LPR-{number:03d}" for number in range(11, 15)),
    "LPR-G050": tuple(f"LPR-{number:03d}" for number in range(15, 19)),
    "LPR-G060": tuple(f"LPR-{number:03d}" for number in range(19, 21)),
    "LPR-G070": tuple(f"LPR-{number:03d}" for number in range(21, 29)),
    "LPR-G080": tuple(f"LPR-{number:03d}" for number in range(29, 34)),
    "LPR-G090": tuple(f"LPR-{number:03d}" for number in range(34, 37)),
    "LPR-G100": tuple(f"LPR-{number:03d}" for number in range(37, 40)),
    "LPR-G110": tuple(f"LPR-{number:03d}" for number in range(40, 43)),
}
SEALED_TASK_CIDS = {
    "LPR-000": "baguqeeraghmkwno643c75mfl6wkop527fctnlvr2vcp75hqgjezjbtwykfba",
    "LPR-001": "baguqeerayc34j6hwclkgxtvpdtzrz2jeg4too7svhefrkalj4j3en33xj7za",
    "LPR-002": "baguqeeraxap7q3pgjkq52kigah7zonlyf2qggdqihdg5rgirwcdchatrmwqa",
    "LPR-003": "baguqeeraocf3cpabiqbnprvhd5xgozsm3krhcd5lx4kdhd4e2cko3fckxyoa",
    "LPR-004": "baguqeeraomaxlzfz65p3w54n4p5dqviob55kp6fmbbl76vpz3uiiqkyzasba",
    "LPR-005": "baguqeerasuk2vq2a5bebcbbagnf74tyctffyvioiip6unret7hvqggqpuhbq",
    "LPR-006": "baguqeera336ia7zhyqowqeumksvivc74hsi2b3cab6eq6sj7hx3xes2peijq",
    "LPR-007": "baguqeera4vofxqdgmufuwzvgqc2cgznnfuwlcji5dvvbw64nt3f6q3sehdbq",
    "LPR-008": "baguqeerab4c55bq2xgnj54u6je7bo7ad3kog3iqepdgrej6pk2ktr7bqhasa",
    "LPR-009": "baguqeeraewfuaopv5oq5nvdaxugnmlt3p4oirpqr6skeozegjrmcpkhb7uuq",
    "LPR-010": "baguqeerasbdu7kd7wmljmv3yaati6ustghuxvgjeed7k7e7ceg5atctmaayq",
    "LPR-011": "baguqeerax4ljlzcnvmrrsd23aet3rgzgb7mh4mzsumwda7f5ucnqxgljfqkq",
    "LPR-012": "baguqeeraoiq7u3uvj7o6xohs67pwp5cvwrdik7sho7wumnvtemxx3agdzwpq",
    "LPR-013": "baguqeeraiqiejrxknzaiolzdsbrj6n5sf5b5tpdzyw2pvv7a42yvwe6s4tmq",
    "LPR-014": "baguqeeraghaogvemb5mkx6inric73a2ihw7iwqjoa27k4g4y6aei2ma5ik5a",
    "LPR-015": "baguqeeracztvzyzvi5jqktj3xgi5tmhl7z6wof25uic7oa2dafl3p4bqejva",
    "LPR-016": "baguqeerakajl2hwt25v4p5vzxw36vvtrdhkhspjtch6nargg43yr4sfgokga",
    "LPR-017": "baguqeerazuh55ipsotr4techk3pkypbsnycrs3wiv72qaxhpnlnp26zoyqqa",
    "LPR-018": "baguqeeraro7i2dd4jww2v4acemnbf2623mwm67xsk5bqhau4loohgtvpaoaa",
    "LPR-019": "baguqeeraredrtw3ii37f2qremewtxbclyza6hbxponbioj2f7jarcifkdp3a",
    "LPR-020": "baguqeerar7wqiy2dgveasdr5dd2wfwkl5imlxlwtsn2xo7uucum5bdhxjmoq",
    "LPR-021": "baguqeerazwracagotvzmexqk4ht2phu6w3wxftxlmzjsvuxqwuvb3ger5gra",
    "LPR-022": "baguqeerabu4bumj3uoena3yaq3znw77idhwigvv33qva7rpaaxjofr3ridzq",
    "LPR-023": "baguqeerapjwox65apbq75pi7cxsonosqnnbgliwcbvietseecpser2d3erta",
    "LPR-024": "baguqeeraa42emzjcq6xgl5n4d6ht2znamryewqce2accgmpugtp6yeph3wqq",
    "LPR-025": "baguqeeragjgqtl4td6jakairf52m4w5mwhek7lfoaghzjftbu6nvz33fzayq",
    "LPR-026": "baguqeerawip4zmwwmq3hwracynwt5r7p3u5mqra6hpyhwjrbpu7gar4asbfq",
    "LPR-027": "baguqeera7ozs3jt43kslzau6kigegkpwqjwu6ul46dda74kgmnx6ee7zcfda",
    "LPR-028": "baguqeera45xoblimrra4eq7bt37ph5h4ue2xwxnywv24jtc5vjc3x3konzna",
    "LPR-029": "baguqeera2vr5jn3onchwhtt3iwsnlhvf5qngjd5b7izw5w4vazpa357h6bhq",
    "LPR-030": "baguqeeraly4crux6kyvmwiwfpq7nmizahsgqkpxifiuz64z2kev4okrgpm7q",
    "LPR-031": "baguqeeraoraambpjxn3yi6hmbsbpaldnowwy436xjfbbsaxhu5wsfnagivka",
    "LPR-032": "baguqeeraoypedtpd236ngorqznujj6d76jwv3r43sgb3mouljvcf6uyfkyaa",
    "LPR-033": "baguqeera3mgezpvlmrdfqzrh3q4r6mr34y4pxjash2mybksszau6o4m5e4ca",
    "LPR-034": "baguqeeram6epwdikainljlbxutsku35iz4gpe2zislcllzifmd3g5jrn2tuq",
    "LPR-035": "baguqeeraavebfpdqfsmcxq6egec5ragv6i5uhsqzvkakrgfpp54m3lyh6cxq",
    "LPR-036": "baguqeerafugs5p3jiy4ddatfxejphnmnjhywg3mbnpca7aiq56yxithdepla",
    "LPR-037": "baguqeerakgwznpdesqxlnx7v7ofi4f3aoezy6oet4vswce4wdqnnngo6zlja",
    "LPR-038": "baguqeerake5cr6xs4hzfnc65x2a667r5mwvcsn3qe6rt2n75k62zwnbyocja",
    "LPR-039": "baguqeera4ru3y6hhk7podc5yi37nonnhip6gtkittyp7ep6wm4chxm422b3q",
    "LPR-040": "baguqeera5u2slkl7fxqhaddse2r2qhgbtrapydjekmcifst2nstcj455p5ea",
    "LPR-041": "baguqeerab6cyfvu6ygnmalptp3cyt3rnez6mfg2io633ynoltu42x63ljr6q",
    "LPR-042": "baguqeera5ntdpawrkkgilqzos3hc3miad2zaimdte6s3ke4bjjz5y3bjfecq",
}
POST_BOOTSTRAP_READY = ("LPR-001", "LPR-002", "LPR-003", "LPR-004")
CONTROL_ARTIFACTS = (
    PLAN_PATH,
    OBJECTIVE_PATH,
    TODO_PATH,
    SCHEDULER_PATH,
    VALIDATOR_PATH,
    LAUNCHER_PATH,
)
BOOTSTRAP_OUTPUTS = (*CONTROL_ARTIFACTS, BOOTSTRAP_TEST_PATH)
REQUIRED_TASK_METADATA = (
    "goal id",
    "board namespace",
    "bundle",
    "parallel lane",
    "resource class",
    "resource stage",
    "token class",
    "estimated tokens",
    "implementation timeout seconds",
    "predicted files",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "acceptance",
    "embedding query",
)
ZERO_SAFETY_FLOORS = (
    "missed_resolved_impacted_consumer_rate",
    "unreconstructed_logic_or_unvalidated_countermodel_admission_rate",
    "unauthorized_premise_or_axiom_admission_rate",
    "behavior_invented_without_independent_authority_rate",
    "wrong_value_source_or_placement_admission_rate",
    "stale_root_corpus_or_receipt_admission_rate",
    "failed_obligation_override_rate",
    "llm_scope_or_semantic_escape_rate",
    "partial_transaction_completion_rate",
    "false_fixed_point_completion_rate",
    "deterministic_doctor_llm_invocation_rate",
    "graph_vector_or_embedding_authority_promotion_rate",
    "stale_poisoned_or_mismatched_doctor_cache_admission_rate",
    "forged_or_mismatched_doctor_cid_admission_rate",
    "incomplete_doctor_impact_or_open_frontier_mutation_rate",
    "unauthorized_tcb_or_path_escape_write_rate",
    "doctor_sandbox_escape_rate",
    "non_atomic_doctor_mutation_rate",
    "doctor_rollback_restoration_failure_rate",
    "nondeterministic_doctor_replay_rate",
    "false_deterministic_doctor_completion_rate",
)
NON_AUTHORITY_FLAGS = (
    "tactician_semantic_authority",
    "vector_semantic_authority",
    "embedding_semantic_authority",
    "knowledge_graph_semantic_authority",
    "learned_ranking_semantic_authority",
    "hammer_candidate_semantic_authority",
    "raw_countermodel_semantic_authority",
    "ordinary_test_semantic_authority",
    "runtime_witness_semantic_authority",
    "llm_router_semantic_authority",
    "llm_router_write_authority",
)
REQUIRED_AUTHORITY_GATES = (
    "native_kernel_reconstruction_required_for_proof",
    "independent_countermodel_validation_required_for_refutation",
    "existing_rpr_plan_lease_and_transaction_authority_required",
)
ROLLOUT_OFF_FLAGS = (
    "logic_prediction_enabled",
    "learned_tactician_ranking_enabled",
    "hammer_execution_enabled",
    "counterexample_refinement_enabled",
    "llm_router_enabled",
    "narrow_autonomous_mutation_enabled",
)


class BoardValidationError(RuntimeError):
    """Raised when a sealed control-plane invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoardValidationError(message)


def _strings(value: object, *, name: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{name} must be a JSON list")
    result = tuple(str(item).strip() for item in value)
    _require(all(result), f"{name} contains an empty value")
    _require(len(result) == len(set(result)), f"{name} contains duplicates")
    return result


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _safe_relative(path_text: str, *, field: str) -> None:
    path = PurePosixPath(path_text)
    _require(path_text == path_text.strip(), f"{field} has surrounding whitespace")
    _require(not path.is_absolute(), f"{field} must be repository-relative: {path_text}")
    _require(".." not in path.parts, f"{field} escapes the repository: {path_text}")
    _require("\x00" not in path_text, f"{field} contains NUL")


def _assert_acyclic(graph: Mapping[str, Iterable[str]], *, label: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, trail: tuple[str, ...]) -> None:
        if node in visiting:
            raise BoardValidationError(
                f"{label} cycle: {' -> '.join((*trail, node))}"
            )
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node, ())


def _load_scheduler() -> dict[str, object]:
    try:
        payload = json.loads((REPO_ROOT / SCHEDULER_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoardValidationError(f"cannot read scheduler: {exc}") from exc
    _require(isinstance(payload, dict), "scheduler root must be a JSON object")
    return payload


def _validate_control_artifacts() -> None:
    for path in BOOTSTRAP_OUTPUTS:
        _require((REPO_ROOT / path).is_file(), f"missing control artifact: {path}")


def _validate_goals() -> tuple[object, ...]:
    goals = tuple(
        parse_goal_heap((REPO_ROOT / OBJECTIVE_PATH).read_text(encoding="utf-8"))
    )
    ids = tuple(goal.goal_id for goal in goals)
    _require(len(ids) == len(set(ids)), "duplicate goal id")
    _require(set(ids) == set(EXPECTED_GOAL_IDS), f"unexpected goal ids: {sorted(ids)}")
    by_id = {goal.goal_id: goal for goal in goals}
    graph: dict[str, tuple[str, ...]] = {}
    expected_dependencies = {
        "LPR-G000": (),
        "LPR-G010": (),
        "LPR-G020": ("LPR-G010",),
        "LPR-G030": ("LPR-G010", "LPR-G020"),
        "LPR-G040": ("LPR-G030",),
        "LPR-G050": ("LPR-G040",),
        "LPR-G060": ("LPR-G050",),
        "LPR-G070": ("LPR-G060",),
        "LPR-G080": ("LPR-G060",),
        "LPR-G090": ("LPR-G080",),
        "LPR-G100": ("LPR-G090",),
        "LPR-G110": ("LPR-G070", "LPR-G100"),
    }
    for goal in goals:
        _require(re.fullmatch(r"LPR-G\d{3}", goal.goal_id) is not None, f"bad goal id: {goal.goal_id}")
        dependencies = tuple(goal.dependencies)
        unknown = sorted(set(dependencies) - set(by_id))
        _require(not unknown, f"unknown goal dependencies for {goal.goal_id}: {unknown}")
        _require(
            dependencies == expected_dependencies[goal.goal_id],
            f"goal dependency mismatch for {goal.goal_id}: {dependencies}",
        )
        graph[goal.goal_id] = dependencies
        parent = goal.fields.get("parent", "").strip()
        if goal.goal_id == "LPR-G000":
            _require(not parent, "root goal must not have a parent")
            children = _csv(goal.fields.get("subgoals", ""))
            _require(set(children) == set(EXPECTED_GOAL_IDS[1:]), "root subgoal set mismatch")
            _require(
                _csv(goal.fields.get("evidence", "")) == EXPECTED_GOAL_IDS[1:],
                "root goal evidence set/order mismatch",
            )
        else:
            _require(parent == "LPR-G000", f"{goal.goal_id} must be parented by LPR-G000")
            _require(
                _csv(goal.fields.get("evidence", ""))
                == EXPECTED_GOAL_TASK_IDS[goal.goal_id],
                f"goal evidence mismatch for {goal.goal_id}",
            )
    _assert_acyclic(graph, label="goal dependency")
    return goals


def _normalized_task_metadata(task: object) -> dict[str, str]:
    return {
        str(key).strip().lower().replace("_", " "): str(value).strip()
        for key, value in task.metadata.items()
        if str(value).strip()
    }


def _resolution_receipt_digest(receipt: Mapping[str, object]) -> str:
    """Return the content digest for a resolution receipt without its digest."""

    payload = {
        str(key): value
        for key, value in receipt.items()
        if str(key) != "receipt_digest"
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{sha256(canonical).hexdigest()}"


def _validate_reconciliation_resolution_receipt(
    *,
    repair: object,
    metadata: Mapping[str, str],
    discovery_path: PurePosixPath,
    candidate_count: int,
) -> None:
    """Require content-addressed postconditions before a guardrail completes."""

    path = Path(str(discovery_path))
    try:
        _require(
            path.stat().st_size <= 1_048_576,
            f"{repair.task_id} reconciliation discovery is unbounded",
        )
        discovery_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BoardValidationError(
            f"{repair.task_id} completed reconciliation evidence is unavailable"
        ) from exc
    matches = re.findall(
        r"^## Resolution Receipt\s*\n\s*```json\s*\n(.*?)\n```",
        discovery_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    _require(
        len(matches) == 1,
        f"{repair.task_id} must have one machine-readable resolution receipt",
    )
    try:
        receipt = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise BoardValidationError(
            f"{repair.task_id} resolution receipt is malformed"
        ) from exc
    _require(
        isinstance(receipt, dict),
        f"{repair.task_id} resolution receipt must be an object",
    )
    _require(
        receipt.get("schema") == RECONCILIATION_RESOLUTION_SCHEMA
        and receipt.get("task_id") == repair.task_id
        and receipt.get("reconciliation_fingerprint")
        == metadata.get("reconciliation fingerprint")
        and receipt.get("kind") == metadata.get("reconciliation kind")
        and receipt.get("reason") == metadata.get("reconciliation reason")
        and receipt.get("resolved") is True,
        f"{repair.task_id} resolution receipt binding mismatch",
    )
    _require(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
            str(receipt.get("resolved_at") or ""),
        )
        is not None,
        f"{repair.task_id} resolution timestamp is invalid",
    )
    _require(
        re.fullmatch(
            r"[a-z][a-z0-9_]{2,127}",
            str(receipt.get("resolution_method") or ""),
        )
        is not None,
        f"{repair.task_id} resolution method is invalid",
    )
    postconditions = receipt.get("postconditions")
    _require(
        isinstance(postconditions, dict)
        and postconditions.get("candidate_count_before") == candidate_count
        and postconditions.get("candidate_count_after") == 0
        and postconditions.get("active_blocker_present_after") is False
        and postconditions.get("dirty_worktree_group_count_after") == 0
        and postconditions.get("cleanup_skip_count_after") == 0,
        f"{repair.task_id} resolution postconditions are incomplete",
    )
    evidence = receipt.get("evidence")
    _require(
        isinstance(evidence, dict) and bool(evidence),
        f"{repair.task_id} resolution evidence is empty",
    )
    _require(
        receipt.get("receipt_digest") == _resolution_receipt_digest(receipt),
        f"{repair.task_id} resolution receipt digest mismatch",
    )


def _validate_reconciliation_guardrail_task(
    repair: object,
    *,
    metadata: Mapping[str, str],
) -> None:
    """Validate one operator-gated cleanup appendix independently of the DAG."""

    kind = metadata.get("reconciliation kind", "")
    reason = metadata.get("reconciliation reason", "")
    fingerprint = metadata.get("reconciliation fingerprint", "")
    _require(
        metadata.get("generated by") == RECONCILIATION_GUARDRAIL_SCHEMA
        and metadata.get("canonical board task") == "false",
        f"{repair.task_id} lacks explicit reconciliation provenance",
    )
    _require(
        kind in RECONCILIATION_GUARDRAIL_KINDS,
        f"{repair.task_id} reconciliation kind is unsupported",
    )
    _require(
        reason in RECONCILIATION_REASONS_BY_KIND[kind],
        f"{repair.task_id} reconciliation reason is unsupported for {kind}",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", fingerprint) is not None
        and metadata.get("fingerprint") == fingerprint,
        f"{repair.task_id} reconciliation fingerprint mismatch",
    )
    expected_dedupe = {
        "main_checkout_dirty": "reconciliation_guardrail:main_checkout_dirty",
        "preflight_merge_conflict": (
            "reconciliation_guardrail:preflight_merge_conflict"
        ),
        "dirty_backlogged_worktree": (
            f"reconciliation_guardrail:dirty_backlogged_worktree:{reason}"
        ),
    }[kind]
    _require(
        metadata.get("dedupe key") == expected_dedupe,
        f"{repair.task_id} reconciliation dedupe key mismatch",
    )
    _require(
        repair.status in {"blocked", "completed"},
        f"{repair.task_id} reconciliation status is unsafe",
    )
    _require(repair.completion == "manual", f"{repair.task_id} must be manual")
    expected_priority = (
        "P1"
        if kind != "dirty_backlogged_worktree" or reason == "unsupported_status"
        else "P2"
    )
    _require(
        repair.priority == expected_priority,
        f"{repair.task_id} reconciliation priority mismatch",
    )
    _require(repair.track == "ops", f"{repair.task_id} track mismatch")
    _require(
        metadata.get("is schedulable") == "false"
        and metadata.get("review only") == "true"
        and metadata.get("blocked reason") == "operator_reconciliation_required",
        f"{repair.task_id} reconciliation authority gate mismatch",
    )
    _require(
        not repair.depends_on,
        f"{repair.task_id} reconciliation appendix must not alter the sealed DAG",
    )
    _require(
        len(repair.outputs) == 2
        and repair.outputs[1].replace("\\", "/") == str(TODO_PATH),
        f"{repair.task_id} reconciliation output scope mismatch",
    )
    discovery_root_text = repair.outputs[0].replace("\\", "/")
    discovery_root = PurePosixPath(discovery_root_text)
    _require(
        discovery_root.name == "discovery"
        and ".." not in discovery_root.parts
        and "\x00" not in discovery_root_text,
        f"{repair.task_id} reconciliation discovery output is unsafe",
    )
    _require(
        len(repair.validation) == 1,
        f"{repair.task_id} must have one reconciliation validation",
    )
    try:
        validation = shlex.split(repair.validation[0])
    except ValueError as exc:
        raise BoardValidationError(
            f"{repair.task_id} reconciliation validation is malformed"
        ) from exc
    _require(
        len(validation) == 3 and validation[:2] == ["test", "-f"],
        f"{repair.task_id} reconciliation validation is not fail-closed",
    )
    discovery_path_text = validation[2].replace("\\", "/")
    discovery_path = PurePosixPath(discovery_path_text)
    _require(
        discovery_path.parent == discovery_root
        and ".." not in discovery_path.parts
        and "\x00" not in discovery_path_text,
        f"{repair.task_id} reconciliation validation escapes its output",
    )
    _require(
        metadata.get("reconciliation discovery", "").replace("\\", "/")
        == discovery_path_text,
        f"{repair.task_id} reconciliation discovery provenance mismatch",
    )
    expected_name = (
        rf"\d{{4}}-\d{{2}}-\d{{2}}-{repair.task_id.lower()}-"
        rf"reconciliation-{fingerprint[:12]}\.md"
    )
    _require(
        re.fullmatch(expected_name, discovery_path.name) is not None,
        f"{repair.task_id} reconciliation discovery filename mismatch",
    )
    title_patterns = {
        "main_checkout_dirty": (
            r"^Resolve dirty main checkout blocking (?P<count>[1-9]\d*) "
            r"worktree merges$"
        ),
        "preflight_merge_conflict": (
            r"^Resolve (?P<count>[1-9]\d*) preflight-conflicting "
            r"backlogged worktree merges$"
        ),
        "dirty_backlogged_worktree": (
            rf"^Resolve (?P<count>[1-9]\d*) dirty backlogged worktrees "
            rf"blocked by {re.escape(reason)}$"
        ),
    }
    title_match = re.fullmatch(title_patterns[kind], repair.title)
    _require(
        title_match is not None,
        f"{repair.task_id} reconciliation title mismatch",
    )
    candidate_count_text = title_match.group("count")
    _require(
        discovery_path_text in repair.acceptance
        and f"because {candidate_count_text} branch or worktree cleanup candidates"
        in repair.acceptance
        and f"blocked by {reason}" in repair.acceptance,
        f"{repair.task_id} reconciliation acceptance/evidence mismatch",
    )
    if repair.status == "completed":
        _validate_reconciliation_resolution_receipt(
            repair=repair,
            metadata=metadata,
            discovery_path=discovery_path,
            candidate_count=int(candidate_count_text),
        )


def _validate_operational_repair_tasks(
    repairs: Sequence[object],
    *,
    canonical_by_id: Mapping[str, object],
) -> None:
    """Validate bounded operational appendices without changing the sealed DAG."""

    _require(
        len(repairs) <= MAX_OPERATIONAL_REPAIR_TASKS,
        "operational appendix exceeds its finite bound",
    )
    expected_number = len(EXPECTED_TASK_IDS)
    previous_source_kind: dict[tuple[str, str], object] = {}
    retry_repair_count = 0
    reconciliation_count = 0
    active_reconciliation_count = 0
    reconciliation_dedupe_tasks: dict[str, object] = {}
    reconciliation_fingerprint_tasks: dict[str, object] = {}
    for offset, repair in enumerate(repairs):
        expected_id = f"LPR-{expected_number + offset:03d}"
        _require(
            repair.task_id == expected_id,
            f"operational appendix ids must be contiguous: {repair.task_id}",
        )
        metadata = _normalized_task_metadata(repair)
        if metadata.get("generated by") == RECONCILIATION_GUARDRAIL_SCHEMA:
            reconciliation_count += 1
            _require(
                reconciliation_count <= MAX_OPERATIONAL_RECONCILIATION_TASKS,
                "reconciliation appendix exceeds its finite bound",
            )
            _validate_reconciliation_guardrail_task(
                repair,
                metadata=metadata,
            )
            if repair.status != "completed":
                active_reconciliation_count += 1
                _require(
                    active_reconciliation_count
                    <= MAX_ACTIVE_OPERATIONAL_RECONCILIATION_TASKS,
                    "active reconciliation appendix exceeds its finite bound",
                )
            dedupe_key = metadata["dedupe key"]
            fingerprint = metadata["reconciliation fingerprint"]
            previous_dedupe = reconciliation_dedupe_tasks.get(dedupe_key)
            _require(
                previous_dedupe is None or previous_dedupe.status == "completed",
                (
                    "concurrent duplicate operational reconciliation task: "
                    f"{dedupe_key}"
                ),
            )
            previous_fingerprint = reconciliation_fingerprint_tasks.get(
                fingerprint
            )
            _require(
                previous_fingerprint is None
                or previous_fingerprint.status == "completed",
                (
                    "concurrent duplicate operational reconciliation "
                    f"fingerprint: {fingerprint}"
                ),
            )
            reconciliation_dedupe_tasks[dedupe_key] = repair
            reconciliation_fingerprint_tasks[fingerprint] = repair
            continue
        retry_repair_count += 1
        _require(
            retry_repair_count <= MAX_OPERATIONAL_RETRY_REPAIR_TASKS,
            "retry-repair appendix exceeds its finite bound",
        )
        source_task_id, failure_kind = retry_budget_repair_source(repair)
        _require(
            source_task_id in canonical_by_id
            and failure_kind in {"validation", "implementation", "merge"},
            f"unrecognized operational retry-repair task: {repair.task_id}",
        )
        source = canonical_by_id[source_task_id]
        source_kind = (source_task_id, failure_kind)
        previous = previous_source_kind.get(source_kind)
        _require(
            previous is None or previous.status == "completed",
            f"concurrent duplicate operational retry-repair task: {source_kind}",
        )
        previous_source_kind[source_kind] = repair
        if repair.task_id not in LEGACY_OPERATIONAL_REPAIR_TASK_IDS:
            _require(
                metadata.get("generated by") == RETRY_BUDGET_REPAIR_SCHEMA
                and metadata.get("retry repair source") == source_task_id
                and metadata.get("retry failure kind") == failure_kind
                and metadata.get("canonical board task") == "false",
                f"{repair.task_id} lacks explicit retry-repair provenance",
            )
        _require(repair.completion == "manual", f"{repair.task_id} must be manual")
        _require(repair.priority == "P1", f"{repair.task_id} priority mismatch")
        _require(repair.track == "ops", f"{repair.task_id} track mismatch")
        _require(
            tuple(repair.depends_on) == tuple(source.depends_on),
            f"{repair.task_id} dependency scope differs from {source_task_id}",
        )
        source_outputs = tuple(source.outputs)
        repair_outputs = tuple(repair.outputs)
        _require(
            repair_outputs[: len(source_outputs)] == source_outputs
            and len(repair_outputs) == len(source_outputs) + 1,
            f"{repair.task_id} output scope is not the source scope plus discovery",
        )
        discovery_root_text = repair_outputs[-1].replace("\\", "/")
        discovery_root = PurePosixPath(discovery_root_text)
        _require(
            discovery_root.name == "discovery"
            and ".." not in discovery_root.parts
            and "\x00" not in discovery_root_text,
            f"{repair.task_id} discovery output is unsafe",
        )
        _require(
            len(repair.validation) == 1,
            f"{repair.task_id} must have one discovery validation",
        )
        try:
            validation = shlex.split(repair.validation[0])
        except ValueError as exc:
            raise BoardValidationError(
                f"{repair.task_id} discovery validation is malformed"
            ) from exc
        _require(
            len(validation) == 3 and validation[:2] == ["test", "-f"],
            f"{repair.task_id} discovery validation is not fail-closed",
        )
        discovery_path_text = validation[2].replace("\\", "/")
        discovery_path = PurePosixPath(discovery_path_text)
        _require(
            discovery_path.parent == discovery_root
            and ".." not in discovery_path.parts
            and "\x00" not in discovery_path_text,
            f"{repair.task_id} discovery validation escapes its output",
        )
        suffix = {
            "validation": "retry-budget",
            "implementation": "implementation-retry-budget",
            "merge": "merge-retry-budget",
        }[failure_kind]
        expected_name = (
            rf"\d{{4}}-\d{{2}}-\d{{2}}-{repair.task_id.lower()}-"
            rf"{source_task_id.lower()}-{suffix}\.md"
        )
        _require(
            re.fullmatch(expected_name, discovery_path.name) is not None,
            f"{repair.task_id} discovery filename mismatch",
        )
        _require(
            discovery_path_text in repair.acceptance,
            f"{repair.task_id} acceptance omits its discovery evidence",
        )


def _validate_tasks(
    goal_ids: set[str],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    all_tasks = tuple(
        parse_task_file(REPO_ROOT / TODO_PATH, task_header_prefix=TASK_PREFIX)
    )
    ids = tuple(task.task_id for task in all_tasks)
    _require(len(ids) == len(set(ids)), "duplicate task id")
    tasks = tuple(task for task in all_tasks if task.task_id in EXPECTED_TASK_IDS)
    repairs = tuple(task for task in all_tasks if task.task_id not in EXPECTED_TASK_IDS)
    canonical_ids = tuple(sorted(task.task_id for task in tasks))
    _require(
        canonical_ids == EXPECTED_TASK_IDS,
        f"canonical task ids changed: {list(canonical_ids)}",
    )
    _require(
        set(SEALED_TASK_CIDS) == set(EXPECTED_TASK_IDS),
        "sealed task CID map does not cover the exact task set",
    )
    by_id = {task.task_id: task for task in tasks}
    for task_id, expected_cid in SEALED_TASK_CIDS.items():
        _require(
            by_id[task_id].canonical_task_cid == expected_cid,
            f"sealed task identity changed: {task_id}",
        )
    _validate_operational_repair_tasks(repairs, canonical_by_id=by_id)
    graph: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        _require(re.fullmatch(r"LPR-\d{3}", task.task_id) is not None, f"bad task id: {task.task_id}")
        unknown = sorted(set(task.depends_on) - set(by_id))
        _require(not unknown, f"unknown dependencies for {task.task_id}: {unknown}")
        graph[task.task_id] = tuple(task.depends_on)
        missing = [name for name in REQUIRED_TASK_METADATA if not task.metadata.get(name, "").strip()]
        _require(not missing, f"{task.task_id} missing metadata: {missing}")
        _require(task.metadata["goal id"] in goal_ids, f"{task.task_id} has unknown goal")
        _require(task.board_namespace == BOARD_NAMESPACE, f"{task.task_id} namespace mismatch")
        _require(task.completion in {"auto", "manual"}, f"{task.task_id} completion mismatch")
        _require(
            bool(task.validation)
            and all(str(command).strip() for command in task.validation),
            f"{task.task_id} has no validation command",
        )
        _require(task.acceptance.strip(), f"{task.task_id} has no acceptance criteria")
        for output in task.outputs:
            _safe_relative(output, field=f"{task.task_id} output")
        for predicted in _csv(task.metadata["predicted files"]):
            _safe_relative(predicted, field=f"{task.task_id} predicted file")
        try:
            estimated = int(task.metadata["estimated tokens"])
            timeout = int(task.metadata["implementation timeout seconds"])
        except ValueError as exc:
            raise BoardValidationError(f"{task.task_id} has a non-integer bound") from exc
        _require(0 < estimated <= 100_000, f"{task.task_id} token bound is unsafe")
        _require(0 < timeout <= 14_400, f"{task.task_id} timeout bound is unsafe")
    for goal_id, expected_task_ids in EXPECTED_GOAL_TASK_IDS.items():
        observed_task_ids = tuple(
            task.task_id
            for task in tasks
            if task.metadata["goal id"] == goal_id
        )
        _require(
            observed_task_ids == expected_task_ids,
            f"task projection mismatch for {goal_id}: {observed_task_ids}",
        )
    _assert_acyclic(graph, label="task dependency")
    roots = sorted(task_id for task_id, dependencies in graph.items() if not dependencies)
    _require(roots == ["LPR-000"], f"task roots mismatch: {roots}")
    expected_tail = {
        "LPR-020": ("LPR-019",),
        "LPR-021": ("LPR-020",),
        "LPR-022": ("LPR-021",),
        "LPR-023": ("LPR-022",),
        "LPR-024": ("LPR-022",),
        "LPR-025": ("LPR-021",),
        "LPR-026": ("LPR-022",),
        "LPR-027": ("LPR-023", "LPR-024", "LPR-025", "LPR-026"),
        "LPR-028": ("LPR-027",),
        "LPR-029": ("LPR-020",),
        "LPR-030": ("LPR-020",),
        "LPR-031": ("LPR-020",),
        "LPR-032": ("LPR-020",),
        "LPR-033": ("LPR-029", "LPR-030"),
        "LPR-034": ("LPR-029", "LPR-030", "LPR-031"),
        "LPR-035": ("LPR-032", "LPR-034"),
        "LPR-036": ("LPR-033", "LPR-035"),
        "LPR-037": ("LPR-030", "LPR-036"),
        "LPR-038": ("LPR-032", "LPR-037"),
        "LPR-039": ("LPR-038",),
        "LPR-040": ("LPR-030", "LPR-031", "LPR-032", "LPR-039"),
        "LPR-041": ("LPR-040",),
        "LPR-042": ("LPR-028", "LPR-041"),
    }
    for task_id, dependencies in expected_tail.items():
        _require(
            graph[task_id] == dependencies,
            f"{task_id} dependency mismatch: {graph[task_id]}",
        )
    consumed_dependencies = {
        dependency for dependencies in graph.values() for dependency in dependencies
    }
    sinks = sorted(set(graph) - consumed_dependencies)
    _require(sinks == ["LPR-042"], f"terminal task mismatch: {sinks}")

    simulated_completed = {"LPR-000"}
    ready = tuple(
        sorted(
            task.task_id
            for task in tasks
            if task.task_id != "LPR-000"
            and set(task.depends_on).issubset(simulated_completed)
        )
    )
    _require(ready == POST_BOOTSTRAP_READY, f"post-bootstrap ready set mismatch: {ready}")
    bootstrap = by_id["LPR-000"]
    _require(
        tuple(bootstrap.outputs) == tuple(str(path) for path in BOOTSTRAP_OUTPUTS),
        "LPR-000 bootstrap output list mismatch",
    )

    foundations = [by_id[task_id] for task_id in POST_BOOTSTRAP_READY]
    owned: list[tuple[PurePosixPath, str]] = []
    for task in foundations:
        for predicted in _csv(task.metadata["predicted files"]):
            path = PurePosixPath(predicted)
            for other, owner in owned:
                overlaps = path == other or path in other.parents or other in path.parents
                _require(
                    not overlaps or owner == task.task_id,
                    f"foundation path conflict: {predicted} ({owner}, {task.task_id})",
                )
            owned.append((path, task.task_id))
    _require(
        bootstrap.status == "completed",
        "LPR-000 must be completed before the sealed board is launched",
    )
    return tasks, repairs


def _validate_scheduler(scheduler: Mapping[str, object], tasks: Sequence[object]) -> None:
    expected_scalars = {
        "schema": "ipfs_accelerate_py.agent_supervisor.tactician_hammer_logic_repair.scheduler_config@1",
        "taskboard_path": str(TODO_PATH),
        "objectives_path": str(OBJECTIVE_PATH),
        "plan_path": str(PLAN_PATH),
        "validator_path": str(VALIDATOR_PATH),
        "launcher_path": str(LAUNCHER_PATH),
        "task_prefix": TASK_PREFIX,
        "board_namespace": BOARD_NAMESPACE,
        "merge_target_branch": TARGET_BRANCH,
        "max_lanes": 4,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
    }
    for key, expected in expected_scalars.items():
        _require(scheduler.get(key) == expected, f"scheduler {key} mismatch")
    for key in (
        "poll_interval_seconds",
        "daemon_interval_seconds",
        "check_interval_seconds",
        "stale_seconds",
        "watchdog_startup_grace_seconds",
        "max_restarts",
        "max_task_attempts",
        "implementation_retry_budget",
        "validation_retry_budget",
        "merge_retry_budget",
        "implementation_timeout_seconds",
        "implementation_max_timeout_seconds",
        "implementation_log_stall_seconds",
    ):
        value = scheduler.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"scheduler {key} must be positive")
    _require(scheduler["implementation_max_timeout_seconds"] >= scheduler["implementation_timeout_seconds"], "max timeout is below default timeout")
    _require(_strings(scheduler.get("worktree_submodule_paths"), name="worktree_submodule_paths") == ("ipfs_datasets_py",), "datasets gitlink binding missing")
    protected = _strings(scheduler.get("protected_paths"), name="protected_paths")
    _require(protected == tuple(str(path) for path in CONTROL_ARTIFACTS), "protected control artifacts mismatch")
    for path in protected:
        _safe_relative(path, field="protected path")

    source = scheduler.get("source_binding")
    _require(isinstance(source, dict), "source_binding must be an object")
    for key in (
        "require_exact_accelerator_branch",
        "require_initialized_datasets_gitlink",
        "require_superproject_gitlink_equals_nested_head",
        "record_accelerator_and_datasets_revisions_at_launch",
    ):
        _require(source.get(key) is True, f"source binding disabled: {key}")
    _require(source.get("accelerator_branch") == TARGET_BRANCH, "source branch binding mismatch")
    _require(source.get("datasets_submodule_path") == "ipfs_datasets_py", "datasets source path mismatch")
    _require(
        source.get("datasets_required_ancestor") == DATASETS_TACTICIAN_ANCESTOR,
        "datasets Tactician ancestor binding mismatch",
    )
    _require(
        source.get("datasets_required_interface") == DATASETS_TACTICIAN_INTERFACE,
        "datasets Tactician interface binding mismatch",
    )
    required_datasets_paths = _strings(
        source.get("datasets_required_paths"), name="datasets_required_paths"
    )
    _require(
        required_datasets_paths == DATASETS_REQUIRED_PATHS,
        "datasets deterministic-doctor logic required paths mismatch",
    )
    for path in required_datasets_paths:
        _safe_relative(path, field="datasets required path")

    refactor_sources = scheduler.get("refactor_source_bindings")
    _require(isinstance(refactor_sources, dict), "refactor_source_bindings must be an object")
    vfs_source = refactor_sources.get("ipfs_kit_vfs_assurance")
    _require(isinstance(vfs_source, dict), "VFS generalization source binding is missing")
    _require(
        vfs_source.get("repository_url")
        == "https://github.com/endomorphosis/ipfs_accelerate_py.git",
        "VFS source repository mismatch",
    )
    _require(
        vfs_source.get("revision")
        == "0cc04ebb640c4c981cf4650016e096a73ab0e8c0",
        "VFS source revision mismatch",
    )
    _require(
        vfs_source.get("local_ref")
        == "refs/agent-supervisor/source-locks/vfs-generalization/0cc04ebb640c4c981cf4650016e096a73ab0e8c0",
        "VFS source-lock ref mismatch",
    )
    _require(
        vfs_source.get("merge_or_cherry_pick_source_revision") is False,
        "broad VFS source merge must be forbidden",
    )
    expected_vfs_blobs = {
        "ipfs_accelerate_py/agent_supervisor/vfs_contract_pack.py": "9acc4ceba42b8767f5b4e4b6ce7d4bc55893bcf2",
        "ipfs_accelerate_py/agent_supervisor/vfs_differential_harness.py": "8a6c8af69b6cbcb76a2b79a51f406d13e10947ce",
        "ipfs_accelerate_py/agent_supervisor/vfs_mcp_contract_checker.py": "26144a7b78c1bbbb94edc67ab13e2eab03850924",
        "ipfs_accelerate_py/agent_supervisor/vfs_surface_inventory.py": "76f34e1b9320e4bbc15706e4895c02af805af5e0",
        "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_benchmark.py": "90023a09e9eb01ee454718f60fe758e33434c56b",
        "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_pilot.py": "483ecaf622caa3c91d80d9710b63b1fd36fb8f90",
        "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_rollout.py": "6a1ef7b87172aa413f81b37f0ba36954af774d40",
    }
    _require(
        vfs_source.get("module_blobs") == expected_vfs_blobs,
        "VFS source blob lock mismatch",
    )

    lane_rows = scheduler.get("lanes")
    _require(isinstance(lane_rows, list) and len(lane_rows) == 4, "scheduler must define four lanes")
    expected_initial = {0: ["LPR-004"], 1: ["LPR-001"], 2: ["LPR-002"], 3: ["LPR-003"]}
    observed_initial: dict[int, object] = {}
    for row in lane_rows:
        _require(isinstance(row, dict), "lane row must be an object")
        index = row.get("index")
        _require(isinstance(index, int) and not isinstance(index, bool), "lane index must be an integer")
        _require(index in range(4) and index not in observed_initial, f"invalid or duplicate lane index: {index}")
        _require(row.get("name") == f"lpr-lane-{index}", f"lane {index} name mismatch")
        _require(row.get("strict_shard_remainder") == index, f"lane {index} shard mismatch")
        observed_initial[index] = row.get("initial_task_ids")
    _require(observed_initial == expected_initial, f"initial lane assignment mismatch: {observed_initial}")
    for index, task_ids in observed_initial.items():
        for task_id in task_ids:
            _require(int(task_id.rsplit("-", 1)[1]) % 4 == index, f"{task_id} does not map to lane {index}")

    provider = scheduler.get("provider")
    _require(isinstance(provider, dict), "provider must be an object")
    _require(provider.get("max_concurrency") == 4, "provider concurrency must equal lane count")
    _require(provider.get("secrets_in_argv_or_logs") is False, "secrets must not enter argv/logs")

    rollout = scheduler.get("rollout")
    _require(isinstance(rollout, dict), "rollout must be an object")
    _require(rollout.get("mode") == "shadow", "initial rollout must be shadow")
    for key in ROLLOUT_OFF_FLAGS:
        _require(rollout.get(key) is False, f"initial feature flag must be off: {key}")

    authority = scheduler.get("authority_policy")
    _require(isinstance(authority, dict), "authority_policy must be an object")
    for key in NON_AUTHORITY_FLAGS:
        _require(authority.get(key) is False, f"advisory source promoted to authority: {key}")
    for key in REQUIRED_AUTHORITY_GATES:
        _require(authority.get(key) is True, f"authority gate disabled: {key}")
    _require(authority.get("unknown_or_unsupported_disposition") == "abstain", "unknown/unsupported work must abstain")

    repair = scheduler.get("repair_policy")
    _require(isinstance(repair, dict), "repair_policy must be an object")
    for key in (
        "impact_closure_required_before_mutation",
        "one_disposition_per_resolved_consumer",
        "logic_goal_and_premise_independence_required",
        "tactician_plan_gate_required",
        "native_goal_round_trip_required",
        "analytical_transform_precedes_llm_router",
        "llm_router_requires_admitted_semantics_and_exact_paths",
        "proposal_overlay_analysis_required_for_ordinary_model_edits",
        "atomic_scc_transaction_required",
        "logic_and_program_fixed_point_required",
    ):
        _require(repair.get(key) is True, f"repair gate disabled: {key}")
    _require(repair.get("partial_plan_completion_allowed") is False, "partial completion must be forbidden")
    _require(repair.get("open_required_frontier_disposition") == "abstain", "open required frontier must abstain")
    _require(repair.get("memory_resource_or_type_evidence_implies_memory_safety") is False, "memory safety must not be inferred")

    doctor = scheduler.get("deterministic_doctor_policy")
    _require(isinstance(doctor, dict), "deterministic_doctor_policy must be an object")
    _require(
        doctor.get("schema")
        == "ipfs_accelerate_py.agent_supervisor.deterministic_doctor.policy@1",
        "deterministic doctor policy schema mismatch",
    )
    _require(doctor.get("default_mode") == "report_only", "doctor must default to report-only")
    _require(
        _strings(doctor.get("allowed_modes"), name="deterministic_doctor_policy.allowed_modes")
        == ("report_only", "plan", "sandbox_auto", "narrow_auto"),
        "deterministic doctor mode set/order mismatch",
    )
    for key in (
        "enabled",
        "narrow_autonomous_mutation_enabled",
        "llm_router_enabled",
        "llm_invocations_allowed",
        "remote_model_provider_calls_allowed",
        "remote_embeddings_allowed",
        "network_access_allowed",
        "target_code_import_allowed",
        "knowledge_graph_semantic_authority",
        "vector_semantic_authority",
        "embedding_semantic_authority",
        "tactician_semantic_authority",
        "hammer_candidate_semantic_authority",
        "proof_cache_metadata_semantic_authority",
    ):
        _require(doctor.get(key) is False, f"deterministic doctor safety flag must be false: {key}")
    for key in (
        "explicit_repair_operation_required",
        "exact_evidence_snapshot_required",
        "clean_rebuild_identity_equivalence_required",
        "canonical_cid_preimage_validation_required",
        "proof_cache_binding_revalidation_required",
        "native_kernel_reconstruction_required",
        "independent_countermodel_validation_required",
        "complete_impact_closure_required",
        "one_disposition_per_resolved_consumer",
        "unique_target_value_placement_operator_required",
        "closed_operator_registry_required",
        "isolated_candidate_worktree_required",
        "enforced_sandbox_required_for_target_execution",
        "writer_lease_and_checkpoint_required",
        "atomic_scc_transaction_required",
        "post_edit_reindex_and_cache_invalidation_required",
        "logic_and_program_fixed_point_required",
        "compensating_rollback_required",
    ):
        _require(doctor.get(key) is True, f"deterministic doctor gate disabled: {key}")
    _require(doctor.get("unknown_or_unsupported_disposition") == "abstain", "doctor unknown work must abstain")
    _require(doctor.get("ambiguous_disposition") == "abstain", "doctor ambiguous work must abstain")
    doctor_approval = _strings(
        doctor.get("approval_required_classes"),
        name="deterministic_doctor_policy.approval_required_classes",
    )
    _require(
        set(doctor_approval)
        == {
            "doctor_trusted_computing_base",
            "stateful_behavior",
            "public_api_or_schema",
            "dynamic_or_generated_code",
            "native_or_ffi",
            "cross_repository_edit",
            "new_external_dependency",
            "unsupported_memory_or_lifetime_claim",
        },
        "deterministic doctor approval classes mismatch",
    )
    limits = doctor.get("limits")
    _require(isinstance(limits, dict), "deterministic doctor limits must be an object")
    expected_limit_keys = {
        "max_findings",
        "max_candidates_per_finding",
        "max_graph_nodes_per_query",
        "max_proof_routes_per_goal",
        "max_operators_per_finding",
        "max_plan_steps",
        "max_fixed_point_iterations",
        "max_changed_files",
        "max_changed_bytes",
        "max_processes",
        "max_wall_time_seconds",
        "max_cpu_time_seconds",
        "max_memory_bytes",
    }
    _require(set(limits) == expected_limit_keys, "deterministic doctor limit set mismatch")
    for key, value in limits.items():
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"deterministic doctor limit must be a positive integer: {key}",
        )

    floors = scheduler.get("release_safety_floors")
    _require(isinstance(floors, dict), "release_safety_floors must be an object")
    _require(set(floors) == set(ZERO_SAFETY_FLOORS), "release safety floor set mismatch")
    for key in ZERO_SAFETY_FLOORS:
        value = floors.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value == 0, f"release safety floor must be integer zero: {key}")

    hints = scheduler.get("resource_hints")
    _require(isinstance(hints, dict), "resource_hints must be an object")
    lanes = {task.metadata["parallel lane"] for task in tasks}
    _require(lanes.issubset(hints), f"missing resource hints: {sorted(lanes - set(hints))}")
    for task in tasks:
        lane = task.metadata["parallel lane"]
        _require(
            hints.get(lane) == task.metadata["resource class"],
            f"resource hint mismatch for {task.task_id}: {lane}",
        )


def _validate_authority_language(scheduler: Mapping[str, object]) -> None:
    text = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH)
    ).lower()
    required = (
        "semantic_authority=false",
        "shadow mode remains the default",
        "independently validated countermodel",
        "analytical",
        "fixed point",
        "report-only",
        "never falls back",
        "zero-llm/model-provider-call invariant",
    )
    for phrase in required:
        _require(phrase in text, f"normative authority phrase is missing: {phrase}")
    encoded = json.dumps(scheduler, sort_keys=True).lower()
    for secret_word in ("api_key", "access_token", "bearer_token", "password"):
        _require(secret_word not in encoded, f"scheduler must not contain secret field: {secret_word}")


def _validate_predecessor() -> None:
    path = REPO_ROOT / RPR_TODO_PATH
    _require(path.is_file(), "completed RPR predecessor board is missing")
    tasks = parse_task_file(path, task_header_prefix="RPR-")
    _require(len(tasks) == 48, f"RPR predecessor task count mismatch: {len(tasks)}")
    incomplete = [task.task_id for task in tasks if task.status != "completed"]
    _require(not incomplete, f"RPR predecessor is incomplete: {incomplete}")


def validate_all() -> dict[str, object]:
    _validate_control_artifacts()
    goals = _validate_goals()
    tasks, repairs = _validate_tasks({goal.goal_id for goal in goals})
    scheduler = _load_scheduler()
    _validate_scheduler(scheduler, tasks)
    _validate_authority_language(scheduler)
    _validate_predecessor()
    completed = sorted(task.task_id for task in tasks if task.status == "completed")
    ready = sorted(
        task.task_id
        for task in tasks
        if task.status == "todo" and set(task.depends_on).issubset(completed)
    )
    completed_repairs = sorted(
        task.task_id for task in repairs if task.status == "completed"
    )
    reconciliation_repairs = tuple(
        task
        for task in repairs
        if _normalized_task_metadata(task).get("generated by")
        == RECONCILIATION_GUARDRAIL_SCHEMA
    )
    retry_repairs = tuple(
        task for task in repairs if task not in reconciliation_repairs
    )
    ready_repairs = sorted(
        task.task_id
        for task in repairs
        if task.status == "todo"
        and set(task.depends_on).issubset(completed)
    )
    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.tactician_hammer_logic_repair.board_validation@1",
        "valid": True,
        "goal_count": len(goals),
        "task_count": len(tasks),
        "completed_count": len(completed),
        "ready_task_ids": ready,
        "operational_repair_task_count": len(repairs),
        "completed_operational_repair_count": len(completed_repairs),
        "operational_repair_task_ids": [task.task_id for task in repairs],
        "ready_operational_repair_task_ids": ready_repairs,
        "operational_retry_repair_task_count": len(retry_repairs),
        "operational_reconciliation_task_count": len(
            reconciliation_repairs
        ),
        "completed_operational_reconciliation_count": sum(
            task.status == "completed" for task in reconciliation_repairs
        ),
        "total_task_count": len(tasks) + len(repairs),
        "post_bootstrap_ready_task_ids": list(POST_BOOTSTRAP_READY),
        "lane_count": 4,
        "rollout_mode": "shadow",
        "protected_artifact_count": len(CONTROL_ARTIFACTS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="validate the complete goal/task/scheduler control plane",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.check_all:
        parser.error("--check-all is required")
    try:
        payload = validate_all()
    except BoardValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
