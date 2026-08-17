LGSWF_CONFLICT_SCOPES = (
    "predicted_path",
    "exact_symbol",
    "interface",
    "schema",
    "state",
    "effect",
    "generated",
    "fixture",
    "taskboard",
    "database",
    "merge",
    "external",
    "exclusive_resource",
    "opaque_file",
    "opaque_repository",
)


class SemanticConflictError(ValueError):
    """LGSWF conflict-graph extension rejected an unsafe admission."""


def evaluate_semantic_conflict(left, right):
    """Extend the existing graph with typed semantic conflict scopes."""

    left_mode = str(left.get("mode") or "write")
    right_mode = str(right.get("mode") or "write")
    if left_mode == "read" and right_mode == "read":
        return {
            "conflict": False,
            "reason": "compatible-readers",
            "scope": "shared-read",
        }
    if left_mode == "read" or right_mode == "read":
        if left.get("exclusive_resource") and left.get("exclusive_resource") == right.get(
            "exclusive_resource"
        ):
            return {
                "conflict": True,
                "reason": "exclusive-resource",
                "scope": "exclusive_resource",
            }
        return {
            "conflict": False,
            "reason": "compatible-reader-writer",
            "scope": "shared-read",
        }
    for key in ("symbol", "interface", "schema", "generated", "effect", "exclusive_resource"):
        if left.get(key) and left.get(key) == right.get(key):
            return {
                "conflict": True,
                "reason": f"same-{key}",
                "scope": "exact_symbol" if key == "symbol" else key,
            }
    if left.get("opaque") or right.get("opaque"):
        scope = "opaque_file" if left.get("path") or right.get("path") else "opaque_repository"
        return {
            "conflict": True,
            "reason": "opaque-conservative-fallback",
            "scope": scope,
        }
    if left.get("path") and left.get("path") == right.get("path"):
        return {"conflict": True, "reason": "same-path", "scope": "predicted_path"}
    return {"conflict": False, "reason": "disjoint-writes", "scope": "none"}


def admit_conflict_free_frontier(tasks):
    admitted = []
    rejected = []
    chosen = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        hit = None
        for other in chosen:
            decision = evaluate_semantic_conflict(task, other)
            if decision["conflict"]:
                hit = {
                    "task_id": task_id,
                    "conflicts_with": str(other.get("task_id") or ""),
                    "reason": str(decision["reason"]),
                }
                break
        if hit:
            rejected.append(hit)
        else:
            chosen.append(task)
            admitted.append(task_id)
    return {
        "admitted": tuple(admitted),
        "rejected": tuple(rejected),
        "deterministic": True,
    }
