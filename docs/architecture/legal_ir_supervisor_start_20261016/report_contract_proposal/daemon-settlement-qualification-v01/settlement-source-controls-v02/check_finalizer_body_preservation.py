"""Read-only AST comparison; no target import, connection, provider or Git.

This is authored verification source, not evidence of a completed check.
Run it only inside the reviewed finite native qualification envelope.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "23addb0dfc23e2cb093e7f74abbb7dc84a5308c24259e1eb0e25d721c0a80bed"
CONTRACTS = {
    "implementation_claim_release": {
        "method": "_run_implementation",
        "marker": "_release_implementation_resource_claims",
        "allowed_nonlocal": {"acquired_resource_claims"},
    },
    "pass_dispatch_intent_reconciliation": {
        "method": "_run_once",
        "marker": "_reconcile_unselected_implementation_dispatch_intents",
        "allowed_nonlocal": set(),
    },
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def method(tree, name):
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "PortalImplementationDaemon"]
    require(len(classes) == 1, "exact_original_daemon_class_required")
    found = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
             and node.name == name]
    require(len(found) == 1, "exact_original_method_required")
    return found[0]


def contains_call(nodes, attribute):
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == attribute
               for statement in nodes for node in ast.walk(statement))


def is_original_finalizer(body, stage):
    """Select the exact producer shape, excluding early prompt cleanup."""
    def direct_self_call(statement, name):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return False
        call = statement.value
        return (isinstance(call.func, ast.Attribute) and call.func.attr == name
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "self")
    if stage == "implementation_claim_release":
        return (len(body) == 4 and isinstance(body[0], ast.Try)
                and direct_self_call(body[1], "_release_implementation_resource_claims")
                and len(body[1].value.args) == 1 and not body[1].value.keywords
                and isinstance(body[1].value.args[0], ast.Name)
                and body[1].value.args[0].id == "acquired_resource_claims"
                and isinstance(body[2], ast.Assign) and len(body[2].targets) == 1
                and isinstance(body[2].targets[0], ast.Name)
                and body[2].targets[0].id == "acquired_resource_claims"
                and isinstance(body[2].value, ast.List) and not body[2].value.elts
                and isinstance(body[3], ast.Try)
                and contains_call([body[0]], "_release_implementation_lock")
                and contains_call([body[3]], "_release_implementation_task_claim"))
    if stage == "pass_dispatch_intent_reconciliation":
        return (len(body) == 2 and isinstance(body[0], ast.Assign)
                and len(body[0].targets) == 1 and isinstance(body[0].targets[0], ast.Name)
                and body[0].targets[0].id == "retain_selected_intent"
                and direct_self_call(body[1], "_reconcile_unselected_implementation_dispatch_intents"))
    return False


def select_original_finalizer(original_method, stage):
    candidates = [node.finalbody for node in ast.walk(original_method)
                  if isinstance(node, ast.Try) and node.finalbody
                  and is_original_finalizer(node.finalbody, stage)]
    require(len(candidates) == 1, stage + ":exact_original_finalizer_required")
    return candidates[0]


def compare(original_bytes, proposed_bytes):
    require(hashlib.sha256(original_bytes).hexdigest() == ORIGINAL_SHA256,
            "exact_original_daemon_bytes_required")
    original = ast.parse(original_bytes)
    proposed = ast.parse(proposed_bytes)
    evidence = []
    for stage, contract in CONTRACTS.items():
        old_method = method(original, contract["method"])
        new_method = method(proposed, contract["method"])
        original_finalizer = select_original_finalizer(old_method, stage)
        expected = canonical(original_finalizer)
        matches = []
        for node in ast.walk(new_method):
            if not isinstance(node, ast.FunctionDef) or node is new_method:
                continue
            body = list(node.body)
            declarations = []
            while body and isinstance(body[0], ast.Nonlocal):
                declarations.extend(body.pop(0).names)
            if set(declarations) - contract["allowed_nonlocal"]:
                continue
            if canonical(body) == expected:
                require(not node.decorator_list and not node.args.args
                        and not node.args.posonlyargs and not node.args.kwonlyargs
                        and node.args.vararg is None and node.args.kwarg is None
                        and not node.args.defaults and not node.args.kw_defaults,
                        stage + ":effect_free_zero_argument_source_closure_required")
                matches.append(node)
        require(len(matches) == 1, stage + ":exactly_one_unchanged_original_body_closure_required")
        evidence.append({
            "stage": stage,
            "original_method": contract["method"],
            "closure": matches[0].name,
            "closure_line": matches[0].lineno,
            "original_body_ast_sha256": hashlib.sha256(expected.encode()).hexdigest(),
            "unchanged_original_body": True,
        })
    return {
        "schema": "legal-ir-finalizer-body-preservation@1",
        "original_sha256": ORIGINAL_SHA256,
        "proposed_sha256": hashlib.sha256(proposed_bytes).hexdigest(),
        "checks": evidence,
        "runtime_qualification": False,
        "disposition_settled": False,
        "acceptance_authority": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("proposed", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(compare(arguments.original.read_bytes(), arguments.proposed.read_bytes()),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
