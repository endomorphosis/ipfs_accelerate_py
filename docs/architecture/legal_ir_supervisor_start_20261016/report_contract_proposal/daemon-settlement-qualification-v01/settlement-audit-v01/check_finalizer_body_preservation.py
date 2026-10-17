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


def compare(original_bytes, proposed_bytes):
    require(hashlib.sha256(original_bytes).hexdigest() == ORIGINAL_SHA256,
            "exact_original_daemon_bytes_required")
    original = ast.parse(original_bytes)
    proposed = ast.parse(proposed_bytes)
    evidence = []
    for stage, contract in CONTRACTS.items():
        old_method = method(original, contract["method"])
        new_method = method(proposed, contract["method"])
        original_finalizers = [node.finalbody for node in ast.walk(old_method)
                               if isinstance(node, ast.Try) and node.finalbody
                               and contains_call(node.finalbody, contract["marker"])]
        require(len(original_finalizers) == 1, stage + ":exact_original_finalizer_required")
        expected = canonical(original_finalizers[0])
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
