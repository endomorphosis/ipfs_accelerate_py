"""Closed repair of a direct call that ignores an explicit local import alias.

An existing ``from module import name as alias`` declaration supplies the exact
binding. No dependency, import, argument, or behavior is invented. The reviewed
claim is alias lookup and argument preservation, not equivalence to the broken
call or whole-program correctness. All other source shapes remain unsupported.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import sys

from ..proof.formal_verification_contracts import content_identity

OPERATOR = "closed-imported-alias-call@1"
PROOF_SCOPE = (
    "Under closed local-module resolution, the existing explicit import alias resolves to its signed donor export; "
    "changing only the direct callee identifier preserves the argument expressions "
    "and order, and the projected supplied parameters satisfy the declared signature. "
    "Independent AST/signature replay binds the edit under the closed "
    "local-module import assumptions. Not whole-program correctness or equivalence "
    "to the original unresolved call."
)
MAX_IDENTIFIER_CHARACTERS = 64
MAX_MODULE_BINDINGS = 16
_IDENTIFIER = re.compile(rf"[A-Za-z_][A-Za-z0-9_]{{0,{MAX_IDENTIFIER_CHARACTERS - 1}}}")
_PURE_EXPRESSIONS = (ast.Name, ast.Constant, ast.Load)
_RETURN_EXPRESSIONS = (*_PURE_EXPRESSIONS, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.IfExp, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)
MAX_PARAMETERS = 32
_RESERVED_MODULES = frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names)
_IMPLICIT_GLOBALS = frozenset({"__name__", "__doc__", "__package__", "__loader__", "__spec__",
                               "__builtins__", "__file__", "__cached__", "__annotations__"})


class ImportedAliasContractError(ValueError):
    """The explicit alias, local source population, or exact call is unsupported."""


def _require(condition):
    if not condition:
        raise ImportedAliasContractError("unsupported or inconsistent closed local import-alias contract")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_current_source(path, digest, *, maximum, expected_size=None):
    _require(path.resolve(strict=True) == path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum
                 and (expected_size is None or before.st_size == expected_size))
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(before.st_size + 1)
        after = os.fstat(fd)
        _require(len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == digest
                 and all(getattr(before, key) == getattr(after, key) for key in
                         ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")))
        return raw
    finally:
        os.close(fd)


def read_imported_alias_sources(*, repository, source_hashes):
    """Read a complete flat population within bounds before allocating source."""
    _require(type(source_hashes) is dict and 1 <= len(source_hashes) <= 256
             and all(type(name) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", name)
                     for name in source_hashes))
    sources, remaining = {}, 4_000_000
    for name, digest in sorted(source_hashes.items()):
        raw = _read_current_source(Path(repository) / name, digest, maximum=min(1_000_000, remaining))
        sources[name] = raw.decode("utf-8")
        remaining -= len(raw)
    return sources


def _literal_default(node):
    # Only immutable, literal defaults. Never evaluate source expressions or
    # permit defaults/annotations/decorators to execute during module loading.
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in {int, float}):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    _require(isinstance(node, ast.Constant) and
             type(node.value) in {type(None), bool, int, float, str, bytes})
    return node.value


def _docstring(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and type(node.value.value) is str)


def _return_expression(function):
    body = function.body[1:] if function.body and _docstring(function.body[0]) else function.body
    _require(len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None)
    return body[0].value


def _signature(function):
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    parameters = [*positional, *args.kwonlyargs]
    _require(_IDENTIFIER.fullmatch(function.name) and not function.name.startswith("__")
             and len(parameters) <= MAX_PARAMETERS
             and len({arg.arg for arg in parameters}) == len(parameters)
             and not function.decorator_list and function.returns is None
             and not function.type_comment and not getattr(function, "type_params", ())
             and not args.vararg and not args.kwarg
             and all(_IDENTIFIER.fullmatch(arg.arg) and arg.annotation is None and not arg.type_comment
                     for arg in parameters))
    defaults = [inspect.Parameter.empty] * (len(positional) - len(args.defaults))
    defaults.extend(_literal_default(node) for node in args.defaults)
    return inspect.Signature([
        *(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs)
                            else inspect.Parameter.POSITIONAL_OR_KEYWORD, default=defaults[index])
          for index, arg in enumerate(positional)),
        *(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                            default=inspect.Parameter.empty if default is None else _literal_default(default))
          for arg, default in zip(args.kwonlyargs, args.kw_defaults))])


def _call_binding(signature, call):
    """Finite signature projection, checked separately by inspect and provers."""
    parameters = tuple(signature.parameters.values())
    return {
        "positional_parameters": [p.name for p in parameters if p.kind in {
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}],
        "keyword_parameters": [p.name for p in parameters if p.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}],
        "required_parameters": [p.name for p in parameters if p.default is inspect.Parameter.empty],
        "positional_argument_count": len(call.args),
        "keyword_arguments": [keyword.arg for keyword in call.keywords],
    }


def _population(sources):
    _require(type(sources) is dict and 1 <= len(sources) <= 256)
    _require(all(type(name) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", name)
                 and type(text) is str and len(text.encode()) <= 1_000_000
                 for name, text in sources.items())
             and sum(len(text.encode()) for text in sources.values()) <= 4_000_000)
    modules = {}
    for name, text in sources.items():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as error:
            raise ImportedAliasContractError("closed alias source does not parse") from error
        functions, imports = {}, {}
        for index, node in enumerate(tree.body):
            if index == 0 and _docstring(node):
                continue
            if isinstance(node, ast.FunctionDef):
                _require(node.name not in functions and node.name not in imports)
                _return_expression(node)
                functions[node.name] = (node, _signature(node))
            else:
                _require(isinstance(node, ast.ImportFrom) and node.level == 0
                         and node.module is not None and _IDENTIFIER.fullmatch(node.module)
                         and node.module not in _RESERVED_MODULES
                         and node.module + ".py" in sources and node.module + ".py" != name)
                for alias in node.names:
                    local = alias.asname or alias.name
                    _require(_IDENTIFIER.fullmatch(alias.name) and _IDENTIFIER.fullmatch(local)
                             and local not in imports and local not in functions)
                    imports[local] = (node.module + ".py", alias.name, alias.asname, node)
            _require(len(functions) + len(imports) <= MAX_MODULE_BINDINGS)
        _require(functions and not (set(functions) & set(imports)))
        modules[name] = (functions, imports)
    # Every imported export must be an exact function in a function-only donor.
    # This excludes chains, cycles, __getattr__, packages and import-time effects.
    for functions, imports in modules.values():
        for donor, exported, _, _ in imports.values():
            donor_functions, donor_imports = modules[donor]
            _require(not donor_imports and exported in donor_functions)
    return modules


def _contracts(sources, path):
    modules = _population(sources)
    _require(path in modules)
    source_hashes = {name: _sha(text) for name, text in sorted(sources.items())}
    contracts = []
    call_edges = {}
    for module_path, (functions, imports) in modules.items():
        names = set(functions) | set(imports)
        for owner, (function, signature) in functions.items():
            parameters = set(signature.parameters)
            _require(not parameters.intersection(names))
            expression = _return_expression(function)
            calls = [node for node in ast.walk(expression) if isinstance(node, ast.Call)]
            # Single direct calls with inert argument expressions; no callbacks,
            # nested calls, attributes, comprehensions, assignment or mutation.
            _require(not calls or len(calls) == 1 and calls[0] is expression)
            call = calls[0] if calls else None
            for node in ast.walk(expression):
                if node is call or isinstance(node, ast.keyword):
                    continue
                # Non-call returns may compute over parameters. Their behavior
                # is unchanged, not proved: operators may be overloaded by the
                # caller. Call arguments keep the narrower inert grammar.
                _require(isinstance(node, _PURE_EXPRESSIONS if call else _RETURN_EXPRESSIONS))
                if isinstance(node, ast.Name) and (call is None or node is not call.func):
                    _require(node.id in parameters)
            if call is None:
                continue
            _require(isinstance(call.func, ast.Name) and not any(isinstance(arg, ast.Starred) for arg in call.args)
                     and all(keyword.arg is not None for keyword in call.keywords)
                     and len({keyword.arg for keyword in call.keywords}) == len(call.keywords))
            callee = call.func.id
            _require(callee not in parameters)
            candidate = None
            if callee in functions:
                target_module, target_name = module_path, callee
                target_signature = functions[callee][1]
            elif callee in imports:
                donor, exported, _, _ = imports[callee]
                target_module, target_name = donor, exported
                target_signature = modules[donor][0][exported][1]
            else:
                _require(callee not in vars(builtins) and callee not in _IMPLICIT_GLOBALS)
                matches = [(local, row) for local, row in imports.items()
                           if row[1] == callee and row[2] is not None and local != callee]
                _require(len(matches) == 1)
                local, (donor, exported, _, declaration) = matches[0]
                target_module, target_name = donor, exported
                target_signature = modules[donor][0][exported][1]
                bindings = {name: content_identity({"module": module_path, "declaration": ast.dump(row[0])})
                            for name, row in functions.items()}
                bindings.update({name: content_identity({"module": row[0],
                    "declaration": ast.dump(modules[row[0]][0][row[1]][0])})
                    for name, row in imports.items()})
                lines = sources[module_path].splitlines(keepends=True)
                start = len("".join(lines[:call.func.lineno - 1])) + len(
                    lines[call.func.lineno - 1].encode()[:call.func.col_offset].decode())
                candidate = {
                    "schema": OPERATOR, "path": module_path, "donor_path": donor,
                    "subject": owner, "previous": callee, "replacement": local,
                    "offset": start, "end_offset": start + len(callee),
                    "source_hashes": source_hashes, "bindings": dict(sorted(bindings.items())),
                    "target_binding": bindings[local],
                    "import_ast": content_identity({"ast": ast.dump(declaration)}),
                    "donor_declaration_ast": content_identity({"ast": ast.dump(modules[donor][0][exported][0])}),
                    "call_arguments_ast": content_identity({"args": [ast.dump(arg) for arg in call.args],
                        "keywords": [ast.dump(keyword) for keyword in call.keywords]}),
                    "call_binding": _call_binding(target_signature, call),
                }
            call_edges[(module_path, owner)] = (target_module, target_name)
            try:
                target_signature.bind(*([None] * len(call.args)), **{kw.arg: None for kw in call.keywords})
            except TypeError as error:
                raise ImportedAliasContractError("declared alias call does not bind the donor signature") from error
            if candidate is not None:
                _require(module_path == path)
                contracts.append(candidate)
    for node in call_edges:
        seen = set()
        while node in call_edges:
            _require(node not in seen)
            seen.add(node)
            node = call_edges[node]
    return contracts


@dataclass(frozen=True)
class ImportedAliasRepair:
    """An immutable candidate that reconstructs its own source-derived contract."""
    _contract_json: str = field(repr=False)
    _sources_json: str = field(repr=False)

    def __post_init__(self):
        _require(type(self._contract_json) is str and len(self._contract_json.encode()) <= 262144
                 and type(self._sources_json) is str and len(self._sources_json.encode()) <= 8_000_000)
        contract, sources = self.to_dict(), self.sources()
        _require(type(contract) is dict and _contracts(sources, contract.get("path")) == [contract])

    def to_dict(self):
        return json.loads(self._contract_json)

    def sources(self):
        return json.loads(self._sources_json)

    @property
    def contract_id(self):
        return content_identity(self.to_dict())

    def formal_projection(self, consequence):
        """Project the exact finite source environment, never target execution.

        Python import machinery is an explicit assumption. AST replay supplies
        the finite environment; neither Lean nor Z3 is a Python interpreter.
        """
        contract = self.to_dict()
        quote = json.dumps
        bindings = ", ".join(f"({quote(name)}, {quote(value)})"
                             for name, value in contract["bindings"].items())
        alias, original, target = (quote(contract[key])
                                   for key in ("replacement", "previous", "target_binding"))
        binding = contract["call_binding"]
        literal = lambda values: "[" + ", ".join(quote(value) for value in values) + "]"
        lean = (
            f"-- theorem:{OPERATOR} property:source-alias-argument-preservation "
            f"claim:closed-local-alias-binding contract:{self.contract_id} consequence:{consequence}\n"
            "def lookupBinding (key : String) : List (String × String) → Option String\n"
            "  | [] => none\n"
            "  | (name, entity) :: rest => if key == name then some entity else lookupBinding key rest\n"
            f"def sourceBindings : List (String × String) := [{bindings}]\n"
            f"def positionalParameters : List String := {literal(binding['positional_parameters'])}\n"
            f"def keywordParameters : List String := {literal(binding['keyword_parameters'])}\n"
            f"def requiredParameters : List String := {literal(binding['required_parameters'])}\n"
            f"def positionalArgumentCount : Nat := {binding['positional_argument_count']}\n"
            f"def keywordArguments : List String := {literal(binding['keyword_arguments'])}\n"
            "def suppliedParameters : List String := positionalParameters.take positionalArgumentCount ++ keywordArguments\n"
            "def declaredCallBinds : Bool :=\n"
            "  decide (positionalArgumentCount ≤ positionalParameters.length) &&\n"
            "  keywordArguments.all (fun name => keywordParameters.contains name) &&\n"
            "  requiredParameters.all (fun name => suppliedParameters.contains name) &&\n"
            "  suppliedParameters.all (fun name => suppliedParameters.count name == 1)\n"
            f"def repairCall (args : List String) : String × List String := ({alias}, args)\n"
            f"theorem resolved_alias : lookupBinding (repairCall []).1 sourceBindings = some {target} := by decide\n"
            f"theorem original_unbound : lookupBinding {original} sourceBindings = none := by decide\n"
            "theorem retained_arguments (args : List String) : (repairCall args).2 = args := by rfl\n"
            "theorem declared_call_binds : declaredCallBinds = true := by decide\n"
            "#print axioms resolved_alias\n#print axioms original_unbound\n#print axioms retained_arguments\n"
            "#print axioms declared_call_binds\n"
        )
        lookup = '""'
        for name, entity in reversed(tuple(contract["bindings"].items())):
            lookup = f"(ite (= key {quote(name)}) {quote(entity)} {lookup})"
        def member(name, population):
            return "(or false " + " ".join(f"(= {quote(name)} {quote(item)})" for item in population) + ")"

        supplied = (binding["positional_parameters"][:binding["positional_argument_count"]]
                    + binding["keyword_arguments"])
        binding_clauses = [f"(<= {binding['positional_argument_count']} {len(binding['positional_parameters'])})"]
        binding_clauses.extend(member(name, binding["keyword_parameters"]) for name in binding["keyword_arguments"])
        binding_clauses.extend(member(name, supplied) for name in binding["required_parameters"])
        binding_clauses.extend(f"(not (= {quote(name)} {quote(other)}))"
                               for index, name in enumerate(supplied) for other in supplied[index + 1:])
        smt = ("(set-logic QF_SLIA)\n"
               f"(define-fun lookupBinding ((key String)) String {lookup})\n"
               f"(define-fun declaredCallBinds () Bool (and {' '.join(binding_clauses)}))\n"
               "(declare-const args String)\n"
               "(define-fun retainedArgs ((value String)) String value)\n"
               f"(assert (or (not (= (lookupBinding {alias}) {target})) "
               f"(not (= (lookupBinding {original}) \"\")) (not (= (retainedArgs args) args)) "
               "(not declaredCallBinds)))\n"
               "(check-sat)\n")
        return {"lean": lean, "smt": smt,
                "expected_axioms": [f"'{name}' does not depend on any axioms"
                                    for name in ("resolved_alias", "original_unbound", "retained_arguments", "declared_call_binds")]}

    def reconstruct(self, request, subject):
        contract, sources = self.to_dict(), self.sources()
        _require(_contracts(sources, contract["path"]) == [contract])
        proposal = request.proposal
        site = proposal.edit_site
        _require(proposal.kind.value == "exact_rename" and subject == contract["subject"]
                 and site.path == contract["path"] and site.span_start == contract["offset"]
                 and site.span_end == contract["end_offset"]
                 and proposal.previous_parameter_name == contract["previous"]
                 and proposal.parameter_name == contract["replacement"]
                 and request.file_text == sources[contract["path"]]
                 and request.span_text == contract["previous"])
        before = request.file_text
        after = before[:site.span_start] + contract["replacement"] + before[site.span_end:]
        # Independent replacement-tree check: only the nominated call head changes.
        tree = ast.parse(before)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == subject)
        expression = _return_expression(function)
        _require(isinstance(expression, ast.Call))
        expression.func.id = contract["replacement"]
        _require(ast.dump(tree) == ast.dump(ast.parse(after)))
        self.validate_candidate(after)
        return after

    def validate_candidate(self, after):
        contract, sources = self.to_dict(), self.sources()
        before = sources[contract["path"]]
        _require(after == before[:contract["offset"]] + contract["replacement"] + before[contract["end_offset"]:])
        sources[contract["path"]] = after
        _require(_contracts(sources, contract["path"]) == [])

    def assert_current(self, repository):
        sources = self.sources()
        for name, digest in self.to_dict()["source_hashes"].items():
            path = Path(repository) / name
            size = len(sources[name].encode("utf-8"))
            _read_current_source(path, digest, maximum=size, expected_size=size)


def discover_imported_alias_repair(*, sources: dict[str, str], path: str):
    """Select exactly one existing explicit alias; never guess among alternatives."""
    contracts = _contracts(sources, path)
    _require(len(contracts) <= 1)
    return ImportedAliasRepair(_json(contracts[0]), _json(sources)) if contracts else None
