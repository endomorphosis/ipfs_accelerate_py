import ast, hashlib, json, pathlib, sys
request = json.loads(pathlib.Path("request.json").read_text(encoding="ascii"))
raw = pathlib.Path("captured_source.py").read_bytes()
assert hashlib.sha256(raw).hexdigest() == request["source_sha256"]
tree = ast.parse(raw, type_comments=True)
assert len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef
namespace = {"__builtins__": {"int": int}}
exec(compile(raw, "captured_source.py", "exec"), namespace, namespace)
function = namespace[request["function_name"]]
rows = []
trace = {"schema": "codebase-finite-integer-trace@1", "source_cid": request["source_cid"],
         "source_sha256": request["source_sha256"], "domain_cid": request["domain_cid"],
         "inputs": request["inputs"], "observations": rows, "status": "complete",
         "exception_type": None,
         "python": {"executable": str(pathlib.Path(sys.executable).resolve()),
                    "version": sys.version, "implementation": sys.implementation.name,
                    "cache_tag": sys.implementation.cache_tag}}
try:
    for value in request["inputs"]:
        assert type(value) is int and abs(value) <= 2**31
        result = function(value)
        if type(result) is not int:
            raise TypeError("exact integer result required")
        rows.append({"input": value, "output": result, "input_type": "int", "output_type": "int"})
except BaseException as error:
    trace["status"] = "exception"
    trace["exception_type"] = type(error).__name__
print(json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
