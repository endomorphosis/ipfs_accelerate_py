from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_provider import (
    POLYGLOT_AST_PROVIDER_SCHEMA,
    TYPESCRIPT_EXTRACTOR_VERSION,
    PolyglotASTInput,
    PolyglotASTLimits,
    PolyglotASTProvider,
    PolyglotASTProviderError,
    build_polyglot_ast_blob_record,
    build_structured_schema_ast_blob_record,
    language_for_path,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ASTBlobRecord


_MODULE = "ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_provider"
_EXTRACTOR = (
    Path(__file__).resolve().parents[2] / "scripts" / "extract_typescript_ast.mjs"
)


def _hash(source: str) -> str:
    return "sha256:" + hashlib.sha256(
        source.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _successful_response(
    request: dict,
    *,
    version: str = "5.7.3",
    parse_error: str = "",
) -> bytes:
    facts = {
        "qualified_symbols": [] if parse_error else ["Service", "Service.run"],
        "imports": [] if parse_error else ['import {Client} from "@scope/client"'],
        "calls": [] if parse_error else ["Service.run->this.client.send"],
        "state_transitions": (
            [] if parse_error else ["Service.run:this.status:assign:\"ready\""]
        ),
        "interfaces": (
            [] if parse_error else ["Service.run:run(input: Input): Output"]
        ),
        "symbol_hashes": (
            {}
            if parse_error
            else {
                "Service": "sha256:" + "1" * 64,
                "Service.run": "sha256:" + "2" * 64,
            }
        ),
        "symbol_lines": (
            {} if parse_error else {"Service": [3, 9], "Service.run": [5, 8]}
        ),
    }
    return json.dumps(
        {
            "protocol_version": 1,
            "ok": True,
            "producer": "typescript-compiler-api",
            "producer_version": TYPESCRIPT_EXTRACTOR_VERSION,
            "compiler": {"name": "typescript", "version": version},
            "language": request["language"],
            "source_sha256": request["source_sha256"],
            "parse_error": parse_error,
            "facts": facts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _fixture_runner(calls: list[dict], *, version: str = "5.7.3"):
    def run(command, request, timeout, max_output, environment):
        payload = json.loads(request)
        calls.append(
            {
                "command": tuple(command),
                "request": payload,
                "timeout": timeout,
                "max_output": max_output,
                "environment": dict(environment),
            }
        )
        return 0, _successful_response(payload, version=version), b""

    return run


def test_cold_import_and_construction_never_start_node() -> None:
    package_root = Path(__file__).resolve().parents[2]
    code = f"""
import subprocess

def forbidden(*args, **kwargs):
    raise AssertionError("cold import started a child process")

subprocess.Popen = forbidden
subprocess.run = forbidden
from {_MODULE} import PolyglotASTProvider
provider = PolyglotASTProvider()
assert provider.schema.endswith("@1")
assert provider.limits.max_files > 0
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(package_root), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_python_and_structured_adapters_emit_canonical_source_free_records() -> None:
    python_source = """from typing import Protocol

class Runner(Protocol):
    def run(self, request): ...

def dispatch(request):
    state = "ready"
    return Runner().run(request)
"""
    python_record = build_polyglot_ast_blob_record(
        python_source,
        "python",
        blob_identity="blob:python-fixture",
    )

    assert isinstance(python_record, ASTBlobRecord)
    assert python_record.blob_identity == "blob:python-fixture"
    assert python_record.source_sha256 == _hash(python_source)
    assert {"Runner", "Runner.run", "dispatch"}.issubset(
        python_record.qualified_symbols
    )
    assert python_record.calls
    assert python_record.interfaces
    assert python_record.state_transitions
    assert "source" not in python_record.to_dict()

    schema = {
        "$id": "contract-v1",
        "$defs": {
            "Request": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "parent": {"$ref": "#/$defs/Request"},
                },
            }
        },
    }
    first = build_structured_schema_ast_blob_record(schema)
    second = build_structured_schema_ast_blob_record(
        json.dumps(schema, indent=4, sort_keys=False)
    )

    assert {"Request", "Request.id", "Request.parent"}.issubset(
        first.qualified_symbols
    )
    assert "$ref:#/$defs/Request" in first.imports
    assert "Request.id:type=string;required=True" in first.interfaces
    assert first.symbol_hashes == second.symbol_hashes
    assert first.qualified_symbols == second.qualified_symbols
    assert '"source"' not in json.dumps(first.to_dict(), sort_keys=True)


def test_typescript_response_preserves_all_facts_and_binds_tool_version() -> None:
    calls: list[dict] = []
    provider = PolyglotASTProvider(
        expected_typescript_version="5.7.3",
        typescript_path="/toolchains/typescript",
        process_runner=_fixture_runner(calls),
    )
    source = """import { Client } from "@scope/client";
export class Service {
  private status = "new";
  run(input: Input): Output {
    this.status = "ready";
    return this.client.send(input);
  }
}
"""
    extraction = provider.extract_with_metadata(
        source,
        "ts",
        blob_identity="blob:typescript-fixture",
    )
    record = extraction.record

    assert extraction.language == "typescript"
    assert extraction.compiler_name == "typescript"
    assert extraction.compiler_version == "5.7.3"
    assert extraction.tool_identity == (
        "typescript-ast-extractor@2/typescript@5.7.3"
    )
    assert record.language == "typescript@typescript-5.7.3"
    assert record.qualified_symbols == ("Service", "Service.run")
    assert record.imports == ('import {Client} from "@scope/client"',)
    assert record.calls == ("Service.run->this.client.send",)
    assert record.state_transitions == (
        'Service.run:this.status:assign:"ready"',
    )
    assert record.interfaces == (
        "Service.run:run(input: Input): Output",
    )
    assert record.symbol_lines == {"Service": (3, 9), "Service.run": (5, 8)}
    assert all(value.startswith("sha256:") for value in record.symbol_hashes.values())
    serialized = extraction.to_dict()
    assert serialized["provider_schema"] == POLYGLOT_AST_PROVIDER_SCHEMA
    assert "source" not in serialized
    assert calls[0]["request"]["source"] == source
    assert calls[0]["request"]["source_sha256"] == _hash(source)
    assert calls[0]["environment"]["TYPESCRIPT_PATH"] == (
        "/toolchains/typescript"
    )
    assert calls[0]["command"][1].startswith("--max-old-space-size=")

    other_compiler = PolyglotASTProvider(
        process_runner=_fixture_runner([], version="5.8.0")
    ).extract(source, "typescript")
    assert other_compiler.record_id != record.record_id


def test_explicit_typescript_path_is_normalized_to_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    provider = PolyglotASTProvider(
        typescript_path="toolchains/typescript/lib/typescript.js"
    )

    assert provider.typescript_path == str(
        (
            tmp_path
            / "toolchains"
            / "typescript"
            / "lib"
            / "typescript.js"
        ).resolve()
    )


def test_typed_parse_error_is_a_version_bound_empty_record() -> None:
    parse_error = "typescript_parse_error:TS1005@2:12:'}' expected."

    def runner(command, request, timeout, maximum, environment):
        payload = json.loads(request)
        return 0, _successful_response(payload, parse_error=parse_error), b""

    record = PolyglotASTProvider(process_runner=runner).extract(
        "export function broken( {",
        "typescript",
        blob_identity="blob:broken",
    )

    assert record.parse_error == parse_error
    assert record.language == "typescript@typescript-5.7.3"
    assert record.qualified_symbols == ()
    assert record.calls == ()
    assert record.symbol_hashes == {}


def test_file_count_per_file_total_and_identity_limits_fail_before_process() -> None:
    calls: list[dict] = []
    provider = PolyglotASTProvider(
        PolyglotASTLimits(
            max_files=2,
            max_file_bytes=8,
            max_total_bytes=10,
            max_output_bytes=1024,
        ),
        process_runner=_fixture_runner(calls),
    )

    with pytest.raises(PolyglotASTProviderError) as count:
        provider.extract_many(
            [
                PolyglotASTInput("x", "js"),
                PolyglotASTInput("y", "js"),
                PolyglotASTInput("z", "js"),
            ]
        )
    assert count.value.reason_code == "file_limit_exceeded"

    with pytest.raises(PolyglotASTProviderError) as per_file:
        provider.extract("nine-byte", "js")
    assert per_file.value.reason_code == "file_bytes_exceeded"

    with pytest.raises(PolyglotASTProviderError) as total:
        provider.extract_many(
            [
                PolyglotASTInput("123456", "python"),
                PolyglotASTInput("123456", "python"),
            ]
        )
    assert total.value.reason_code == "total_bytes_exceeded"

    with pytest.raises(PolyglotASTProviderError) as identity:
        provider.extract("value", "js", source_sha256="sha256:" + "0" * 64)
    assert identity.value.reason_code == "source_identity_mismatch"
    assert calls == []


def test_unsupported_inputs_and_compiler_version_are_typed() -> None:
    with pytest.raises(PolyglotASTProviderError) as unsupported:
        PolyglotASTProvider().extract("fn main() {}", "rust")
    assert unsupported.value.reason_code == "unsupported_language"

    with pytest.raises(PolyglotASTProviderError) as suffix:
        language_for_path("src/service.rs")
    assert suffix.value.reason_code == "unsupported_language"

    provider = PolyglotASTProvider(
        expected_typescript_version="5.7.2",
        process_runner=_fixture_runner([], version="5.7.3"),
    )
    with pytest.raises(PolyglotASTProviderError) as mismatch:
        provider.extract("export const value = 1;", "typescript")
    assert mismatch.value.reason_code == "compiler_version_mismatch"
    assert mismatch.value.details == {"expected": "5.7.2", "actual": "5.7.3"}


def test_extractor_protocol_and_output_limit_fail_closed() -> None:
    def invalid_protocol(command, request, timeout, maximum, environment):
        return 0, b'{"protocol_version":999,"ok":true}', b""

    with pytest.raises(PolyglotASTProviderError) as protocol:
        PolyglotASTProvider(process_runner=invalid_protocol).extract(
            "const value = 1;",
            "js",
        )
    assert protocol.value.reason_code == "protocol_error"

    def excessive_output(command, request, timeout, maximum, environment):
        raise PolyglotASTProviderError(
            "output_bytes_exceeded",
            "fixture exceeded the response cap",
        )

    with pytest.raises(PolyglotASTProviderError) as output:
        PolyglotASTProvider(process_runner=excessive_output).extract(
            "const value = 1;",
            "js",
        )
    assert output.value.reason_code == "output_bytes_exceeded"


def test_real_process_timeout_and_output_caps_kill_extractors(tmp_path: Path) -> None:
    hanging = tmp_path / "hang.mjs"
    hanging.write_text(
        "process.stdin.resume(); setInterval(() => {}, 1000);",
        encoding="utf-8",
    )
    timeout_provider = PolyglotASTProvider(
        PolyglotASTLimits(
            process_timeout_seconds=0.1,
            max_output_bytes=1024,
        ),
        extractor_path=hanging,
    )
    with pytest.raises(PolyglotASTProviderError) as timeout:
        timeout_provider.extract("const value = 1;", "javascript")
    assert timeout.value.reason_code == "process_timeout"

    noisy = tmp_path / "noisy.mjs"
    noisy.write_text(
        "process.stdin.resume(); process.stdout.write('x'.repeat(100000));",
        encoding="utf-8",
    )
    output_provider = PolyglotASTProvider(
        PolyglotASTLimits(max_output_bytes=1024),
        extractor_path=noisy,
    )
    with pytest.raises(PolyglotASTProviderError) as output:
        output_provider.extract("const value = 1;", "javascript")
    assert output.value.reason_code == "output_bytes_exceeded"


def _local_typescript_path() -> str:
    explicit = os.environ.get("TYPESCRIPT_PATH", "")
    if explicit:
        return explicit
    result = subprocess.run(
        [
            "node",
            "-e",
            "try{process.stdout.write(require.resolve('typescript'))}"
            "catch(error){process.exit(2)}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def test_local_typescript_compiler_extracts_stable_semantic_facts() -> None:
    typescript_path = _local_typescript_path()
    if not typescript_path:
        pytest.skip("local TypeScript compiler API is not installed")
    source = """import type { Request } from "./request";
import { Client as Transport } from "./client";

export interface Runner {
  run(request: Request): Promise<string>;
}

export class Service implements Runner {
  private status = "idle";
  constructor(private readonly client: Transport) {}

  async run(request: Request): Promise<string> {
    this.status = "running";
    this.setState("active");
    return this.client.send(request);
  }
}

export function leavesTypedLocalUninitialized(): void {
  let pending: string;
}
"""
    provider = PolyglotASTProvider(typescript_path=typescript_path)
    first = provider.extract_with_metadata(
        source,
        "typescript",
        blob_identity="blob:ts-v1",
    )
    second = provider.extract_with_metadata(
        source,
        "typescript",
        blob_identity="blob:ts-v1",
    )

    assert first.to_dict() == second.to_dict()
    record = first.record
    assert {
        "Runner",
        "Runner.run",
        "Service",
        "Service.constructor",
        "Service.run",
        "leavesTypedLocalUninitialized",
    }.issubset(record.qualified_symbols)
    assert 'import type {Request} from "./request"' in record.imports
    assert 'import {Client as Transport} from "./client"' in record.imports
    assert "Service.run->this.client.send" in record.calls
    assert any(
        transition.startswith("Service.run:this.status:assign:")
        for transition in record.state_transitions
    )
    assert any(":this.setState:call(" in item for item in record.state_transitions)
    assert record.symbol_lines["Service.run"] == (12, 16)
    assert all(
        value.startswith("sha256:") and len(value) == 71
        for value in record.symbol_hashes.values()
    )
    assert first.compiler_version


def test_extractor_contains_no_llm_or_network_path() -> None:
    provider_source = Path(
        sys.modules[_MODULE].__file__
    ).read_text(encoding="utf-8").casefold()
    extractor_source = _EXTRACTOR.read_text(encoding="utf-8").casefold()

    forbidden = (
        "openai",
        "anthropic",
        "generativelanguage",
        "fetch(",
        "https://",
        "http://",
    )
    assert not any(token in provider_source for token in forbidden)
    assert not any(token in extractor_source for token in forbidden)
