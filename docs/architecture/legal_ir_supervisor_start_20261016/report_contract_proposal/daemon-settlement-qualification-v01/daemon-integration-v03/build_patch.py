"""Exact source-only successor: retain the original finally continuations.

No daemon import, database, Git or supervisor execution occurs in this builder.
The original product checkout and prior frozen generations are never edited.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import importlib.util
import json

HERE=Path(__file__).absolute().parent
BASE=Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-bridge-20261017-01/daemon-integration-v02')
EXPECTED_BUILDER='38e1fa5c63800a6e10c9ba54788010c75710daad867d41704cc7084677d46870'
EXPECTED_BASE='9da3b39ff403317d5d3b5343a6b2f6bb4224510f676a458ee71d683dff4b6179'
RELATIVE='ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
STAGES={
    '_run_implementation':('implementation_claim_release','_settle_report_only_implementation_claims'),
    '_run_once':('pass_dispatch_intent_reconciliation','_settle_report_only_dispatch_intents'),
}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def original_method(source,name):
    cls=next(row for row in ast.parse(source).body if isinstance(row,ast.ClassDef)and row.name=='PortalImplementationDaemon')
    return next(row for row in cls.body if isinstance(row,ast.FunctionDef)and row.name==name)


def successor(source):
    if sha(source.encode())!=EXPECTED_BASE:raise ValueError('exact_frozen_daemon_V02_required')
    for method_name,(stage,continuation_name)in STAGES.items():
        method=original_method(source,method_name)
        tries=[row for row in ast.walk(method)if isinstance(row,ast.Try)and row.finalbody
               and isinstance(row.finalbody[0],ast.If)
               and any(isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)
                       and call.func.attr=='before_original_finalizer'for call in ast.walk(row.finalbody[0]))]
        if len(tries)!=1:raise ValueError('one_original_guarded_finalizer_required')
        original=tries[0].finalbody
        guard=original[0]
        call=next(row for row in ast.walk(guard)if isinstance(row,ast.Call)and isinstance(row.func,ast.Attribute)
                  and row.func.attr=='before_original_finalizer')
        frame=next(keyword.value for keyword in call.keywords if keyword.arg=='original_frame')
        lines=source.splitlines(keepends=True)
        tail=''.join(lines[original[1].lineno-1:original[-1].end_lineno])
        nested='            def '+continuation_name+'():\n'
        if method_name=='_run_implementation':nested+='                nonlocal acquired_resource_claims\n'
        nested+=''.join('    '+line for line in tail.splitlines(keepends=True))
        nested+='            if self.report_only_candidate_freezing:\n'
        nested+='                from .report_only_native_host import run_original_finalizer\n\n'
        nested+='                run_original_finalizer(\n'
        nested+='                    self, stage='+repr(stage)+',\n'
        nested+='                    original_frame='+ast.unparse(frame)+',\n'
        nested+='                    continuation='+continuation_name+',\n'
        nested+='                )\n'
        nested+='            else:\n                '+continuation_name+'()\n'
        source=''.join(lines[:guard.lineno-1])+nested+''.join(lines[original[-1].end_lineno:])
    ast.parse(source)
    return source


def assemble():
    p=BASE/'build_patch.py'
    if sha(p.read_bytes())!=EXPECTED_BUILDER:raise ValueError('original_source_assembler_changed')
    spec=importlib.util.spec_from_file_location('frozen_V02_source_only_assembler',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original,base=module.assemble()
    return original,base,successor(base)


def main():
    original,base,proposed=assemble()
    outputs={'implementation_daemon.patch':''.join(difflib.unified_diff(original.splitlines(True),proposed.splitlines(True),fromfile='a/'+RELATIVE,tofile='b/'+RELATIVE)),
             'continuation-successor.patch':''.join(difflib.unified_diff(base.splitlines(True),proposed.splitlines(True),fromfile='daemon-integration-v02/'+RELATIVE,tofile='daemon-integration-v03/'+RELATIVE))}
    for name,text in outputs.items():
        with(HERE/name).open('x')as stream:stream.write(text)
    with(HERE/'ASSEMBLY.json').open('x')as stream:
        json.dump({'schema':'legal-ir-finalizer-source-assembly@1','original_sha256':sha(original.encode()),
                   'base_sha256':sha(base.encode()),'proposed_sha256':sha(proposed.encode()),
                   'runtime_enabled':False,'qualification':'source assembly only'},stream,indent=2,sort_keys=True);stream.write('\n')


if __name__=='__main__':main()
