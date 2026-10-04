"""Authored programs for the benchmark-informed local header guard operator."""

import ast
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.doctor_header_contracts import (
    WsgiHeaderProtocolContract, analyze_http_header_contracts,
    verify_header_candidate,
)


PROTOCOL = WsgiHeaderProtocolContract("test-reviewed-wsgi-role", "respond")
PROGRAM = '''# Authored fixture, independent of any benchmark repository.
def convert(raw, encoding='utf8', errors='strict'):
    if isinstance(raw, (bytes, bytearray)):
        return str(raw, encoding, errors)
    return '' if raw is None else str(raw)

def clean_label(raw):
    label = convert(raw)
    return label.title().replace('_', '-')

def clean_payload(raw):
    payload = convert(raw)
    return payload

class WireResponse:
    def __init__(self):
        self._values = {}

    def put(self, label, payload):
        self._values[clean_label(label)] = [clean_payload(payload)]

    @property
    def fields_for_wire(self):
        pairs = list(self._values.items())
        return [(label, value) for label, values in pairs for value in values]

def application(environ, respond):
    response = WireResponse()
    respond('200 OK', response.fields_for_wire)
'''


def analyze(source=PROGRAM):
    return analyze_http_header_contracts(source, protocol=PROTOCOL)


def namespace(source):
    # Execute only this separately authored inert fixture, never target source.
    result = {}
    exec(compile(source, '<authored-header-fixture>', 'exec'), result)
    return result


def test_candidate_has_bound_contract_witnesses_exact_preimages_and_no_authority():
    result = analyze()
    assert result.status == 'candidate'
    candidate = result.candidate
    assert verify_header_candidate(PROGRAM, candidate)
    assert len(candidate.edits) == 2
    assert {edit.role for edit in candidate.edits} == {'field_name', 'field_value'}
    for edit in candidate.edits:
        assert PROGRAM[edit.start:edit.end] == edit.before
    contract = candidate.contract
    assert contract['forbidden_codepoints'] == [0, 10, 13]
    assert contract['error_type'] == 'ValueError'
    assert contract['protocol']['review_ref'] == PROTOCOL.review_ref
    assert 'reviewed_unique_property_receiver_binding' in contract['assumptions']
    assert 'whole_program_security' in result.open_frontiers
    assert 'not a whole-program' in contract['proof_scope']
    assert ast.parse(candidate.source)


@pytest.mark.parametrize('value', ['hello', '', 'éλ日本語', None, 42, b'hello', bytearray(b'value')])
def test_preserves_accepted_conversion_and_normalization(value):
    original = namespace(PROGRAM)
    candidate = namespace(analyze().candidate.source)
    for helper in ('clean_label', 'clean_payload'):
        assert candidate[helper](value) == original[helper](value)


@pytest.mark.parametrize('control', ['\r', '\n', '\x00'])
@pytest.mark.parametrize('position', ['', 'prefix', 'éλ'])
@pytest.mark.parametrize('as_bytes', [False, True])
def test_forbidden_converted_inputs_raise_value_error(control, position, as_bytes):
    module = namespace(analyze().candidate.source)
    value = position + control + 'tail'
    if as_bytes:
        value = value.encode()
    for helper in ('clean_label', 'clean_payload'):
        with pytest.raises(ValueError, match='forbidden control character'):
            module[helper](value)


def test_conversion_exception_is_preserved_and_report_not_generated_by_operator():
    module = namespace(analyze().candidate.source)
    with pytest.raises(UnicodeDecodeError):
        module['clean_payload'](b'\xff')
    assert 'report.jsonl' not in analyze().candidate.source


def test_renamed_helpers_receiver_store_property_and_callback():
    source = PROGRAM
    for before, after in [('clean_label', 'normalize_α'), ('clean_payload', 'text_β'),
                          ('_values', 'entries'), ('fields_for_wire', 'outgoing'),
                          ('WireResponse', 'Result'), ('respond', 'emit')]:
        source = source.replace(before, after)
    result = analyze_http_header_contracts(source, protocol=WsgiHeaderProtocolContract('renamed-protocol', 'emit'))
    assert result.status == 'candidate'
    assert verify_header_candidate(source, result.candidate)
    assert {edit.symbol for edit in result.candidate.edits} == {'normalize_α', 'text_β'}


def test_idempotent_and_partial_guard_repair():
    candidate = analyze().candidate
    again = analyze(candidate.source)
    assert again.status == 'already_satisfied'
    assert again.candidate is None
    first = candidate.edits[0]
    partial = PROGRAM[:first.start] + first.after + PROGRAM[first.end:]
    result = analyze(partial)
    assert result.status == 'candidate'
    assert len(result.candidate.edits) == 1
    assert result.candidate.source == candidate.source


@pytest.mark.parametrize('source, reason', [
    (PROGRAM.replace('response.fields_for_wire', '[]'), 'ambiguous_or_missing_reviewed_wsgi_sink'),
    (PROGRAM.replace('payload = convert(raw)', 'payload = arbitrary(raw)'), 'unsupported_normalizer_or_conversion_shape'),
    (PROGRAM.replace('return payload\n', "return payload.replace('safe', '\\n')\n"), 'unsupported_normalizer_or_conversion_shape'),
    (PROGRAM.replace('return payload\n', 'return eval(payload)\n'), 'unsupported_normalizer_or_conversion_shape'),
    (PROGRAM.replace('payload = convert(raw)', 'payload = convert(*raw)'), 'unsupported_normalizer_or_conversion_shape'),
    (PROGRAM + '\nValueError = RuntimeError\n', 'shadowed_contract_builtin'),
    (PROGRAM + '\nstr = lambda value: value\n', 'unsupported_normalizer_or_conversion_shape'),
    (PROGRAM + '\nclean_payload = other\n', 'ambiguous_normalizer_binding'),
    (PROGRAM.replace("errors='strict'", "errors='ignore'"), 'unsupported_normalizer_or_conversion_shape'),
    (PROGRAM.replace('return payload\n', "if '\\n' in payload:\n        raise Exception()\n    return payload\n"), 'unsupported_normalizer_or_conversion_shape'),
    (PROGRAM.replace('self._values.items()', 'self._values.values()'), 'ambiguous_or_missing_header_store_projection'),
    (PROGRAM.replace('class WireResponse:', 'class WireResponse:\n    @property\n    def another(self):\n        return []\n'), None),
])
def test_negative_and_unrelated_program_shapes(source, reason):
    result = analyze(source)
    if reason is None:
        assert result.status == 'candidate'
    else:
        assert result.status == 'unsupported'
        assert result.reason_codes == (reason,)
        assert result.candidate is None


def test_ambiguous_store_pair_and_duplicate_property_abstain():
    duplicate = PROGRAM.replace('    @property', '''    def other(self, label, payload):
        self._values[other_label(label)] = [clean_payload(payload)]

    @property''')
    assert analyze(duplicate).reason_codes == ('ambiguous_or_missing_header_normalizer_pair',)
    duplicate = PROGRAM + '\nclass Other:\n    @property\n    def fields_for_wire(self):\n        return list(self._anything.items())\n'
    assert analyze(duplicate).reason_codes == ('ambiguous_or_missing_header_property',)


def test_generic_dictionary_normalizers_without_wsgi_contract_witness_abstain():
    source = PROGRAM[:PROGRAM.index('def application')]
    assert analyze(source).status == 'unsupported'
    with pytest.raises(TypeError):
        analyze_http_header_contracts(PROGRAM, protocol=None)
    with pytest.raises(ValueError):
        WsgiHeaderProtocolContract('')


def test_exact_replay_rejects_stale_source_forged_edit_and_contract():
    candidate = analyze().candidate
    assert not verify_header_candidate(PROGRAM + '# drift\n', candidate)
    assert not verify_header_candidate(PROGRAM, replace(candidate, source=candidate.source + '# extra\n'))
    assert not verify_header_candidate(PROGRAM, replace(candidate, after_sha256='0' * 64))
    edit = replace(candidate.edits[0], after=candidate.edits[0].after.replace('ValueError', 'RuntimeError'))
    assert not verify_header_candidate(PROGRAM, replace(candidate, edits=(edit, *candidate.edits[1:])))
    forged = {**candidate.contract, 'forbidden_codepoints': [10]}
    assert not verify_header_candidate(PROGRAM, replace(candidate, contract=forged))


def test_docstrings_comments_unicode_separators_and_no_final_newline():
    source = PROGRAM.replace('def clean_label(raw):\n', 'def clean_label(raw):\n    """Name μ\u2028 documentation."""\n')
    source = source.replace('    return payload\n', '    # μ\u0085 does not split a Python physical line\n    return payload  # unchanged\n')
    source = source.rstrip('\n')
    result = analyze(source)
    assert result.status == 'candidate'
    assert verify_header_candidate(source, result.candidate)
    assert namespace(result.candidate.source)['clean_label']('x_key') == 'X-Key'


@pytest.mark.parametrize('replacement', [
    'def clean_payload(ValueError):\n    payload = convert(ValueError)\n    return payload',
    'def clean_payload(raw):\n    ValueError = convert(raw)\n    return ValueError',
    'def clean_payload(convert):\n    payload = convert(convert)\n    return payload',
    'def clean_payload(raw):\n    convert = convert(raw)\n    return convert',
])
def test_normalizer_local_binding_shadows_rejected(replacement):
    original = 'def clean_payload(raw):\n    payload = convert(raw)\n    return payload'
    result = analyze(PROGRAM.replace(original, replacement))
    assert result.reason_codes == ('unsupported_normalizer_or_conversion_shape',)


@pytest.mark.parametrize('name', ['str', 'isinstance', 'bytes', 'bytearray'])
def test_conversion_formal_builtin_shadow_rejected(name):
    source = PROGRAM.replace("convert(raw, encoding='utf8', errors='strict')", f"convert({name}, encoding='utf8', errors='strict')")
    source = source.replace('isinstance(raw,', f'isinstance({name},').replace('str(raw,', f'str({name},')
    source = source.replace("'' if raw is None else str(raw)", f"'' if {name} is None else str({name})")
    assert analyze(source).status == 'unsupported'


@pytest.mark.parametrize('suffix', [
    "\ndef change():\n    global ValueError\n    ValueError = RuntimeError\n",
    "\ndef change():\n    global clean_payload\n    clean_payload = lambda value: value\n",
    "\nglobals()['convert'] = other\n",
    "\n__builtins__['ValueError'] = RuntimeError\n",
])
def test_explicit_contract_binding_mutations_rejected(suffix):
    assert analyze(PROGRAM + suffix).reason_codes == ('explicit_contract_binding_mutation',)


@pytest.mark.parametrize('body', [
    'pairs = list(self._values.items())\n        return []',
    'pairs = list(self._values.items())\n        pairs = []\n        return [(label, value) for label, values in pairs for value in values]',
    'pairs = list(self._values.items())\n        if True:\n            pairs = []\n        return [(label, value) for label, values in pairs for value in values]',
    'pairs = list(self._values.items())\n        pairs.clear()\n        return [(label, value) for label, values in pairs for value in values]',
])
def test_unused_and_overwritten_store_read_is_not_a_flow_witness(body):
    original = 'pairs = list(self._values.items())\n        return [(label, value) for label, values in pairs for value in values]'
    assert analyze(PROGRAM.replace(original, body)).reason_codes == ('unproved_header_store_projection',)


def test_rebound_wsgi_callback_does_not_establish_protocol_role():
    source = PROGRAM.replace('    response = WireResponse()', '    respond = arbitrary\n    response = WireResponse()')
    assert analyze(source).reason_codes == ('reviewed_callback_rebound',)


@pytest.mark.parametrize('normalization', ['label', 'label.lower()', 'label.upper()', 'label.casefold()', "label.replace('_', '-')"])
def test_holdout_name_normalizations_preserved(normalization):
    source = PROGRAM.replace("label.title().replace('_', '-')", normalization)
    result = analyze(source)
    assert result.status == 'candidate'
    original = namespace(source)
    repaired = namespace(result.candidate.source)
    assert repaired['clean_label']('HELLO_é') == original['clean_label']('HELLO_é')
    with pytest.raises(ValueError):
        repaired['clean_label']('x\rvalue')


def test_direct_builtin_str_conversion_holdout():
    source = PROGRAM.replace('label = convert(raw)', 'label = str(raw)').replace('payload = convert(raw)', 'payload = str(raw)')
    candidate = analyze(source).candidate
    assert candidate is not None
    assert namespace(candidate.source)['clean_payload'](42) == '42'


def test_setter_formal_shadow_and_property_decorator_shadow_rejected():
    source = PROGRAM.replace('def put(self, label, payload):', 'def put(self, clean_label, payload):')
    source = source.replace('self._values[clean_label(label)]', 'self._values[clean_label(clean_label)]')
    assert analyze(source).reason_codes == ('shadowed_setter_normalizer',)
    source = PROGRAM.replace('class WireResponse:\n', 'class WireResponse:\n    property = arbitrary\n')
    assert analyze(source).reason_codes == ('shadowed_property_decorator',)


def test_getter_local_list_shadow_rejected_and_extra_header_sources_remain_unproved():
    source = PROGRAM.replace('        pairs = list(self._values.items())',
                             '        list = arbitrary\n        pairs = list(self._values.items())')
    assert analyze(source).reason_codes == ('unproved_header_store_projection',)
    source = PROGRAM.replace('        return [(label, value) for label, values in pairs for value in values]',
        "        output = [(label, value) for label, values in pairs for value in values]\n"
        "        output.append(('unsafe\\nlabel', 'raw'))\n        return output")
    result = analyze(source)
    assert result.status == 'candidate'
    assert 'additional_header_sources_unproved' in result.open_frontiers
    assert 'additional header sources unproved' in result.contracts[0]['witnesses'][0]['flow_claim']
