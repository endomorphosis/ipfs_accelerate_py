"""Owner-selected stop before a later semantic provider callback.

Prior complete scoped observations can stop a later callback. The first callback
can overshoot; native CLI internal requests remain uncapped. This is neither a
provider reservation nor a complete task/billing estimate. Native process
custody and cancellation remain with the existing provider adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import threading

from .native_session_usage_projection import MAX_COUNTER, reconcile_native_usage_observations

SCHEMA = 'observed-native-usage-stop@1'
MAX_CALLBACKS = 4
MAX_RECEIPT_BYTES = 65_536
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z')
_MODEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def validate_observed_native_token_threshold(value):
    """Validate the optional owner setting before provider or tokenizer work."""
    if value is not None and (type(value) is not int or not 1 <= value <= MAX_COUNTER):
        raise ValueError('positive bounded observed native token threshold required')
    return value


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class ObservedNativeUsageBudget:
    """One owner-controlled dialogue/provider/model/workspace callback guard.

    ``provider_callback_dispatches`` counts calls into the wrapped callable,
    not independently verified native processes or paid model requests.
    Original underlying exceptions are re-raised unchanged, including when
    accounting is unavailable. Public observations are compact independent
    copies; full retained receipts/projections stay private to this guard.
    """

    def __init__(self, invoke, *, provider, model, workspace,
                 max_observed_native_tokens_before_next_dispatch):
        threshold = validate_observed_native_token_threshold(
            max_observed_native_tokens_before_next_dispatch)
        if (threshold is None or not callable(invoke) or provider not in ('codex_cli', 'grok_cli')
                or type(model) is not str or _MODEL.fullmatch(model) is None):
            raise ValueError('explicit bound provider callback and threshold required')
        root = Path(workspace).absolute()
        if root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError('canonical existing callback workspace required')
        self._invoke, self._provider, self._model = invoke, provider, model
        self._workspace, self._threshold = root, threshold
        self._lock = threading.RLock()
        self._in_flight = False
        self._session = None
        self._next_generation = 0
        self._seen = set()
        self._receipts = []
        self._unknown = 0
        self._receipt_omitted = 0
        self._projection = None
        self._projection_sha256 = None
        self._observed_total = None
        self._usage_scope = None
        self._known = False
        self._callback_attempts = self._dispatches = self._denied = 0
        self._decision, self._reason = 'not_checked', 'first_dispatch'
        self._decision_basis = None
        self._blocked = None

    def _observation(self):
        projection = self._projection
        return {'schema':SCHEMA, 'threshold':self._threshold,
            'provider':self._provider, 'model':self._model,
            'workspace_binding_sha256':_sha(str(self._workspace)),
            'session_binding_sha256':_sha(self._session) if self._session is not None else None,
            'callback_attempts':self._callback_attempts,
            'provider_callback_dispatches':self._dispatches, 'denied_callbacks':self._denied,
            'decision':self._decision, 'reason':self._reason, 'decision_basis_total':self._decision_basis,
            'scoped_observed_total':self._observed_total, 'usage_scope':self._usage_scope,
            'known_for_continuing':self._known, 'projection_sha256':self._projection_sha256,
            'projection_observations_received':projection['observations_received'] if projection else 0,
            'projection_issue_count':len(projection['issues']) if projection else 0,
            'projection_omitted_issue_count':projection['omitted_issue_count'] if projection else 0,
            'prior_receipt_count':len(self._receipts), 'prior_unknown_receipt_count':self._unknown,
            'receipt_observation_omitted_count':self._receipt_omitted,
            'first_dispatch_may_overshoot':True, 'hidden_native_requests_capped':False,
            'native_cumulative_cap_enforced':False, 'task_attempt_total':None,
            'campaign_usage_complete':None, 'billing_total_verified':False,
            'completion_authority':False, 'execution_authority':False}

    def observation(self):
        with self._lock:
            return json.loads(_encoded(self._observation()))

    def _refresh(self):
        self._known = False
        self._observed_total = self._usage_scope = None
        self._projection = self._projection_sha256 = None
        try:
            projection = reconcile_native_usage_observations(self._receipts)
            self._projection = projection
            self._projection_sha256 = hashlib.sha256(_encoded(projection)).hexdigest()
            if self._unknown or projection['issues'] or projection['omitted_issue_count']:
                return
            if self._provider == 'codex_cli':
                native = projection['codex_cumulative']
                total = native['provider_total_tokens']
                known = (native['supplied_receipt_usage_complete'] is True
                    and native['unique_invocation_ids'] == self._dispatches
                    and not projection['grok_final_envelopes']['rows'])
                scope = 'codex_native_session_cumulative'
            else:
                native = projection['grok_final_envelopes']
                total = native['diagnostic_subtotals']['normalized_total_tokens']
                known = (native['diagnostic_subtotal_available'] is True
                    and len(native['rows']) == self._dispatches
                    and all(row['observed_native_scope_complete'] is True for row in native['rows'])
                    and projection['codex_cumulative']['observations_received'] == 0)
                scope = 'grok_native_final_envelope'
            # Keep numeric diagnostics even when completeness is unknown.
            if type(total) is int and 0 <= total <= MAX_COUNTER:
                self._observed_total, self._usage_scope = total, scope
                self._known = bool(known)
            elif self._provider == 'codex_cli':
                subtotal = native['known_provider_token_subtotal']
                if type(subtotal) is int and 0 <= subtotal <= MAX_COUNTER:
                    self._observed_total, self._usage_scope = subtotal, scope
        except Exception:
            # A malformed observation cannot mask an underlying provider error
            # or create permission to dispatch again.
            self._known = False

    def _retain(self, receipt, call):
        try:
            from .router_implementation_runner import render_model_prompt
            if type(receipt) is not dict:
                raise ValueError('provider receipt mapping required')
            raw = _encoded(receipt)
            if len(raw) > MAX_RECEIPT_BYTES or len(self._receipts) >= MAX_CALLBACKS:
                raise ValueError('bounded provider receipt required')
            held = json.loads(raw)
            self._receipts.append(held)
            rendered, _ = render_model_prompt(
                prompt=call.prompt, purpose='coding', workspace=self._workspace)
            expected = {'schema':'router-semantic-decision-provider@1',
                'call_id':call.call_id, 'session_id':call.session_id, 'generation':call.generation,
                'binding_sha256':call.binding_sha256, 'provider':self._provider, 'model':self._model,
                'workspace':str(self._workspace), 'prompt_sha256':_sha(call.prompt),
                'prompt_bytes':len(call.prompt.encode('utf-8')),
                'model_prompt_sha256':call.model_prompt_sha256,
                'model_prompt_bytes':len(rendered.encode('utf-8'))}
            if (not all(type(held.get(key)) is type(value) and held.get(key) == value
                        for key, value in expected.items())
                    or _sha(rendered) != call.model_prompt_sha256
                    or held.get('provider_called') is not True):
                self._unknown += 1
        except Exception:
            self._unknown += 1
        self._refresh()

    def _denial(self, call, reason):
        from .router_implementation_runner import render_model_prompt
        from .semantic_decision_dialogue import SemanticDecisionProviderFailure
        self._decision, self._reason = 'deny', reason
        self._decision_basis = self._observed_total
        self._denied = min(MAX_COUNTER, self._denied + 1)
        self._blocked = reason
        rendered, _ = render_model_prompt(prompt=call.prompt, purpose='coding', workspace=self._workspace)
        receipt = {'schema':'router-semantic-decision-provider@1',
            'call_id':call.call_id, 'session_id':call.session_id, 'generation':call.generation,
            'binding_sha256':call.binding_sha256, 'provider':call.provider, 'model':call.model,
            'workspace':str(self._workspace), 'prompt_sha256':_sha(call.prompt),
            'prompt_bytes':len(call.prompt.encode('utf-8')), 'model_prompt_sha256':call.model_prompt_sha256,
            'model_prompt_bytes':len(rendered.encode('utf-8')), 'status':'local_refusal',
            'provider_called':False, 'usage':None, 'usage_status':'unknown',
            'native_input_delivered':False, 'provider_output_token_cap_enforced':False,
            'observed_native_usage_stop':self._observation(), 'completion_authority':False}
        return SemanticDecisionProviderFailure('observed_native_usage_stop', receipt=receipt)

    def __call__(self, call):
        from .semantic_decision_dialogue import (
            SemanticDecisionCall, SemanticDecisionProviderFailure, SemanticDecisionReply,
        )
        if type(call) is not SemanticDecisionCall:
            raise ValueError('typed semantic decision call required')
        try:
            valid = (all(type(value) is str and _ID.fullmatch(value) is not None
                         for value in (call.call_id, call.session_id))
                and type(call.generation) is int and 0 <= call.generation < MAX_CALLBACKS
                and type(call.binding_sha256) is str and _SHA.fullmatch(call.binding_sha256) is not None
                and type(call.model_prompt_sha256) is str and _SHA.fullmatch(call.model_prompt_sha256) is not None
                and type(call.provider) is str and call.provider in ('codex_cli', 'grok_cli')
                and type(call.model) is str and _MODEL.fullmatch(call.model) is not None
                and type(call.prompt) is str and 0 < len(call.prompt.encode('utf-8')) <= 256000)
        except UnicodeError:
            valid = False
        if not valid:
            raise ValueError('bounded bound semantic decision call required')
        with self._lock:
            self._callback_attempts = min(MAX_COUNTER, self._callback_attempts + 1)
            if self._in_flight:
                raise self._denial(call, 'concurrent_dispatch_refused')
            if (self._blocked is not None):
                raise self._denial(call, self._blocked)
            if (call.provider != self._provider or call.model != self._model
                    or Path.cwd() != self._workspace or call.call_id in self._seen
                    or call.generation != self._next_generation
                    or self._session is not None and call.session_id != self._session):
                raise self._denial(call, 'binding_or_replay_refused')
            if self._dispatches:
                self._refresh()
                if not self._known:
                    raise self._denial(call, 'prior_usage_unknown_or_conflicting')
                if self._observed_total >= self._threshold:
                    raise self._denial(call, 'threshold_reached')
            self._decision, self._reason = 'allow', 'first_dispatch' if not self._dispatches else 'below_threshold'
            self._decision_basis = self._observed_total
            self._session = call.session_id
            self._seen.add(call.call_id)
            self._next_generation += 1
            self._dispatches += 1
            self._in_flight = True
        try:
            reply = self._invoke(call)
        except BaseException as error:
            with self._lock:
                try:
                    receipt = error.receipt if isinstance(error, SemanticDecisionProviderFailure) else None
                    self._retain(receipt, call)
                except BaseException:
                    self._unknown += 1
                    self._known = False
                finally:
                    self._in_flight = False
            raise
        with self._lock:
            try:
                self._retain(reply.receipt if type(reply) is SemanticDecisionReply else None, call)
            finally:
                self._in_flight = False
            if type(reply) is not SemanticDecisionReply:
                return reply
            # Preserve original text and receipt fields without mutating either.
            try:
                receipt = json.loads(_encoded(reply.receipt))
                original = dict(receipt)
                receipt['observed_native_usage_stop'] = self._observation()
                if len(_encoded(receipt)) > MAX_RECEIPT_BYTES:
                    self._receipt_omitted += 1
                    return SemanticDecisionReply(reply.text, original)
                return SemanticDecisionReply(reply.text, receipt)
            except Exception:
                return reply
