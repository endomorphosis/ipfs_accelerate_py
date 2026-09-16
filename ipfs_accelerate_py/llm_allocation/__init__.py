"""DuckDB-backed LLM allocation: observe CLI/API health, spend, and limits.

Cold import does not open DuckDB or contact providers.
"""

from .allocator import rank_provider_names, score_provider
from .api_key_slots import (
    bind_session_api_key,
    fingerprint_api_key,
    list_api_key_slots,
    register_api_key,
    select_api_key,
)
from .cli_status import cli_tools_status, render_cli_tools_status
from .intelligence_index import (
    CATALOG_REVISION,
    INDEX_VERSION,
    discover_available_providers,
    intelligence_cost_matrix,
    load_intelligence_index_models,
    select_efficient_model,
    select_efficient_route,
)
from .model_manager_sync import (
    populate_cli_models,
    populate_intelligence_index,
    populate_router_catalog,
)
from .session_route import (
    choose_cli_route,
    migrate_cli_session,
    resolve_session,
    resume_kwargs_for_session,
)
from .duckdb_store import (
    AllocationStore,
    default_allocation_db_path,
    get_allocation_store,
    llm_allocate_enabled,
    llm_observe_enabled,
    reset_allocation_store,
    sanitize_session_metadata,
)
from .limits import PROVIDER_LIMIT_HINTS, ProviderLimitHint, limit_hint_for
from .observations import (
    CallErrorKind,
    CallFailure,
    CallObservation,
    CallProtocol,
    classify_provider_failure,
    observation_from_exception,
    protocol_for_provider,
)
from .paths import (
    API_PROVIDERS,
    CLI_PROVIDERS,
    CLI_SESSION_CONTRACT,
    PROVIDER_PATH_METADATA,
    CliResumeStyle,
    RoutingPath,
    cli_session_contract,
    filter_names_for_path,
    inject_cli_session_kwargs,
    native_session_from_kwargs,
    path_for_provider,
    provider_path_metadata,
    providers_for_path,
)

__all__ = [
    "API_PROVIDERS",
    "AllocationStore",
    "CLI_PROVIDERS",
    "CLI_SESSION_CONTRACT",
    "CliResumeStyle",
    "CallErrorKind",
    "CallFailure",
    "CallObservation",
    "CallProtocol",
    "PROVIDER_LIMIT_HINTS",
    "PROVIDER_PATH_METADATA",
    "ProviderLimitHint",
    "RoutingPath",
    "bind_session_api_key",
    "CATALOG_REVISION",
    "INDEX_VERSION",
    "choose_cli_route",
    "cli_tools_status",
    "fingerprint_api_key",
    "list_api_key_slots",
    "register_api_key",
    "select_api_key",
    "classify_provider_failure",
    "cli_session_contract",
    "default_allocation_db_path",
    "filter_names_for_path",
    "inject_cli_session_kwargs",
    "native_session_from_kwargs",
    "get_allocation_store",
    "limit_hint_for",
    "migrate_cli_session",
    "llm_allocate_enabled",
    "llm_observe_enabled",
    "observation_from_exception",
    "path_for_provider",
    "protocol_for_provider",
    "provider_path_metadata",
    "providers_for_path",
    "rank_provider_names",
    "resolve_session",
    "render_cli_tools_status",
    "reset_allocation_store",
    "resume_kwargs_for_session",
    "sanitize_session_metadata",
    "score_provider",
    "discover_available_providers",
    "intelligence_cost_matrix",
    "load_intelligence_index_models",
    "populate_cli_models",
    "populate_intelligence_index",
    "populate_router_catalog",
    "select_efficient_model",
    "select_efficient_route",
]
