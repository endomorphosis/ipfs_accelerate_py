"""Native unified admin tools for mcp_server."""

from .native_admin_tools import (
	cleanup_resources,
	configure_system,
	get_system_status,
	manage_endpoints,
	manage_service,
	register_native_admin_tools,
	system_health,
	system_maintenance,
	update_configuration,
)

__all__ = [
	"manage_endpoints",
	"system_maintenance",
	"configure_system",
	"system_health",
	"get_system_status",
	"manage_service",
	"update_configuration",
	"cleanup_resources",
	"register_native_admin_tools",
]
