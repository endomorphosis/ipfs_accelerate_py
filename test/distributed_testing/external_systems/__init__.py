"""
External Systems Integration modules for Distributed Testing Framework

This package provides connectors for interacting with various external systems:
- JIRA
- Slack
- Discord
- Telegram
- Email
- MS Teams

These connectors enable the distributed testing framework to interact with external
systems for issue tracking, notifications, test management, and more.

The package features a standardized API interface to ensure consistent behavior across
different systems and make it easy to add new connectors.
"""

# Import standardized interface
from .api_interface import (
    ExternalSystemInterface, 
    ConnectorCapabilities,
    ExternalSystemResult,
    ExternalSystemFactory
)

# Import implementation classes
from .jira_connector import JiraConnector
from .slack_connector import SlackConnector
from .discord_connector import DiscordConnector
from .telegram_connector import TelegramConnector

# Export key classes for easy import
__all__ = [
    "ExternalSystemInterface",
    "ConnectorCapabilities",
    "ExternalSystemResult",
    "ExternalSystemFactory",
    "JiraConnector",
    "SlackConnector",
    "DiscordConnector",
    "TelegramConnector"
]