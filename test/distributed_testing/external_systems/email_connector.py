#!/usr/bin/env python3
"""
Email Connector for Distributed Testing Framework

This module provides a connector for sending email notifications from the distributed testing framework
using SMTP with optional TLS/SSL support.
"""

import anyio
import logging
import smtplib
import os
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formatdate
from typing import Dict, List, Any, Optional, Union, Tuple

# Import the standardized interface
from .external_systems.api_interface import (
    ExternalSystemInterface,
    ConnectorCapabilities,
    ExternalSystemResult,
    ExternalSystemFactory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailConnector(ExternalSystemInterface):
    """
    Connector for sending email notifications.
    
    This connector implements the standardized ExternalSystemInterface for email
    communication and provides methods for sending emails with attachments.
    """
    
    def __init__(self):
        """
        Initialize the Email connector.
        """
        self.smtp_server = None
        self.smtp_port = None
        self.username = None
        self.password = None
        self.use_tls = False
        self.use_ssl = False
        self.default_sender = None
        self.default_recipients = []
        self.rate_limit = 60  # Default: 60 emails per minute
        self.rate_limit_sleep = 1.0  # Seconds to sleep when rate limited
        self.last_sent_time = 0
        self.sent_count = 0
        self.connected = False
        self.smtp = None
        
        # Cache for message templates
        self.templates = {}
        
        # Capabilities
        self.capabilities = ConnectorCapabilities(
            supports_create=True,
            supports_update=False,  # Emails can't be updated after sending
            supports_delete=False,  # Emails can't be deleted after sending
            supports_query=False,   # Can't query sent emails
            supports_batch_operations=True,
            supports_attachments=True,
            supports_comments=False,
            supports_custom_fields=False,
            supports_relationships=False,
            supports_history=False,
            item_types=["email"],
            query_operators=[],
            max_batch_size=50,
            rate_limit=60,  # 60 emails per minute
            supports_html=True,    # Email specific capability
            supports_plain_text=True,  # Email specific capability
            supports_attachments_size_limit=10_000_000  # 10MB default attachment limit
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Email connector with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - smtp_server: SMTP server address
                   - smtp_port: SMTP server port
                   - username: SMTP username (optional)
                   - password: SMTP password (optional)
                   - use_tls: Whether to use TLS (optional, default False)
                   - use_ssl: Whether to use SSL (optional, default False)
                   - default_sender: Default sender email address (optional)
                   - default_recipients: Default recipient email addresses (optional)
                   - templates_dir: Directory containing email templates (optional)
            
        Returns:
            True if initialization succeeded
        """
        self.smtp_server = config.get("smtp_server")
        self.smtp_port = config.get("smtp_port")
        self.username = config.get("username")
        self.password = config.get("password")
        self.use_tls = config.get("use_tls", False)
        self.use_ssl = config.get("use_ssl", False)
        self.default_sender = config.get("default_sender")
        self.default_recipients = config.get("default_recipients", [])
        self.rate_limit = config.get("rate_limit", self.rate_limit)
        
        # Optional: load email templates
        templates_dir = config.get("templates_dir")
        if templates_dir and os.path.isdir(templates_dir):
            self._load_templates(templates_dir)
        
        if not self.smtp_server:
            logger.error("SMTP server is required")
            return False
        
        if not self.smtp_port:
            logger.error("SMTP port is required")
            return False
        
        if self.use_tls and self.use_ssl:
            logger.error("Cannot use both TLS and SSL at the same time")
            return False
        
        logger.info(f"EmailConnector initialized for server {self.smtp_server}:{self.smtp_port}")
        return True
    
    def _load_templates(self, templates_dir: str) -> None:
        """
        Load email templates from a directory.
        
        Args:
            templates_dir: Directory containing template files
        """
        try:
            for filename in os.listdir(templates_dir):
                if filename.endswith(".html") or filename.endswith(".txt"):
                    template_name = os.path.splitext(filename)[0]
                    with open(os.path.join(templates_dir, filename), 'r') as f:
                        content = f.read()
                        self.templates[template_name] = content
                        logger.debug(f"Loaded email template: {template_name}")
        except Exception as e:
            logger.warning(f"Error loading email templates: {str(e)}")
    
    async def _ensure_connection(self):
        """Ensure SMTP connection is established."""
        if not self.connected or self.smtp is None:
            await self.connect()
    
    async def connect(self) -> bool:
        """
        Establish connection to SMTP server.
        
        Returns:
            True if connection succeeded
        """
        try:
            # Since SMTP operations are blocking, run them in a separate thread
            loop = # TODO: Remove event loop management - anyio
            result = await loop.run_in_executor(
                None, self._connect_sync
            )
            self.connected = result
            return result
        except Exception as e:
            logger.error(f"Exception connecting to SMTP server: {str(e)}")
            self.connected = False
            return False
    
    def _connect_sync(self) -> bool:
        """Synchronous implementation of SMTP connection."""
        try:
            if self.use_ssl:
                self.smtp = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                self.smtp = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            if self.use_tls:
                self.smtp.starttls()
            
            if self.username and self.password:
                self.smtp.login(self.username, self.password)
            
            logger.info(f"Connected to SMTP server at {self.smtp_server}:{self.smtp_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SMTP server: {str(e)}")
            if self.smtp:
                try:
                    self.smtp.quit()
                except:
                    pass
                self.smtp = None
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to the SMTP server.
        
        Returns:
            True if connected
        """
        if not self.smtp:
            return False
        
        try:
            # Since SMTP operations are blocking, run them in a separate thread
            loop = # TODO: Remove event loop management - anyio
            result = await loop.run_in_executor(
                None, lambda: self.smtp.noop()[0] == 250
            )
            self.connected = result
            return result
        except Exception:
            self.connected = False
            self.smtp = None
            return False
    
    async def _handle_rate_limit(self):
        """Handle rate limiting to avoid exceeding email sending limits."""
        # Implement a simple token bucket algorithm for rate limiting
        self.sent_count += 1
        if self.sent_count >= self.rate_limit:
            # Reset counter and wait before allowing more sends
            wait_time = self.rate_limit_sleep
            logger.info(f"Rate limit reached, pausing for {wait_time} seconds")
            await anyio.sleep(wait_time)
            self.sent_count = 0
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation for email sending.
        
        Args:
            operation: The operation to execute
            params: Parameters for the operation
            
        Returns:
            Dictionary with operation result
        """
        await self._ensure_connection()
        
        try:
            result = ExternalSystemResult(
                success=False,
                operation=operation,
                error_message="Operation not implemented",
                error_code="NOT_IMPLEMENTED"
            )
            
            # Map operations to email functions
            if operation == "send_email":
                result = await self._send_email(params)
            elif operation == "send_batch_emails":
                result = await self._send_batch_emails(params)
            elif operation == "send_template_email":
                result = await self._send_template_email(params)
            elif operation == "validate_email":
                result = await self._validate_email(params)
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Exception executing email operation {operation}: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation=operation,
                error_message=str(e),
                error_code="EXCEPTION"
            ).to_dict()
    
    async def _send_email(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send an email.
        
        Args:
            params: Parameters for sending an email
            
        Returns:
            ExternalSystemResult with send operation result
        """
        # Get required parameters
        recipients = params.get("recipients", self.default_recipients)
        subject = params.get("subject", "")
        body = params.get("body", "")
        html_body = params.get("html_body")
        sender = params.get("sender", self.default_sender)
        cc = params.get("cc", [])
        bcc = params.get("bcc", [])
        attachments = params.get("attachments", [])
        
        if not recipients:
            return ExternalSystemResult(
                success=False,
                operation="send_email",
                error_message="Recipients list is required",
                error_code="MISSING_RECIPIENTS"
            )
        
        if not sender:
            return ExternalSystemResult(
                success=False,
                operation="send_email",
                error_message="Sender is required",
                error_code="MISSING_SENDER"
            )
        
        # Validate email addresses
        for email in [sender] + recipients + cc + bcc:
            if not self._is_valid_email(email):
                return ExternalSystemResult(
                    success=False,
                    operation="send_email",
                    error_message=f"Invalid email address: {email}",
                    error_code="INVALID_EMAIL"
                )
        
        # Prepare the email message
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg["Date"] = formatdate(localtime=True)
        
        if cc:
            msg["Cc"] = ", ".join(cc)
        
        # Add plain text body
        if body:
            msg.attach(MIMEText(body, "plain"))
        
        # Add HTML body if provided
        if html_body:
            msg.attach(MIMEText(html_body, "html"))
        
        # Add attachments if provided
        for attachment in attachments:
            if "path" in attachment:
                # Attach file from path
                path = attachment["path"]
                try:
                    with open(path, "rb") as f:
                        part = MIMEApplication(f.read(), Name=os.path.basename(path))
                        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
                        msg.attach(part)
                except Exception as e:
                    return ExternalSystemResult(
                        success=False,
                        operation="send_email",
                        error_message=f"Failed to attach file {path}: {str(e)}",
                        error_code="ATTACHMENT_ERROR"
                    )
            elif "content" in attachment and "name" in attachment:
                # Attach from provided content
                content = attachment["content"]
                name = attachment["name"]
                try:
                    part = MIMEApplication(content.encode() if isinstance(content, str) else content)
                    part["Content-Disposition"] = f'attachment; filename="{name}"'
                    msg.attach(part)
                except Exception as e:
                    return ExternalSystemResult(
                        success=False,
                        operation="send_email",
                        error_message=f"Failed to create attachment {name}: {str(e)}",
                        error_code="ATTACHMENT_ERROR"
                    )
        
        try:
            # Handle rate limiting
            await self._handle_rate_limit()
            
            # Since SMTP operations are blocking, run them in a separate thread
            loop = # TODO: Remove event loop management - anyio
            
            # Get all recipients for sending
            all_recipients = recipients + cc + bcc
            
            await loop.run_in_executor(
                None, lambda: self.smtp.sendmail(sender, all_recipients, msg.as_string())
            )
            
            logger.info(f"Email sent to {len(all_recipients)} recipients with subject: {subject}")
            
            return ExternalSystemResult(
                success=True,
                operation="send_email",
                result_data={
                    "recipients": recipients,
                    "cc": cc,
                    "bcc": bcc,
                    "subject": subject,
                    "sender": sender,
                    "attachments_count": len(attachments)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            
            # Check if we need to reconnect
            if "not connected" in str(e).lower() or "connection refused" in str(e).lower():
                self.connected = False
                self.smtp = None
            
            return ExternalSystemResult(
                success=False,
                operation="send_email",
                error_message=f"Failed to send email: {str(e)}",
                error_code="SEND_ERROR"
            )
    
    async def _send_batch_emails(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send multiple emails in a batch.
        
        Args:
            params: Parameters for batch email sending
            
        Returns:
            ExternalSystemResult with batch operation result
        """
        emails = params.get("emails", [])
        
        if not emails:
            return ExternalSystemResult(
                success=False,
                operation="send_batch_emails",
                error_message="Emails array is required",
                error_code="MISSING_EMAILS"
            )
        
        results = []
        success_count = 0
        failure_count = 0
        
        for email in emails:
            result = await self._send_email(email)
            results.append({
                "success": result.success,
                "recipients": email.get("recipients"),
                "subject": email.get("subject"),
                "error": None if result.success else result.error_message
            })
            
            if result.success:
                success_count += 1
            else:
                failure_count += 1
        
        return ExternalSystemResult(
            success=success_count > 0,
            operation="send_batch_emails",
            result_data={
                "total": len(emails),
                "success_count": success_count,
                "failure_count": failure_count,
                "results": results
            }
        )
    
    async def _send_template_email(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send an email using a template.
        
        Args:
            params: Parameters for template-based email
            
        Returns:
            ExternalSystemResult with operation result
        """
        template_name = params.get("template_name")
        template_data = params.get("template_data", {})
        
        if not template_name:
            return ExternalSystemResult(
                success=False,
                operation="send_template_email",
                error_message="Template name is required",
                error_code="MISSING_TEMPLATE"
            )
        
        if template_name not in self.templates:
            return ExternalSystemResult(
                success=False,
                operation="send_template_email",
                error_message=f"Template not found: {template_name}",
                error_code="TEMPLATE_NOT_FOUND"
            )
        
        # Apply template variables
        template_content = self.templates[template_name]
        body = self._apply_template_variables(template_content, template_data)
        
        # Prepare email parameters
        email_params = {
            "recipients": params.get("recipients", self.default_recipients),
            "subject": params.get("subject", ""),
            "sender": params.get("sender", self.default_sender),
            "cc": params.get("cc", []),
            "bcc": params.get("bcc", []),
            "attachments": params.get("attachments", [])
        }
        
        # Determine if this is HTML or plain text
        if template_name.endswith("_html") or "<html" in template_content.lower():
            email_params["html_body"] = body
        else:
            email_params["body"] = body
        
        # Send the email
        return await self._send_email(email_params)
    
    def _apply_template_variables(self, template: str, data: Dict[str, Any]) -> str:
        """
        Apply variable substitution to a template.
        
        Args:
            template: Template string
            data: Dictionary of variable values
            
        Returns:
            Template with variables replaced
        """
        for key, value in data.items():
            # Simple variable substitution using {{variable}} format
            template = template.replace("{{" + key + "}}", str(value))
        return template
    
    async def _validate_email(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Validate an email address.
        
        Args:
            params: Parameters with email address
            
        Returns:
            ExternalSystemResult with validation result
        """
        email = params.get("email")
        
        if not email:
            return ExternalSystemResult(
                success=False,
                operation="validate_email",
                error_message="Email address is required",
                error_code="MISSING_EMAIL"
            )
        
        is_valid = self._is_valid_email(email)
        
        return ExternalSystemResult(
            success=True,
            operation="validate_email",
            result_data={
                "email": email,
                "is_valid": is_valid
            }
        )
    
    def _is_valid_email(self, email: str) -> bool:
        """
        Validate an email address using regex.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email is valid
        """
        # Use a simplified regex pattern for email validation
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query is not supported for email.
        
        Args:
            query_params: Query parameters
            
        Returns:
            Empty list (email connector doesn't support queries)
        """
        logger.warning("Query operation is not supported for email connector")
        return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in the Email system (send an email).
        
        Args:
            item_type: Type of item to create (must be "email")
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type != "email":
            raise Exception(f"Unsupported item type: {item_type}. Only 'email' is supported.")
        
        result = await self._send_email(item_data)
        
        if result.success:
            return result.result_data
        else:
            raise Exception(f"Failed to send email: {result.error_message}")
    
    async def update_item(self, item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update is not supported for email.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            Always False (email connector doesn't support updates)
        """
        logger.warning("Update operation is not supported for email connector")
        return False
    
    async def delete_item(self, item_type: str, item_id: str) -> bool:
        """
        Delete is not supported for email.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            Always False (email connector doesn't support deletion)
        """
        logger.warning("Delete operation is not supported for email connector")
        return False
    
    async def get_item(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """
        Get item is not supported for email.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Empty dictionary (email connector doesn't support item retrieval)
        """
        logger.warning("Get item operation is not supported for email connector")
        return {}
    
    async def system_info(self) -> Dict[str, Any]:
        """
        Get information about the email system.
        
        Returns:
            Dictionary with system information
        """
        try:
            is_connected = await self.is_connected()
            
            return {
                "system_type": "email",
                "connected": is_connected,
                "smtp_server": self.smtp_server,
                "smtp_port": self.smtp_port,
                "use_tls": self.use_tls,
                "use_ssl": self.use_ssl,
                "username": self.username,
                "default_sender": self.default_sender,
                "default_recipients_count": len(self.default_recipients),
                "templates_count": len(self.templates),
                "rate_limit": self.rate_limit,
                "capabilities": self.capabilities.to_dict()
            }
        except Exception as e:
            logger.error(f"Exception getting email system info: {str(e)}")
            
            return {
                "system_type": "email",
                "connected": False,
                "smtp_server": self.smtp_server,
                "smtp_port": self.smtp_port,
                "error": str(e),
                "capabilities": self.capabilities.to_dict()
            }
    
    async def close(self) -> None:
        """
        Close the connection to the SMTP server and clean up resources.
        
        Returns:
            None
        """
        if self.smtp:
            try:
                # Since SMTP operations are blocking, run them in a separate thread
                loop = # TODO: Remove event loop management - anyio
                await loop.run_in_executor(None, self.smtp.quit)
            except Exception as e:
                logger.warning(f"Error closing SMTP connection: {str(e)}")
            finally:
                self.smtp = None
                self.connected = False
                logger.info("Email SMTP connection closed")


# Register with factory
ExternalSystemFactory.register_connector("email", EmailConnector)