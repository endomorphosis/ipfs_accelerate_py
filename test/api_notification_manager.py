#!/usr/bin/env python
"""
API Notification Manager

This module provides notification capabilities for the API Anomaly Detection system,
allowing automatic alerts when performance issues or unexpected API behavior is detected.

Features:
1. Multiple notification channels (email, webhook, log, callback)
2. Configurable notification rules based on severity and anomaly type
3. Rate limiting to prevent notification storms
4. Templated messages with customizable formatting
5. Notification history tracking

Usage:
    Import this module into the API monitoring dashboard for automatic notifications.
"""

import os
import sys
import time
import logging
import json
import smtplib
import ssl
import requests
import threading
import queue
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_notification_manager")

# Import anomaly detection types if available
try:
    from api_anomaly_detection import AnomalySeverity, AnomalyType, NotificationChannel
    ANOMALY_TYPES_AVAILABLE = True
except ImportError:
    logger.warning("Anomaly detection types not available, using local definitions")
    ANOMALY_TYPES_AVAILABLE = False
    
    # Local definitions if imports fail
    class AnomalySeverity(Enum):
        """Severity levels for detected anomalies."""
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class AnomalyType(Enum):
        """Types of anomalies that can be detected."""
        LATENCY_SPIKE = "latency_spike"
        THROUGHPUT_DROP = "throughput_drop"
        ERROR_RATE_INCREASE = "error_rate_increase"
        COST_SPIKE = "cost_spike"
        PATTERN_CHANGE = "pattern_change"
        SEASONAL_DEVIATION = "seasonal_deviation"
        TREND_BREAK = "trend_break"
        PERSISTENT_DEGRADATION = "persistent_degradation"
        OSCILLATION = "oscillation"
    
    class NotificationChannel(Enum):
        """Supported notification channels."""
        EMAIL = "email"
        WEBHOOK = "webhook"
        LOG = "log"
        CALLBACK = "callback"


class NotificationRule:
    """
    Rule for determining when to send notifications.
    
    A rule consists of conditions (severity, anomaly types, APIs)
    and a list of notification channels to use when conditions are met.
    """
    
    def __init__(
        self,
        name: str,
        min_severity: AnomalySeverity = AnomalySeverity.MEDIUM,
        anomaly_types: Optional[List[AnomalyType]] = None,
        apis: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        channels: Optional[List[NotificationChannel]] = None,
        cooldown_minutes: int = 15
    ):
        """
        Initialize a notification rule.
        
        Args:
            name: Name of the rule
            min_severity: Minimum severity level to trigger notification
            anomaly_types: List of anomaly types to trigger notification (None = all)
            apis: List of APIs to monitor (None = all)
            metrics: List of metrics to monitor (None = all)
            channels: List of notification channels to use
            cooldown_minutes: Minimum time between notifications from this rule
        """
        self.name = name
        self.min_severity = min_severity
        self.anomaly_types = anomaly_types
        self.apis = apis
        self.metrics = metrics
        self.channels = channels or [NotificationChannel.LOG]
        self.cooldown_minutes = cooldown_minutes
        
        # Last notification time for cooldown
        self.last_notification_time: Dict[str, float] = {}
    
    def matches(self, anomaly: Dict[str, Any]) -> bool:
        """
        Check if an anomaly matches this rule.
        
        Args:
            anomaly: The anomaly to check
            
        Returns:
            True if the rule matches the anomaly
        """
        # Check severity
        severity = anomaly.get("severity", "low")
        severity_order = {
            AnomalySeverity.LOW.value: 0,
            AnomalySeverity.MEDIUM.value: 1,
            AnomalySeverity.HIGH.value: 2,
            AnomalySeverity.CRITICAL.value: 3
        }
        
        min_severity_level = severity_order.get(self.min_severity.value, 0)
        anomaly_severity_level = severity_order.get(severity, 0)
        
        if anomaly_severity_level < min_severity_level:
            return False
        
        # Check anomaly type
        if self.anomaly_types:
            anomaly_type = anomaly.get("anomaly_type")
            if not anomaly_type or anomaly_type not in [t.value for t in self.anomaly_types]:
                return False
        
        # Check API
        if self.apis:
            api = anomaly.get("api")
            if not api or api not in self.apis:
                return False
        
        # Check metric
        if self.metrics:
            metric = anomaly.get("metric_type")
            if not metric or metric not in self.metrics:
                return False
        
        # Check cooldown
        now = time.time()
        key = f"{anomaly.get('api', 'unknown')}:{anomaly.get('metric_type', 'unknown')}"
        last_time = self.last_notification_time.get(key, 0)
        cooldown_seconds = self.cooldown_minutes * 60
        
        if now - last_time < cooldown_seconds:
            logger.debug(f"Rule {self.name} in cooldown for {key}")
            return False
        
        # Update last notification time for this API/metric
        self.last_notification_time[key] = now
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert rule to dictionary for serialization.
        
        Returns:
            Dictionary representation of the rule
        """
        return {
            "name": self.name,
            "min_severity": self.min_severity.value,
            "anomaly_types": [t.value for t in self.anomaly_types] if self.anomaly_types else None,
            "apis": self.apis,
            "metrics": self.metrics,
            "channels": [c.value for c in self.channels],
            "cooldown_minutes": self.cooldown_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotificationRule':
        """
        Create rule from dictionary.
        
        Args:
            data: Dictionary representation of the rule
            
        Returns:
            NotificationRule instance
        """
        # Convert string values to enums
        min_severity = AnomalySeverity(data.get("min_severity", "medium"))
        
        anomaly_types = None
        if "anomaly_types" in data and data["anomaly_types"]:
            anomaly_types = [AnomalyType(t) for t in data["anomaly_types"]]
        
        channels = [NotificationChannel(c) for c in data.get("channels", ["log"])]
        
        return cls(
            name=data.get("name", "Unnamed Rule"),
            min_severity=min_severity,
            anomaly_types=anomaly_types,
            apis=data.get("apis"),
            metrics=data.get("metrics"),
            channels=channels,
            cooldown_minutes=data.get("cooldown_minutes", 15)
        )


class EmailConfig:
    """Email configuration for notifications."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        use_ssl: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_address: str = "api-monitor@example.com",
        to_addresses: List[str] = None,
        subject_prefix: str = "[API Alert]"
    ):
        """
        Initialize email configuration.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            use_ssl: Whether to use SSL for connection
            username: SMTP username (if required)
            password: SMTP password (if required)
            from_address: Sender email address
            to_addresses: List of recipient email addresses
            subject_prefix: Prefix for email subjects
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.use_ssl = use_ssl
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses or []
        self.subject_prefix = subject_prefix


class WebhookConfig:
    """Webhook configuration for notifications."""
    
    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Tuple[str, str]] = None,
        timeout: int = 10,
        retry_count: int = 3
    ):
        """
        Initialize webhook configuration.
        
        Args:
            url: Webhook URL
            method: HTTP method (POST, PUT)
            headers: HTTP headers
            auth: HTTP authentication (username, password)
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
        """
        self.url = url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}
        self.auth = auth
        self.timeout = timeout
        self.retry_count = retry_count


class NotificationHistory:
    """Tracks history of sent notifications."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize notification history.
        
        Args:
            max_history: Maximum number of notifications to keep in history
        """
        self.notifications: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add(
        self, 
        anomaly: Dict[str, Any], 
        rule_name: str, 
        channels: List[NotificationChannel],
        status: str = "sent"
    ) -> None:
        """
        Add a notification to history.
        
        Args:
            anomaly: The anomaly that triggered the notification
            rule_name: Name of the rule that matched
            channels: Channels the notification was sent to
            status: Status of the notification (sent, failed)
        """
        notification = {
            "timestamp": time.time(),
            "anomaly": anomaly,
            "rule_name": rule_name,
            "channels": [c.value for c in channels],
            "status": status
        }
        
        self.notifications.append(notification)
        
        # Trim history if needed
        if len(self.notifications) > self.max_history:
            self.notifications = self.notifications[-self.max_history:]
    
    def get_recent(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent notifications.
        
        Args:
            hours: Hours to look back
            
        Returns:
            List of recent notifications
        """
        cutoff_time = time.time() - (hours * 60 * 60)
        return [n for n in self.notifications if n["timestamp"] >= cutoff_time]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert history to dictionary for serialization.
        
        Returns:
            Dictionary representation of history
        """
        return {
            "notifications": self.notifications,
            "max_history": self.max_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotificationHistory':
        """
        Create history from dictionary.
        
        Args:
            data: Dictionary representation of history
            
        Returns:
            NotificationHistory instance
        """
        history = cls(max_history=data.get("max_history", 100))
        history.notifications = data.get("notifications", [])
        return history


class NotificationManager:
    """
    Manager for sending notifications based on detected anomalies.
    
    This class handles sending notifications through multiple channels
    based on configurable rules.
    """
    
    def __init__(
        self,
        rules: Optional[List[NotificationRule]] = None,
        email_config: Optional[EmailConfig] = None,
        webhook_config: Optional[WebhookConfig] = None,
        data_dir: str = "notification_data",
        enabled: bool = True
    ):
        """
        Initialize notification manager.
        
        Args:
            rules: List of notification rules
            email_config: Email configuration
            webhook_config: Webhook configuration
            data_dir: Directory to store notification data
            enabled: Whether notifications are enabled
        """
        self.rules = rules or []
        self.email_config = email_config
        self.webhook_config = webhook_config
        self.data_dir = data_dir
        self.enabled = enabled
        
        # Initialize default rule if none provided
        if not self.rules:
            self.rules.append(NotificationRule(
                name="Default Rule",
                min_severity=AnomalySeverity.HIGH,
                channels=[NotificationChannel.LOG]
            ))
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Notification history
        self.history = NotificationHistory()
        
        # Notification queue for async processing
        self.notification_queue: queue.Queue = queue.Queue()
        self.notification_thread = None
        self.running = False
        
        # Callback functions by notification type
        self.callbacks: Dict[str, List[Callable]] = {}
        
        logger.info(f"Notification manager initialized with {len(self.rules)} rules")
    
    def start(self) -> None:
        """Start notification processing thread."""
        if self.running:
            logger.warning("Notification manager already running")
            return
        
        self.running = True
        self.notification_thread = threading.Thread(
            target=self._process_notifications,
            daemon=True
        )
        self.notification_thread.start()
        logger.info("Notification manager started")
    
    def stop(self) -> None:
        """Stop notification processing thread."""
        if not self.running:
            logger.warning("Notification manager not running")
            return
        
        self.running = False
        if self.notification_thread:
            self.notification_thread.join(timeout=5)
        logger.info("Notification manager stopped")
    
    def add_rule(self, rule: NotificationRule) -> None:
        """
        Add a notification rule.
        
        Args:
            rule: The rule to add
        """
        self.rules.append(rule)
        logger.info(f"Added notification rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a notification rule by name.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                logger.info(f"Removed notification rule: {rule_name}")
                return True
        
        logger.warning(f"Rule not found: {rule_name}")
        return False
    
    def register_callback(self, callback_id: str, callback_fn: Callable) -> None:
        """
        Register a callback function for notifications.
        
        Args:
            callback_id: Unique identifier for the callback
            callback_fn: Callback function that accepts anomaly dict
        """
        if callback_id not in self.callbacks:
            self.callbacks[callback_id] = []
        
        self.callbacks[callback_id].append(callback_fn)
        logger.info(f"Registered callback: {callback_id}")
    
    def unregister_callback(self, callback_id: str) -> bool:
        """
        Unregister a callback function.
        
        Args:
            callback_id: Identifier of the callback to remove
            
        Returns:
            True if callback was removed
        """
        if callback_id in self.callbacks:
            del self.callbacks[callback_id]
            logger.info(f"Unregistered callback: {callback_id}")
            return True
        
        logger.warning(f"Callback not found: {callback_id}")
        return False
    
    def notify(self, anomaly: Dict[str, Any]) -> bool:
        """
        Process an anomaly and send notifications if it matches any rules.
        
        Args:
            anomaly: The anomaly to process
            
        Returns:
            True if any notifications were queued
        """
        if not self.enabled:
            logger.debug("Notifications disabled, ignoring anomaly")
            return False
        
        # Find matching rules
        matching_rules = []
        for rule in self.rules:
            if rule.matches(anomaly):
                matching_rules.append(rule)
        
        if not matching_rules:
            logger.debug("No matching rules for anomaly")
            return False
        
        # Queue notifications for each matching rule
        for rule in matching_rules:
            self.notification_queue.put((anomaly, rule))
            logger.debug(f"Queued notification for rule: {rule.name}")
        
        return True
    
    def _process_notifications(self) -> None:
        """Process notifications from the queue."""
        while self.running:
            try:
                # Get notification from queue with timeout
                try:
                    anomaly, rule = self.notification_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process notification
                logger.debug(f"Processing notification for rule: {rule.name}")
                
                all_success = True
                used_channels = []
                
                # Send through each channel
                for channel in rule.channels:
                    try:
                        if channel == NotificationChannel.EMAIL:
                            success = self._send_email(anomaly, rule)
                        elif channel == NotificationChannel.WEBHOOK:
                            success = self._send_webhook(anomaly, rule)
                        elif channel == NotificationChannel.LOG:
                            success = self._send_log(anomaly, rule)
                        elif channel == NotificationChannel.CALLBACK:
                            success = self._send_callback(anomaly, rule)
                        else:
                            logger.warning(f"Unknown notification channel: {channel}")
                            success = False
                        
                        if success:
                            used_channels.append(channel)
                        else:
                            all_success = False
                            
                    except Exception as e:
                        logger.error(f"Error sending notification through {channel.value}: {e}")
                        all_success = False
                
                # Record in history
                status = "sent" if all_success else "partial_failure" if used_channels else "failed"
                self.history.add(anomaly, rule.name, used_channels, status)
                
                # Mark as done
                self.notification_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    def _send_email(self, anomaly: Dict[str, Any], rule: NotificationRule) -> bool:
        """
        Send email notification.
        
        Args:
            anomaly: The anomaly to notify about
            rule: The rule that matched
            
        Returns:
            True if email was sent successfully
        """
        if not self.email_config:
            logger.warning("Email configuration not provided")
            return False
        
        if not self.email_config.to_addresses:
            logger.warning("No email recipients configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            
            # Format subject
            severity = anomaly.get("severity", "unknown").upper()
            api = anomaly.get("api", "unknown").upper()
            metric = anomaly.get("metric_type", "unknown")
            
            subject = f"{self.email_config.subject_prefix} {severity} - {api} {metric} anomaly detected"
            msg["Subject"] = subject
            msg["From"] = self.email_config.from_address
            msg["To"] = ", ".join(self.email_config.to_addresses)
            
            # Format plain text body
            text_body = f"""
API Anomaly Alert
-----------------

Severity: {severity}
API: {api}
Metric: {metric}
Time: {datetime.fromtimestamp(anomaly.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}
Detection Method: {anomaly.get('detection_method', 'Unknown')}

Description: {anomaly.get('description', 'Unknown anomaly')}

Value: {anomaly.get('value', 'N/A')}
Expected: {anomaly.get('expected_value', 'N/A')}

This alert was triggered by rule: {rule.name}
            """
            
            # Format HTML body
            html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-bottom: 1px solid #dee2e6; }}
        .severity {{ font-weight: bold; }}
        .severity-critical {{ color: #dc3545; }}
        .severity-high {{ color: #dc3545; }}
        .severity-medium {{ color: #ffc107; }}
        .severity-low {{ color: #17a2b8; }}
        .content {{ padding: 20px; }}
        .footer {{ margin-top: 30px; font-size: 0.8em; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>API Anomaly Alert</h2>
    </div>
    <div class="content">
        <p><strong>Severity:</strong> <span class="severity severity-{anomaly.get('severity', 'low')}">{severity}</span></p>
        <p><strong>API:</strong> {api}</p>
        <p><strong>Metric:</strong> {metric}</p>
        <p><strong>Time:</strong> {datetime.fromtimestamp(anomaly.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Detection Method:</strong> {anomaly.get('detection_method', 'Unknown')}</p>
        
        <h3>Description</h3>
        <p>{anomaly.get('description', 'Unknown anomaly')}</p>
        
        <h3>Details</h3>
        <p><strong>Value:</strong> {anomaly.get('value', 'N/A')}</p>
        <p><strong>Expected:</strong> {anomaly.get('expected_value', 'N/A')}</p>
        
        <div class="footer">
            <p>This alert was triggered by rule: {rule.name}</p>
        </div>
    </div>
</body>
</html>
            """
            
            # Attach parts
            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            if self.email_config.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self.email_config.smtp_server, 
                    self.email_config.smtp_port, 
                    context=context
                ) as server:
                    if self.email_config.username and self.email_config.password:
                        server.login(self.email_config.username, self.email_config.password)
                    
                    server.sendmail(
                        self.email_config.from_address,
                        self.email_config.to_addresses,
                        msg.as_string()
                    )
            else:
                with smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port) as server:
                    if self.email_config.username and self.email_config.password:
                        server.login(self.email_config.username, self.email_config.password)
                    
                    server.sendmail(
                        self.email_config.from_address,
                        self.email_config.to_addresses,
                        msg.as_string()
                    )
            
            logger.info(f"Sent email notification to {len(self.email_config.to_addresses)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _send_webhook(self, anomaly: Dict[str, Any], rule: NotificationRule) -> bool:
        """
        Send webhook notification.
        
        Args:
            anomaly: The anomaly to notify about
            rule: The rule that matched
            
        Returns:
            True if webhook was sent successfully
        """
        if not self.webhook_config:
            logger.warning("Webhook configuration not provided")
            return False
        
        try:
            # Prepare payload
            payload = {
                "timestamp": time.time(),
                "anomaly": anomaly,
                "rule": rule.name,
                "severity": anomaly.get("severity", "low"),
                "api": anomaly.get("api", "unknown"),
                "metric": anomaly.get("metric_type", "unknown"),
                "description": anomaly.get("description", "Unknown anomaly")
            }
            
            # Send webhook with retries
            retry_count = 0
            max_retries = self.webhook_config.retry_count
            
            while retry_count <= max_retries:
                try:
                    if self.webhook_config.method.upper() == "POST":
                        response = requests.post(
                            self.webhook_config.url,
                            json=payload,
                            headers=self.webhook_config.headers,
                            auth=self.webhook_config.auth,
                            timeout=self.webhook_config.timeout
                        )
                    elif self.webhook_config.method.upper() == "PUT":
                        response = requests.put(
                            self.webhook_config.url,
                            json=payload,
                            headers=self.webhook_config.headers,
                            auth=self.webhook_config.auth,
                            timeout=self.webhook_config.timeout
                        )
                    else:
                        logger.error(f"Unsupported webhook method: {self.webhook_config.method}")
                        return False
                    
                    # Check response
                    response.raise_for_status()
                    logger.info(f"Sent webhook notification, status code: {response.status_code}")
                    return True
                    
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    logger.warning(f"Webhook attempt {retry_count} failed: {e}")
                    
                    if retry_count > max_retries:
                        logger.error(f"Webhook failed after {max_retries} retries")
                        return False
                    
                    # Exponential backoff
                    time.sleep(2 ** retry_count)
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    def _send_log(self, anomaly: Dict[str, Any], rule: NotificationRule) -> bool:
        """
        Send log notification.
        
        Args:
            anomaly: The anomaly to notify about
            rule: The rule that matched
            
        Returns:
            True if log was sent successfully
        """
        try:
            severity = anomaly.get("severity", "unknown").upper()
            api = anomaly.get("api", "unknown").upper()
            metric = anomaly.get("metric_type", "unknown")
            description = anomaly.get("description", "Unknown anomaly")
            
            logger.warning(f"ANOMALY ALERT [{severity}] - {api} {metric}: {description}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending log notification: {e}")
            return False
    
    def _send_callback(self, anomaly: Dict[str, Any], rule: NotificationRule) -> bool:
        """
        Send callback notification.
        
        Args:
            anomaly: The anomaly to notify about
            rule: The rule that matched
            
        Returns:
            True if any callbacks were called successfully
        """
        if not self.callbacks:
            logger.warning("No callbacks registered")
            return False
        
        success = False
        
        for callback_id, callbacks in self.callbacks.items():
            for callback in callbacks:
                try:
                    callback(anomaly)
                    success = True
                except Exception as e:
                    logger.error(f"Error in callback {callback_id}: {e}")
        
        if success:
            logger.info(f"Called {sum(len(cbs) for cbs in self.callbacks.values())} callbacks")
        
        return success
    
    def save_data(self) -> bool:
        """
        Save notification data to disk.
        
        Returns:
            True if data was saved successfully
        """
        try:
            # Save rules
            rules_data = [rule.to_dict() for rule in self.rules]
            rules_file = os.path.join(self.data_dir, "notification_rules.json")
            with open(rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            # Save history
            history_file = os.path.join(self.data_dir, "notification_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.history.to_dict(), f, indent=2)
            
            logger.info(f"Saved notification data to {self.data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving notification data: {e}")
            return False
    
    def load_data(self) -> bool:
        """
        Load notification data from disk.
        
        Returns:
            True if data was loaded successfully
        """
        success = True
        
        # Load rules
        rules_file = os.path.join(self.data_dir, "notification_rules.json")
        if os.path.exists(rules_file):
            try:
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                
                self.rules = [NotificationRule.from_dict(rule_data) for rule_data in rules_data]
                logger.info(f"Loaded {len(self.rules)} notification rules")
            except Exception as e:
                logger.error(f"Error loading notification rules: {e}")
                success = False
        
        # Load history
        history_file = os.path.join(self.data_dir, "notification_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                self.history = NotificationHistory.from_dict(history_data)
                logger.info(f"Loaded notification history with {len(self.history.notifications)} entries")
            except Exception as e:
                logger.error(f"Error loading notification history: {e}")
                success = False
        
        return success
    
    def get_notification_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent notifications.
        
        Returns:
            Dictionary with notification summary
        """
        # Get recent notifications (last 24 hours)
        recent = self.history.get_recent(hours=24)
        
        if not recent:
            return {
                "total_notifications": 0,
                "status_counts": {
                    "sent": 0,
                    "partial_failure": 0,
                    "failed": 0
                },
                "rule_counts": {},
                "channel_counts": {},
                "api_counts": {},
                "latest_notification": None
            }
        
        # Count notifications by status
        status_counts = {
            "sent": 0,
            "partial_failure": 0,
            "failed": 0
        }
        
        for notification in recent:
            status = notification.get("status", "unknown")
            if status in status_counts:
                status_counts[status] += 1
        
        # Count notifications by rule
        rule_counts = {}
        for notification in recent:
            rule = notification.get("rule_name", "unknown")
            if rule not in rule_counts:
                rule_counts[rule] = 0
            rule_counts[rule] += 1
        
        # Count notifications by channel
        channel_counts = {}
        for notification in recent:
            channels = notification.get("channels", [])
            for channel in channels:
                if channel not in channel_counts:
                    channel_counts[channel] = 0
                channel_counts[channel] += 1
        
        # Count notifications by API
        api_counts = {}
        for notification in recent:
            api = notification.get("anomaly", {}).get("api", "unknown")
            if api not in api_counts:
                api_counts[api] = 0
            api_counts[api] += 1
        
        # Get latest notification
        latest = recent[0] if recent else None
        
        return {
            "total_notifications": len(recent),
            "status_counts": status_counts,
            "rule_counts": rule_counts,
            "channel_counts": channel_counts,
            "api_counts": api_counts,
            "latest_notification": latest
        }


def generate_default_email_config() -> EmailConfig:
    """Generate default email configuration for example use."""
    return EmailConfig(
        smtp_server="smtp.example.com",
        smtp_port=587,
        use_ssl=True,
        username="api-monitor@example.com",
        password="your-password",
        from_address="api-monitor@example.com",
        to_addresses=["alerts@example.com"],
        subject_prefix="[API Alert]"
    )


def generate_default_webhook_config() -> WebhookConfig:
    """Generate default webhook configuration for example use."""
    return WebhookConfig(
        url="https://hooks.example.com/services/webhook",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer your-token"
        }
    )


def main():
    """Example usage of the notification manager."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create notification manager
    manager = NotificationManager()
    
    # Add rules
    manager.add_rule(NotificationRule(
        name="Critical Alerts",
        min_severity=AnomalySeverity.CRITICAL,
        channels=[NotificationChannel.LOG]
    ))
    
    manager.add_rule(NotificationRule(
        name="Latency Monitoring",
        min_severity=AnomalySeverity.MEDIUM,
        anomaly_types=[AnomalyType.LATENCY_SPIKE],
        channels=[NotificationChannel.LOG]
    ))
    
    # Start notification manager
    manager.start()
    
    # Create example anomaly
    anomaly = {
        "timestamp": time.time(),
        "api": "openai",
        "metric_type": "latency",
        "value": 3.5,
        "expected_value": 1.2,
        "detection_method": "zscore",
        "severity": "high",
        "anomaly_type": "latency_spike",
        "description": "Latency spike detected for OpenAI API"
    }
    
    # Send notification
    manager.notify(anomaly)
    
    # Wait for notifications to process
    time.sleep(1)
    
    # Print summary
    summary = manager.get_notification_summary()
    print(f"Sent {summary['total_notifications']} notifications in the last 24 hours")
    
    # Save data
    manager.save_data()
    
    # Stop notification manager
    manager.stop()


if __name__ == "__main__":
    main()