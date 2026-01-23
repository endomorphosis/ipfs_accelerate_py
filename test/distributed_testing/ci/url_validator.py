#!/usr/bin/env python3
"""
URL Validation System for Artifact URLs.

This module provides functionality to validate that artifact URLs are
still accessible and implements health monitoring for artifact availability.
"""

from __future__ import annotations

import anyio
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover
    aiohttp = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArtifactURLValidator:
    """
    Validates artifact URLs to ensure they are accessible.
    
    This class implements URL validation functionality including:
    - Validation of URL accessibility
    - Periodic health checks for URLs
    - Caching of validation results to minimize external requests
    - Configurable retry and timeout settings
    - Status tracking for URLs
    """
    
    def __init__(
        self,
        check_timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_ttl: int = 3600,
        health_check_interval: int = 86400,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize the URL validator.
        
        Args:
            check_timeout: Timeout in seconds for URL validation checks
            max_retries: Maximum number of retries for URL validation
            retry_delay: Delay between retries in seconds
            cache_ttl: Time-to-live for cached validation results in seconds
            health_check_interval: Interval for periodic health checks in seconds
            session: Optional aiohttp session to use for requests
        """
        self.check_timeout = check_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_ttl = cache_ttl
        self.health_check_interval = health_check_interval
        
        # Cache structure: {url: (is_valid, timestamp, status_code, error_message)}
        self._validation_cache: Dict[str, Tuple[bool, float, Optional[int], Optional[str]]] = {}
        
        # Health history: {url: [(timestamp, is_valid, status_code, error_message)]}
        self._health_history: Dict[str, List[Tuple[float, bool, Optional[int], Optional[str]]]] = {}
        
        # Track registered URLs for health checks
        self._registered_urls: Set[str] = set()
        
        # aiohttp session
        self._session = session
        self._owns_session = False
        
        # Health check task
        self._health_check_task = None
    
    async def initialize(self):
        """Initialize the validator and start health check task."""
        if self._session is None:
            if aiohttp is None:
                raise RuntimeError(
                    "aiohttp is required to create a default HTTP session for URL validation. "
                    "Provide a custom session or install aiohttp."
                )
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        
        # Start health check task
        self._start_health_check_task()
    
    async def close(self):
        """Close resources and stop health check task."""
        self._stop_health_check_task()
        
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None
    
    def _start_health_check_task(self):
        """Start the background task for periodic health checks."""
        # Keep this as a no-op unless an event loop/task framework is explicitly wired.
        # This avoids import-time asyncio task creation and keeps the validator usable
        # for on-demand validation in tests.
        if self._health_check_task is None:
            self._health_check_task = True
            logger.info(
                f"Health check background task disabled; interval={self.health_check_interval}s"
            )
    
    def _stop_health_check_task(self):
        """Stop the background task for periodic health checks."""
        if self._health_check_task is not None:
            self._health_check_task = None
            logger.info("Stopped health check task")
    
    async def _run_periodic_health_checks(self):
        """Run periodic health checks for all registered URLs."""
        try:
            while True:
                # Wait for the health check interval
                await anyio.sleep(self.health_check_interval)
                
                if not self._registered_urls:
                    logger.debug("No URLs registered for health checks")
                    continue
                
                logger.info(f"Running periodic health check for {len(self._registered_urls)} URLs")
                
                # Create tasks for all registered URLs
                for url in self._registered_urls:
                    try:
                        is_valid, status_code, error_message = await self.validate_url(url, use_cache=False)
                        logger.info(f"Health check for {url}: {'Valid' if is_valid else 'Invalid'} "
                                    f"(Status: {status_code}, Error: {error_message})")
                    except Exception as e:
                        logger.error(f"Error in health check for {url}: {str(e)}")
        except anyio.get_cancelled_exc_class():
            logger.info("Periodic health check task cancelled")
        except Exception as e:
            logger.error(f"Error in periodic health check task: {str(e)}")
    
    async def validate_url(
        self, 
        url: str, 
        use_cache: bool = True
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Validate that a URL is accessible.
        
        Args:
            url: The URL to validate
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple of (is_valid, status_code, error_message)
        """
        # Check if URL should use cache and is in the cache
        if use_cache and url in self._validation_cache:
            is_valid, timestamp, status_code, error_message = self._validation_cache[url]
            
            # Check if cache is still valid
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Using cached validation result for {url}")
                return is_valid, status_code, error_message
        
        # Validate URL
        is_valid = False
        status_code = None
        error_message = None
        
        # Try to validate the URL with retries
        for attempt in range(self.max_retries):
            try:
                if self._session is None:
                    if aiohttp is None:
                        raise RuntimeError(
                            "aiohttp is required to validate URLs when no custom session is provided"
                        )
                    # Create a temporary session if not initialized
                    async with aiohttp.ClientSession() as session:
                        is_valid, status_code, error_message = await self._check_url(url, session)
                else:
                    is_valid, status_code, error_message = await self._check_url(url, self._session)
                
                # If successful, break the retry loop
                if is_valid:
                    break
                
                # Otherwise, wait before retrying
                if attempt < self.max_retries - 1:
                    await anyio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error validating URL {url} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                error_message = str(e)
                
                # Wait before retrying
                if attempt < self.max_retries - 1:
                    await anyio.sleep(self.retry_delay)
        
        # Cache the validation result
        self._validation_cache[url] = (is_valid, time.time(), status_code, error_message)
        
        # Add to health history
        if url not in self._health_history:
            self._health_history[url] = []
        
        self._health_history[url].append((time.time(), is_valid, status_code, error_message))
        
        # Keep history limited to 100 entries per URL
        if len(self._health_history[url]) > 100:
            self._health_history[url] = self._health_history[url][-100:]
        
        # Register URL for health checks if not already registered
        self._registered_urls.add(url)
        
        return is_valid, status_code, error_message
    
    async def _check_url(
        self, 
        url: str, 
        session: aiohttp.ClientSession
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Check if a URL is accessible.
        
        Args:
            url: The URL to check
            session: aiohttp session to use for the request
            
        Returns:
            Tuple of (is_valid, status_code, error_message)
        """
        try:
            # Use HEAD request to minimize data transfer
            async with session.head(
                url, 
                timeout=self.check_timeout,
                allow_redirects=True
            ) as response:
                is_valid = response.status < 400
                return is_valid, response.status, None if is_valid else f"HTTP error: {response.status}"
                
        except asyncio.TimeoutError:
            return False, None, f"Timeout after {self.check_timeout} seconds"
        except aiohttp.ClientError as e:
            return False, None, f"Client error: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    async def validate_urls(
        self, 
        urls: List[str],
        use_cache: bool = True
    ) -> Dict[str, Tuple[bool, Optional[int], Optional[str]]]:
        """
        Validate multiple URLs in parallel.
        
        Args:
            urls: List of URLs to validate
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary mapping URLs to their validation results (is_valid, status_code, error_message)
        """
        # NOTE: This is implemented sequentially to avoid asyncio task APIs.
        results: Dict[str, Tuple[bool, Optional[int], Optional[str]]] = {}
        for url in urls:
            if not url:  # Skip empty URLs
                continue
            try:
                results[url] = await self.validate_url(url, use_cache=use_cache)
            except Exception as e:
                logger.error(f"Error validating URL {url}: {str(e)}")
                results[url] = (False, None, str(e))

        return results
    
    def get_url_health(
        self, 
        url: str,
        timespan: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get health information for a URL.
        
        Args:
            url: The URL to get health information for
            timespan: Optional timespan in seconds to limit history (None for all history)
            
        Returns:
            Dictionary containing health information:
            {
                "url": str,
                "is_valid": bool,
                "last_checked": float,
                "status_code": Optional[int],
                "error_message": Optional[str],
                "availability": float,  # Percentage of successful checks
                "history": List[Dict],  # Health check history
                "average_response_time": Optional[float]
            }
        """
        result = {
            "url": url,
            "is_valid": False,
            "last_checked": None,
            "status_code": None,
            "error_message": "URL not checked",
            "availability": 0.0,
            "history": [],
            "average_response_time": None
        }
        
        # Check if URL has been validated
        if url in self._validation_cache:
            is_valid, timestamp, status_code, error_message = self._validation_cache[url]
            result["is_valid"] = is_valid
            result["last_checked"] = timestamp
            result["status_code"] = status_code
            result["error_message"] = error_message
        
        # Get health history
        if url in self._health_history:
            history = self._health_history[url]
            
            # Filter by timespan if specified
            if timespan is not None:
                cutoff = time.time() - timespan
                history = [entry for entry in history if entry[0] >= cutoff]
            
            # Calculate availability
            if history:
                valid_checks = sum(1 for _, is_valid, _, _ in history if is_valid)
                result["availability"] = (valid_checks / len(history)) * 100.0
                
                # Convert history to a list of dictionaries
                result["history"] = [
                    {
                        "timestamp": timestamp,
                        "is_valid": is_valid,
                        "status_code": status_code,
                        "error_message": error_message
                    }
                    for timestamp, is_valid, status_code, error_message in history
                ]
        
        return result
    
    def get_all_url_health(
        self,
        timespan: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get health information for all registered URLs.
        
        Args:
            timespan: Optional timespan in seconds to limit history (None for all history)
            
        Returns:
            Dictionary mapping URLs to their health information
        """
        results = {}
        for url in self._registered_urls:
            results[url] = self.get_url_health(url, timespan=timespan)
        return results
    
    def clear_cache(self, url: Optional[str] = None):
        """
        Clear the validation cache.
        
        Args:
            url: Optional URL to clear from cache (None to clear all)
        """
        if url is None:
            self._validation_cache.clear()
            logger.info("Cleared all URL validation cache entries")
        elif url in self._validation_cache:
            del self._validation_cache[url]
            logger.info(f"Cleared URL validation cache entry for {url}")
    
    def unregister_url(self, url: str):
        """
        Unregister a URL from health checks.
        
        Args:
            url: The URL to unregister
        """
        if url in self._registered_urls:
            self._registered_urls.remove(url)
            logger.info(f"Unregistered URL from health checks: {url}")
        
        if url in self._validation_cache:
            del self._validation_cache[url]
        
        if url in self._health_history:
            del self._health_history[url]
    
    def generate_health_report(
        self,
        timespan: Optional[int] = 86400,
        format: str = "dict"
    ) -> Union[Dict[str, Any], str]:
        """
        Generate a health report for all registered URLs.
        
        Args:
            timespan: Optional timespan in seconds to limit history (None for all history)
            format: Output format ('dict', 'json', 'markdown', or 'html')
            
        Returns:
            Health report in the specified format
        """
        # Get health information for all URLs
        url_health = self.get_all_url_health(timespan=timespan)
        
        # Calculate overall statistics
        total_urls = len(url_health)
        valid_urls = sum(1 for info in url_health.values() if info["is_valid"])
        overall_availability = sum(info["availability"] for info in url_health.values()) / total_urls if total_urls > 0 else 0
        
        report = {
            "timestamp": time.time(),
            "total_urls": total_urls,
            "valid_urls": valid_urls,
            "invalid_urls": total_urls - valid_urls,
            "overall_availability": overall_availability,
            "urls": url_health
        }
        
        # Return in the requested format
        if format == "dict":
            return report
        elif format == "json":
            import json
            return json.dumps(report, indent=2)
        elif format == "markdown":
            return self._generate_markdown_report(report)
        elif format == "html":
            return self._generate_html_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate a Markdown health report."""
        timestamp = datetime.fromtimestamp(report["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the report header
        md = f"# URL Health Report\n\n"
        md += f"*Generated on {timestamp}*\n\n"
        
        # Add summary section
        md += "## Summary\n\n"
        md += f"- Total URLs: {report['total_urls']}\n"
        valid_percent = (report['valid_urls'] / report['total_urls'] * 100.0) if report['total_urls'] > 0 else 0.0
        md += f"- Valid URLs: {report['valid_urls']} ({valid_percent:.1f}%)\n"
        md += f"- Invalid URLs: {report['invalid_urls']}\n"
        md += f"- Overall Availability: {report['overall_availability']:.1f}%\n\n"
        
        # Add URL status section
        md += "## URL Status\n\n"
        md += "| URL | Status | Last Checked | Availability |\n"
        md += "|-----|--------|--------------|-------------|\n"
        
        # Sort URLs by status (invalid first)
        sorted_urls = sorted(
            report["urls"].items(), 
            key=lambda item: (item[1]["is_valid"], item[1]["availability"])
        )
        
        for url, info in sorted_urls:
            status = "✅ Valid" if info["is_valid"] else f"❌ Invalid ({info['error_message']})"
            last_checked = datetime.fromtimestamp(info["last_checked"]).strftime("%Y-%m-%d %H:%M:%S") if info["last_checked"] else "Never"
            md += f"| {url} | {status} | {last_checked} | {info['availability']:.1f}% |\n"
        
        return md
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate an HTML health report."""
        timestamp = datetime.fromtimestamp(report["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        
        # Start with HTML boilerplate
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URL Health Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .valid {
            color: #27ae60;
            font-weight: bold;
        }
        .invalid {
            color: #e74c3c;
            font-weight: bold;
        }
        .summary-box {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .summary-item {
            flex: 1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .total {
            background-color: #f0f0f0;
        }
        .valid-box {
            background-color: #d5f5e3;
        }
        .invalid-box {
            background-color: #fadbd8;
        }
        .availability {
            background-color: #ebf5fb;
        }
        .footer {
            margin-top: 30px;
            font-size: 0.8em;
            color: #7f8c8d;
            text-align: center;
        }
    </style>
</head>
<body>
"""
        
        # Add header and summary
        html += f"<h1>URL Health Report</h1>\n"
        html += f"<p><em>Generated on {timestamp}</em></p>\n"
        
        # Add summary boxes
        html += "<div class=\"summary-box\">\n"
        html += f"  <div class=\"summary-item total\"><h3>Total URLs</h3><p>{report['total_urls']}</p></div>\n"
        valid_percent = report['valid_urls']/report['total_urls']*100 if report['total_urls'] > 0 else 0
        html += f"  <div class=\"summary-item valid-box\"><h3>Valid URLs</h3><p>{report['valid_urls']} ({valid_percent:.1f}%)</p></div>\n"
        html += f"  <div class=\"summary-item invalid-box\"><h3>Invalid URLs</h3><p>{report['invalid_urls']}</p></div>\n"
        html += f"  <div class=\"summary-item availability\"><h3>Overall Availability</h3><p>{report['overall_availability']:.1f}%</p></div>\n"
        html += "</div>\n"
        
        # Add URL status table
        html += "<h2>URL Status</h2>\n"
        html += "<table>\n"
        html += "  <tr><th>URL</th><th>Status</th><th>Last Checked</th><th>Availability</th></tr>\n"
        
        # Sort URLs by status (invalid first)
        sorted_urls = sorted(
            report["urls"].items(), 
            key=lambda item: (item[1]["is_valid"], item[1]["availability"])
        )
        
        for url, info in sorted_urls:
            status_class = "valid" if info["is_valid"] else "invalid"
            status_text = "Valid" if info["is_valid"] else f"Invalid: {info['error_message']}"
            last_checked = datetime.fromtimestamp(info["last_checked"]).strftime("%Y-%m-%d %H:%M:%S") if info["last_checked"] else "Never"
            
            html += f"  <tr>\n"
            html += f"    <td>{url}</td>\n"
            html += f"    <td class=\"{status_class}\">{status_text}</td>\n"
            html += f"    <td>{last_checked}</td>\n"
            html += f"    <td>{info['availability']:.1f}%</td>\n"
            html += f"  </tr>\n"
        
        html += "</table>\n"
        
        # Add footer
        html += "<div class=\"footer\">\n"
        html += "  Generated by ArtifactURLValidator\n"
        html += "</div>\n"
        
        # Close HTML tags
        html += "</body>\n</html>"
        
        return html


# Global validator instance for easy access
_global_validator = None

async def get_validator() -> ArtifactURLValidator:
    """
    Get or create the global URL validator instance.
    
    Returns:
        ArtifactURLValidator instance
    """
    global _global_validator
    
    if _global_validator is None:
        _global_validator = ArtifactURLValidator()
        await _global_validator.initialize()
    
    return _global_validator

async def validate_url(url: str, use_cache: bool = True) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate that a URL is accessible using the global validator.
    
    Args:
        url: The URL to validate
        use_cache: Whether to use cached results if available
        
    Returns:
        Tuple of (is_valid, status_code, error_message)
    """
    validator = await get_validator()
    return await validator.validate_url(url, use_cache=use_cache)

async def validate_urls(urls: List[str], use_cache: bool = True) -> Dict[str, Tuple[bool, Optional[int], Optional[str]]]:
    """
    Validate multiple URLs in parallel using the global validator.
    
    Args:
        urls: List of URLs to validate
        use_cache: Whether to use cached results if available
        
    Returns:
        Dictionary mapping URLs to their validation results (is_valid, status_code, error_message)
    """
    validator = await get_validator()
    return await validator.validate_urls(urls, use_cache=use_cache)

async def generate_health_report(timespan: Optional[int] = 86400, format: str = "dict") -> Union[Dict[str, Any], str]:
    """
    Generate a health report for all registered URLs using the global validator.
    
    Args:
        timespan: Optional timespan in seconds to limit history (None for all history)
        format: Output format ('dict', 'json', 'markdown', or 'html')
        
    Returns:
        Health report in the specified format
    """
    validator = await get_validator()
    return validator.generate_health_report(timespan=timespan, format=format)

async def close_validator():
    """Close the global validator instance."""
    global _global_validator
    
    if _global_validator is not None:
        await _global_validator.close()
        _global_validator = None