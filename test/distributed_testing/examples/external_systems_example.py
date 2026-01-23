#!/usr/bin/env python3
"""
External Systems Connector Example

This script demonstrates how to use the various external system connectors
to interact with Jira, Slack, TestRail, and Prometheus.
"""

import anyio
import argparse
import logging
import json
import os
import sys

# Add the parent directory to the Python path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from distributed_testing.external_systems.register_connectors import create_connector, get_available_connectors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_jira_connector(config):
    """Test the Jira connector."""
    logger.info("Testing Jira connector...")
    
    # Create and initialize the connector
    jira = await create_connector("jira", config)
    
    # Connect to Jira
    connected = await jira.connect()
    if not connected:
        logger.error("Failed to connect to Jira. Check credentials and try again.")
        return
    
    logger.info("Connected to Jira successfully!")
    
    # Get system info
    system_info = await jira.system_info()
    logger.info(f"Jira system info: {json.dumps(system_info, indent=2)}")
    
    # Create an issue
    try:
        issue_data = {
            "summary": "Test issue from distributed testing framework",
            "description": "This is a test issue created by the distributed testing framework example script.",
            "issue_type": "Task",
            "project_key": config.get("project_key")
        }
        
        issue = await jira.create_item("issue", issue_data)
        logger.info(f"Created issue: {issue.get('key')} (ID: {issue.get('id')})")
        
        # Get the issue
        issue_details = await jira.get_item("issue", issue.get('key'))
        logger.info(f"Retrieved issue details: {issue_details.get('key')} - {issue_details.get('fields', {}).get('summary')}")
        
        # Add a comment
        comment_data = {
            "issue_key": issue.get('key'),
            "body": "This is a test comment added by the example script."
        }
        
        comment = await jira.execute_operation("add_comment", comment_data)
        logger.info(f"Added comment: {comment.get('result_data', {}).get('id')}")
        
        # Update the issue
        update_success = await jira.update_item("issue", issue.get('key'), {
            "description": "Updated description from the example script."
        })
        logger.info(f"Updated issue: {update_success}")
        
        # Delete the issue (cleanup)
        if config.get("cleanup", True):
            delete_success = await jira.delete_item("issue", issue.get('key'))
            logger.info(f"Deleted issue: {delete_success}")
        
    except Exception as e:
        logger.error(f"Error testing Jira connector: {str(e)}")
    
    # Close the connection
    await jira.close()
    logger.info("Jira connector test completed.")

async def test_slack_connector(config):
    """Test the Slack connector."""
    logger.info("Testing Slack connector...")
    
    # Create and initialize the connector
    slack = await create_connector("slack", config)
    
    # Connect to Slack
    connected = await slack.connect()
    if not connected:
        logger.error("Failed to connect to Slack. Check credentials and try again.")
        return
    
    logger.info("Connected to Slack successfully!")
    
    # Get system info
    system_info = await slack.system_info()
    logger.info(f"Slack system info: {json.dumps(system_info, indent=2)}")
    
    # Send a message
    try:
        message_data = {
            "channel": config.get("default_channel"),
            "text": "Hello from the distributed testing framework! 👋",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Distributed Testing Framework",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Example script output*\nThis message was sent by the external systems example script."
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "🕒 " + # TODO: Remove event loop management - asyncio.get_event_loop().time().__str__()
                        }
                    ]
                }
            ]
        }
        
        message = await slack.create_item("message", message_data)
        logger.info(f"Sent message: {message.get('ts')} to channel: {message.get('channel')}")
        
        # Add a reaction to the message
        reaction_data = {
            "channel": message.get('channel'),
            "ts": message.get('ts'),
            "name": "robot_face"
        }
        
        reaction_result = await slack.execute_operation("add_reaction", reaction_data)
        logger.info(f"Added reaction: {reaction_result.get('success')}")
        
        # Update the message
        update_data = {
            "channel": message.get('channel'),
            "text": "Updated: Hello from the distributed testing framework! 👋 (edited)",
            "blocks": message_data.get("blocks")  # Keep the same blocks
        }
        
        update_success = await slack.update_item("message", message.get('ts'), update_data)
        logger.info(f"Updated message: {update_success}")
        
        # Delete the message (cleanup)
        if config.get("cleanup", True):
            delete_success = await slack.delete_item("message", message.get('ts'))
            logger.info(f"Deleted message: {delete_success}")
        
    except Exception as e:
        logger.error(f"Error testing Slack connector: {str(e)}")
    
    # Close the connection
    await slack.close()
    logger.info("Slack connector test completed.")

async def test_testrail_connector(config):
    """Test the TestRail connector."""
    logger.info("Testing TestRail connector...")
    
    # Create and initialize the connector
    testrail = await create_connector("testrail", config)
    
    # Connect to TestRail
    connected = await testrail.connect()
    if not connected:
        logger.error("Failed to connect to TestRail. Check credentials and try again.")
        return
    
    logger.info("Connected to TestRail successfully!")
    
    # Get system info
    system_info = await testrail.system_info()
    logger.info(f"TestRail system info: {json.dumps(system_info, indent=2)}")
    
    # Create a test case
    try:
        # First, check if we have a valid section ID
        section_id = config.get("section_id")
        if not section_id:
            logger.error("section_id is required to create a test case.")
            return
        
        test_case_data = {
            "title": "Distributed Testing Framework Test Case",
            "section_id": section_id,
            "priority_id": 2,  # Medium
            "template_id": 1,  # Test Case (Steps)
            "custom_steps": "Step 1: Initialize the test\nStep 2: Execute the test\nStep 3: Verify the results",
            "custom_expected": "Test should pass without errors."
        }
        
        test_case = await testrail.create_item("test_case", test_case_data)
        logger.info(f"Created test case: {test_case.get('id')} - {test_case.get('title')}")
        
        # Create a test run
        test_run_data = {
            "name": "Distributed Testing Framework Test Run",
            "project_id": config.get("project_id"),
            "include_all": False,
            "case_ids": [test_case.get('id')]
        }
        
        if config.get("suite_id"):
            test_run_data["suite_id"] = config.get("suite_id")
        
        test_run = await testrail.create_item("test_run", test_run_data)
        logger.info(f"Created test run: {test_run.get('id')} - {test_run.get('name')}")
        
        # Get the tests in the run
        test_results = await testrail.execute_operation("get_test_results", {"run_id": test_run.get('id')})
        logger.info(f"Test results: {json.dumps(test_results.get('result_data', {}), indent=2)}")
        
        # Add a test result
        # First, get the test ID from the run
        run_info = await testrail.get_item("test_run", test_run.get('id'))
        
        # For demo purposes, we'll just use the first test
        test_id = None
        for test in run_info.get("tests", []):
            test_id = test.get("id")
            break
        
        if test_id:
            test_result_data = {
                "test_id": test_id,
                "status_id": 1,  # Passed
                "comment": "Test passed successfully from the example script.",
                "elapsed": "1m 45s"
            }
            
            test_result = await testrail.create_item("test_result", test_result_data)
            logger.info(f"Added test result: {test_result.get('id')}")
        
        # Close the test run
        if config.get("cleanup", True):
            close_result = await testrail.execute_operation("close_test_run", {"run_id": test_run.get('id')})
            logger.info(f"Closed test run: {close_result.get('success')}")
        
    except Exception as e:
        logger.error(f"Error testing TestRail connector: {str(e)}")
    
    # Close the connection
    await testrail.close()
    logger.info("TestRail connector test completed.")

async def test_prometheus_connector(config):
    """Test the Prometheus connector."""
    logger.info("Testing Prometheus connector...")
    
    # Create and initialize the connector
    prometheus = await create_connector("prometheus", config)
    
    # Connect to Prometheus
    connected = await prometheus.connect()
    if not connected:
        logger.error("Failed to connect to Prometheus. Check URL and try again.")
        return
    
    logger.info("Connected to Prometheus successfully!")
    
    # Get system info
    system_info = await prometheus.system_info()
    logger.info(f"Prometheus system info: {json.dumps(system_info, indent=2)}")
    
    # Query metrics
    try:
        # Execute a simple query to get memory usage
        query_data = {
            "query": "process_resident_memory_bytes"
        }
        
        metrics = await prometheus.execute_operation("query", query_data)
        logger.info(f"Memory usage metrics: {json.dumps(metrics.get('result_data', {}), indent=2)}")
        
        # Execute a range query
        current_time = # TODO: Remove event loop management - asyncio.get_event_loop().time()
        range_query_data = {
            "query": "process_resident_memory_bytes",
            "start": current_time - 3600,  # 1 hour ago
            "end": current_time,
            "step": "5m"  # 5-minute intervals
        }
        
        range_metrics = await prometheus.execute_operation("query_range", range_query_data)
        logger.info(f"Range query results: {len(range_metrics.get('result_data', {}).get('result', []))} data points")
        
        # Push a metric to Pushgateway if configured
        if config.get("pushgateway_url"):
            metric_data = {
                "job_name": "distributed_testing_example",
                "instance": "example_script",
                "metrics": [
                    {
                        "name": "example_script_run_count",
                        "value": 1,
                        "help": "Number of times the example script has been run",
                        "type": "counter",
                        "labels": {
                            "script": "external_systems_example.py",
                            "environment": "development"
                        }
                    },
                    {
                        "name": "example_script_execution_time",
                        "value": 5.67,  # Example value
                        "help": "Time taken to execute the example script",
                        "type": "gauge",
                        "labels": {
                            "script": "external_systems_example.py",
                            "environment": "development"
                        }
                    }
                ]
            }
            
            push_result = await prometheus.create_item("metrics", metric_data)
            logger.info(f"Pushed metrics: {json.dumps(push_result, indent=2)}")
            
            # Delete the metrics (cleanup)
            if config.get("cleanup", True):
                delete_result = await prometheus.delete_item("metrics", "distributed_testing_example")
                logger.info(f"Deleted metrics: {delete_result}")
        
    except Exception as e:
        logger.error(f"Error testing Prometheus connector: {str(e)}")
    
    # Close the connection
    await prometheus.close()
    logger.info("Prometheus connector test completed.")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="External Systems Connector Example")
    parser.add_argument("--connector", choices=["jira", "slack", "testrail", "prometheus", "all"],
                      default="all", help="Connector to test (default: all)")
    parser.add_argument("--config", type=str, default="example_config.json",
                      help="Path to config file (default: example_config.json)")
    parser.add_argument("--no-cleanup", action="store_true",
                      help="Don't clean up test data after tests")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {str(e)}")
        logger.info("Using empty config. Tests may fail if required credentials are not provided.")
        config = {}
    
    # Add cleanup flag to config
    for connector_config in config.values():
        connector_config["cleanup"] = not args.no_cleanup
    
    # Get available connectors
    available_connectors = get_available_connectors()
    logger.info(f"Available connectors: {available_connectors}")
    
    # Run tests
    if args.connector == "all" or args.connector == "jira":
        if "jira" in config:
            await test_jira_connector(config["jira"])
        else:
            logger.error("Jira configuration not found in config file.")
    
    if args.connector == "all" or args.connector == "slack":
        if "slack" in config:
            await test_slack_connector(config["slack"])
        else:
            logger.error("Slack configuration not found in config file.")
    
    if args.connector == "all" or args.connector == "testrail":
        if "testrail" in config:
            await test_testrail_connector(config["testrail"])
        else:
            logger.error("TestRail configuration not found in config file.")
    
    if args.connector == "all" or args.connector == "prometheus":
        if "prometheus" in config:
            await test_prometheus_connector(config["prometheus"])
        else:
            logger.error("Prometheus configuration not found in config file.")
    
    logger.info("Example script completed.")

if __name__ == "__main__":
    anyio.run(main())