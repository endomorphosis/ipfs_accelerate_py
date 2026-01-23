#!/usr/bin/env python3
"""
Simple Test Script for Artifact URL Retrieval

This script tests the get_artifact_url implementation in different CI provider clients.
It uses mock implementations to simulate the behavior of the clients.
"""

import anyio
import json
import logging
import os
import tempfile
import time
from typing import Dict, Any, Optional

import sys
# Add the parent directory to the path
sys.path.append('/home/barberb/ipfs_accelerate_py/test')

from distributed_testing.ci.jenkins_client import JenkinsClient
from distributed_testing.ci.circleci_client import CircleCIClient
from distributed_testing.ci.azure_client import AzureDevOpsClient
from distributed_testing.ci.github_client import GitHubClient
from distributed_testing.ci.bitbucket_client import BitbucketClient
from distributed_testing.ci.teamcity_client import TeamCityClient
from distributed_testing.ci.travis_client import TravisClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock CI provider clients
class MockJenkinsClient(JenkinsClient):
    """Mock Jenkins client for testing."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with mock configuration."""
        self.url = "https://jenkins.example.com/"
        self.user = "mock_user"
        self.token = "mock_token"
        self.job_name = "mock_job"
        self.build_id = "123"
        self.session = None
        
        # Initialize the artifact URL cache
        self._artifact_urls = {
            "jenkins-mock_job-123": {
                "test.log": f"{self.url}job/mock_job/123/artifact/test.log",
                "test_report.json": f"{self.url}job/mock_job/123/artifact/test_report.json"
            }
        }
        
        logger.info(f"Initialized mock Jenkins client")
        return True
    
    async def _ensure_session(self):
        """No-op for mock implementation."""
        pass
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock test run."""
        job_name = test_run_data.get("job_name", "mock_job")
        build_id = test_run_data.get("build_id", "123")
        
        test_run_id = f"jenkins-{job_name}-{build_id}"
        
        # Add to the artifact URL cache
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        logger.info(f"Created mock Jenkins test run: {test_run_id}")
        
        return {
            "id": test_run_id,
            "name": test_run_data.get("name", "Mock Test Run"),
            "job_name": job_name,
            "build_id": build_id,
            "status": "running"
        }
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload mock artifact."""
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        # Extract job_name and build_id from test_run_id
        parts = test_run_id.split("-", 2)
        if len(parts) < 3:
            job_name = "mock_job"
            build_id = "123"
        else:
            job_name = parts[1]
            build_id = parts[2]
        
        # Create mock URL
        url = f"{self.url}job/{job_name}/{build_id}/artifact/{artifact_name}"
        self._artifact_urls[test_run_id][artifact_name] = url
        
        logger.info(f"Uploaded mock artifact {artifact_name} to Jenkins")
        return True
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """Override to use mock implementation."""
        if test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
            return self._artifact_urls[test_run_id][artifact_name]
        return None
    
    async def close(self):
        """No-op for mock implementation."""
        pass

class MockCircleCIClient(CircleCIClient):
    """Mock CircleCI client for testing."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with mock configuration."""
        self.token = "mock_token"
        self.project_slug = "github/mock-org/mock-repo"
        self.api_url = "https://circleci.com/api/v2"
        self.build_num = "123"
        self.session = None
        self.pipeline_id = "mock-pipeline-id"
        self.workflow_id = "mock-workflow-id"
        
        # Initialize the artifact URL cache
        self._artifact_urls = {
            "circleci-github/mock-org/mock-repo-123": {
                "test.log": f"{self.api_url}/project/{self.project_slug}/123/artifacts/test.log",
                "test_report.json": f"{self.api_url}/project/{self.project_slug}/123/artifacts/test_report.json"
            }
        }
        
        logger.info(f"Initialized mock CircleCI client")
        return True
    
    async def _ensure_session(self):
        """No-op for mock implementation."""
        pass
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock test run."""
        project_slug = test_run_data.get("project_slug", self.project_slug)
        job_number = test_run_data.get("job_number", "123")
        
        test_run_id = f"circleci-{project_slug}-{job_number}"
        
        # Add to the artifact URL cache
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        logger.info(f"Created mock CircleCI test run: {test_run_id}")
        
        return {
            "id": test_run_id,
            "name": test_run_data.get("name", "Mock Test Run"),
            "project_slug": project_slug,
            "job_number": job_number,
            "status": "running"
        }
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload mock artifact."""
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        # Extract project_slug and job_number from test_run_id
        parts = test_run_id.split("-", 2)
        if len(parts) < 3:
            project_slug = self.project_slug
            job_number = "123"
        else:
            project_slug = parts[1]
            job_number = parts[2]
        
        # Create mock URL
        url = f"{self.api_url}/project/{project_slug}/{job_number}/artifacts/{artifact_name}"
        self._artifact_urls[test_run_id][artifact_name] = url
        
        logger.info(f"Uploaded mock artifact {artifact_name} to CircleCI")
        return True
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """Override to use mock implementation."""
        if test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
            return self._artifact_urls[test_run_id][artifact_name]
        return None
    
    async def close(self):
        """No-op for mock implementation."""
        pass

class MockAzureDevOpsClient(AzureDevOpsClient):
    """Mock Azure DevOps client for testing."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with mock configuration."""
        self.token = "mock_token"
        self.organization = "mock_org"
        self.project = "mock_project"
        self.api_url = f"https://dev.azure.com/mock_org/mock_project/_apis/"
        self.session = None
        self.repository_id = None
        self.build_id = None
        
        # Initialize the artifact attachments cache
        self._artifact_attachments = {
            "123": {
                "test.log": {
                    "attachment_id": "456",
                    "url": f"{self.api_url}test/runs/123/attachments/456"
                },
                "test_report.json": {
                    "attachment_id": "789",
                    "url": f"{self.api_url}test/runs/123/attachments/789"
                }
            }
        }
        
        logger.info(f"Initialized mock Azure DevOps client")
        return True
    
    async def _ensure_session(self):
        """No-op for mock implementation."""
        pass
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock test run."""
        test_run_id = test_run_data.get("test_run_id", "123")
        
        # Add to the artifact attachments cache
        if test_run_id not in self._artifact_attachments:
            self._artifact_attachments[test_run_id] = {}
        
        logger.info(f"Created mock Azure DevOps test run: {test_run_id}")
        
        return {
            "id": test_run_id,
            "name": test_run_data.get("name", "Mock Test Run"),
            "status": "running"
        }
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload mock artifact."""
        if test_run_id not in self._artifact_attachments:
            self._artifact_attachments[test_run_id] = {}
        
        # Create a mock attachment ID
        attachment_id = f"{hash(artifact_name) % 1000 + 1000}"
        
        # Create mock URL
        url = f"{self.api_url}test/runs/{test_run_id}/attachments/{attachment_id}"
        
        self._artifact_attachments[test_run_id][artifact_name] = {
            "attachment_id": attachment_id,
            "url": url
        }
        
        logger.info(f"Uploaded mock artifact {artifact_name} to Azure DevOps")
        return True
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """Override to use mock implementation."""
        if test_run_id in self._artifact_attachments and artifact_name in self._artifact_attachments[test_run_id]:
            return self._artifact_attachments[test_run_id][artifact_name]["url"]
        return None
    
    async def close(self):
        """No-op for mock implementation."""
        pass

class MockGitHubClient(GitHubClient):
    """Mock GitHub client for testing."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with mock configuration."""
        self.token = "mock_token"
        self.repository = "mock-org/mock-repo"
        self.api_url = "https://api.github.com"
        self.commit_sha = "mock-sha"
        self.session = None
        
        # Initialize the artifact URL cache
        self._artifact_urls = {
            "github-mock-org/mock-repo-123": {
                "test.log": "https://github.com/mock-org/mock-repo/files/123/test.log",
                "test_report.json": "https://github.com/mock-org/mock-repo/files/123/test_report.json"
            }
        }
        
        logger.info(f"Initialized mock GitHub client")
        return True
    
    async def _ensure_session(self):
        """No-op for mock implementation."""
        pass
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock test run."""
        test_run_id = f"github-{self.repository}-123"
        
        # Add to the artifact URL cache
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        logger.info(f"Created mock GitHub test run: {test_run_id}")
        
        return {
            "id": test_run_id,
            "name": test_run_data.get("name", "Mock Test Run"),
            "status": "running"
        }
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload mock artifact."""
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        # Create mock URL
        url = f"https://github.com/mock-org/mock-repo/files/123/{artifact_name}"
        
        self._artifact_urls[test_run_id][artifact_name] = url
        
        logger.info(f"Uploaded mock artifact {artifact_name} to GitHub")
        return True
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """Override to use mock implementation."""
        if test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
            return self._artifact_urls[test_run_id][artifact_name]
        return None
    
    async def close(self):
        """No-op for mock implementation."""
        pass

class MockBitbucketClient(BitbucketClient):
    """Mock Bitbucket client for testing."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with mock configuration."""
        self.username = "mock_user"
        self.app_password = "mock_password"
        self.workspace = "mock_workspace"
        self.repository = "mock_repo"
        self.api_url = "https://api.bitbucket.org/2.0"
        self.session = None
        self.build_number = "123"
        self.commit_hash = "mock_hash"
        
        # Initialize the artifact URL cache
        self._artifact_urls = {
            "bitbucket-mock_workspace-mock_repo-report123": {
                "test.log": "https://bitbucket.org/mock_workspace/mock_repo/downloads/report123-test.log",
                "test_report.json": "https://bitbucket.org/mock_workspace/mock_repo/downloads/report123-test_report.json"
            }
        }
        
        logger.info(f"Initialized mock Bitbucket client")
        return True
    
    async def _ensure_session(self):
        """No-op for mock implementation."""
        pass
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock test run."""
        workspace = test_run_data.get("workspace", self.workspace)
        repository = test_run_data.get("repository", self.repository)
        report_id = test_run_data.get("report_id", "report456")
        
        test_run_id = f"bitbucket-{workspace}-{repository}-{report_id}"
        
        # Add to the artifact URL cache
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        logger.info(f"Created mock Bitbucket test run: {test_run_id}")
        
        return {
            "id": test_run_id,
            "name": test_run_data.get("name", "Mock Test Run"),
            "status": "running",
            "workspace": workspace,
            "repository": repository,
            "report_id": report_id
        }
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload mock artifact."""
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        # Extract workspace, repository, and report ID from test run ID
        parts = test_run_id.split("-", 3)
        if len(parts) < 4:
            workspace = self.workspace
            repository = self.repository
            report_id = "report123"
        else:
            workspace = parts[1]
            repository = parts[2]
            report_id = parts[3]
        
        # Create mock URL using the Bitbucket downloads pattern
        url = f"https://bitbucket.org/{workspace}/{repository}/downloads/{report_id}-{artifact_name}"
        
        self._artifact_urls[test_run_id][artifact_name] = url
        
        logger.info(f"Uploaded mock artifact {artifact_name} to Bitbucket")
        return True
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """Override to use mock implementation."""
        if test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
            return self._artifact_urls[test_run_id][artifact_name]
        return None
    
    async def close(self):
        """No-op for mock implementation."""
        pass

class MockTeamCityClient(TeamCityClient):
    """Mock TeamCity client for testing."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with mock configuration."""
        self.url = "https://teamcity.example.com/"
        self.username = "mock_user"
        self.password = "mock_token"
        self.build_id = "123"
        self.build_type_id = "ExampleProject_Build"
        self.session = None
        
        # Initialize the artifact URL cache
        self._artifact_urls = {
            "teamcity-123-456": {
                "test.log": "https://teamcity.example.com/app/rest/builds/id:123/artifacts/content/test.log",
                "test_report.json": "https://teamcity.example.com/app/rest/builds/id:123/artifacts/content/test_report.json"
            }
        }
        
        logger.info(f"Initialized mock TeamCity client")
        return True
    
    async def _ensure_session(self):
        """No-op for mock implementation."""
        pass
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock test run."""
        build_id = test_run_data.get("build_id", "456")
        
        test_run_id = f"teamcity-{build_id}-{int(time.time())}"
        
        # Add to the artifact URL cache
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        logger.info(f"Created mock TeamCity test run: {test_run_id}")
        
        return {
            "id": test_run_id,
            "name": test_run_data.get("name", "Mock Test Run"),
            "status": "running",
            "build_id": build_id
        }
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload mock artifact."""
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        # Extract build ID from test run ID
        parts = test_run_id.split("-", 2)
        if len(parts) < 3:
            build_id = "456"
        else:
            build_id = parts[1]
        
        # Create mock URL using TeamCity API format
        url = f"{self.url}app/rest/builds/id:{build_id}/artifacts/content/{artifact_name}"
        
        self._artifact_urls[test_run_id][artifact_name] = url
        
        logger.info(f"Uploaded mock artifact {artifact_name} to TeamCity")
        return True
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """Override to use mock implementation."""
        if test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
            return self._artifact_urls[test_run_id][artifact_name]
        return None
    
    async def close(self):
        """No-op for mock implementation."""
        pass

class MockTravisClient(TravisClient):
    """Mock Travis CI client for testing."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with mock configuration."""
        self.token = "mock_token"
        self.repository = "mock-org/mock-repo"
        self.api_url = "https://api.travis-ci.com"
        self.build_id = "123"
        self.session = None
        
        # Initialize the artifact URL cache
        self._artifact_urls = {
            "travis-mock-org/mock-repo-123": {
                "test.log": "https://s3.amazonaws.com/travis-artifacts/mock-org-mock-repo/123/test.log",
                "test_report.json": "https://s3.amazonaws.com/travis-artifacts/mock-org-mock-repo/123/test_report.json"
            }
        }
        
        logger.info(f"Initialized mock Travis CI client")
        return True
    
    async def _ensure_session(self):
        """No-op for mock implementation."""
        pass
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a mock test run."""
        repository = test_run_data.get("repository", self.repository)
        build_id = test_run_data.get("build_id", "456")
        
        test_run_id = f"travis-{repository}-{build_id}"
        
        # Add to the artifact URL cache
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        logger.info(f"Created mock Travis CI test run: {test_run_id}")
        
        return {
            "id": test_run_id,
            "name": test_run_data.get("name", "Mock Test Run"),
            "status": "running",
            "repository": repository,
            "build_id": build_id
        }
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload mock artifact."""
        if test_run_id not in self._artifact_urls:
            self._artifact_urls[test_run_id] = {}
        
        # Extract repository and build ID from test run ID
        parts = test_run_id.split("-", 2)
        if len(parts) < 3:
            repository = self.repository.replace("/", "-")
            build_id = "456"
        else:
            repository = parts[1].replace("/", "-")
            build_id = parts[2]
        
        # Create mock URL using S3 pattern commonly used with Travis
        url = f"https://s3.amazonaws.com/travis-artifacts/{repository}/{build_id}/{artifact_name}"
        
        self._artifact_urls[test_run_id][artifact_name] = url
        
        logger.info(f"Uploaded mock artifact {artifact_name} to Travis CI")
        return True
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """Override to use mock implementation."""
        if test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
            return self._artifact_urls[test_run_id][artifact_name]
        return None
    
    async def close(self):
        """No-op for mock implementation."""
        pass

async def test_jenkins_get_artifact_url():
    """Test the get_artifact_url method of the Jenkins client."""
    logger.info("Testing Jenkins get_artifact_url method...")
    
    # Create mock client
    jenkins = MockJenkinsClient()
    await jenkins.initialize({})
    
    # Create a test run
    test_run = await jenkins.create_test_run({
        "name": "Jenkins Test Run",
        "job_name": "test_job",
        "build_id": "456"
    })
    
    test_run_id = test_run["id"]
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps({"test": "data"}).encode('utf-8'))
        file_path = f.name
    
    try:
        # Upload the artifact
        result = await jenkins.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=file_path,
            artifact_name="test_result.json"
        )
        
        assert result, "Artifact upload failed"
        
        # Get the artifact URL
        url = await jenkins.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="test_result.json"
        )
        
        assert url is not None, "Failed to get artifact URL"
        assert "test_result.json" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"Jenkins artifact URL: {url}")
        
        # Test with a pre-cached artifact
        url = await jenkins.get_artifact_url(
            test_run_id="jenkins-mock_job-123",
            artifact_name="test.log"
        )
        
        assert url is not None, "Failed to get cached artifact URL"
        assert "test.log" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"Jenkins cached artifact URL: {url}")
        
        # Test with a non-existent artifact
        url = await jenkins.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="non_existent.json"
        )
        
        assert url is None, "Expected None for non-existent artifact"
        
        logger.info("Jenkins get_artifact_url test passed")
    
    finally:
        # Clean up
        os.unlink(file_path)
        await jenkins.close()

async def test_circleci_get_artifact_url():
    """Test the get_artifact_url method of the CircleCI client."""
    logger.info("Testing CircleCI get_artifact_url method...")
    
    # Create mock client
    circleci = MockCircleCIClient()
    await circleci.initialize({})
    
    # Create a test run
    test_run = await circleci.create_test_run({
        "name": "CircleCI Test Run",
        "project_slug": "github/test-org/test-repo",
        "job_number": "456"
    })
    
    test_run_id = test_run["id"]
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps({"test": "data"}).encode('utf-8'))
        file_path = f.name
    
    try:
        # Upload the artifact
        result = await circleci.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=file_path,
            artifact_name="test_result.json"
        )
        
        assert result, "Artifact upload failed"
        
        # Get the artifact URL
        url = await circleci.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="test_result.json"
        )
        
        assert url is not None, "Failed to get artifact URL"
        assert "test_result.json" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"CircleCI artifact URL: {url}")
        
        # Test with a pre-cached artifact
        url = await circleci.get_artifact_url(
            test_run_id="circleci-github/mock-org/mock-repo-123",
            artifact_name="test.log"
        )
        
        assert url is not None, "Failed to get cached artifact URL"
        assert "test.log" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"CircleCI cached artifact URL: {url}")
        
        # Test with a non-existent artifact
        url = await circleci.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="non_existent.json"
        )
        
        assert url is None, "Expected None for non-existent artifact"
        
        logger.info("CircleCI get_artifact_url test passed")
    
    finally:
        # Clean up
        os.unlink(file_path)
        await circleci.close()

async def test_azure_get_artifact_url():
    """Test the get_artifact_url method of the Azure DevOps client."""
    logger.info("Testing Azure DevOps get_artifact_url method...")
    
    # Create mock client
    azure = MockAzureDevOpsClient()
    await azure.initialize({})
    
    # Create a test run
    test_run = await azure.create_test_run({
        "name": "Azure DevOps Test Run",
        "test_run_id": "456"
    })
    
    test_run_id = test_run["id"]
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps({"test": "data"}).encode('utf-8'))
        file_path = f.name
    
    try:
        # Upload the artifact
        result = await azure.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=file_path,
            artifact_name="test_result.json"
        )
        
        assert result, "Artifact upload failed"
        
        # Get the artifact URL
        url = await azure.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="test_result.json"
        )
        
        assert url is not None, "Failed to get artifact URL"
        assert "test_result.json" in url or "attachments" in url, f"URL doesn't have expected pattern: {url}"
        
        logger.info(f"Azure DevOps artifact URL: {url}")
        
        # Test with a pre-cached artifact
        url = await azure.get_artifact_url(
            test_run_id="123",
            artifact_name="test.log"
        )
        
        assert url is not None, "Failed to get cached artifact URL"
        assert "attachments" in url, f"URL doesn't contain 'attachments': {url}"
        
        logger.info(f"Azure DevOps cached artifact URL: {url}")
        
        # Test with a non-existent artifact
        url = await azure.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="non_existent.json"
        )
        
        assert url is None, "Expected None for non-existent artifact"
        
        logger.info("Azure DevOps get_artifact_url test passed")
    
    finally:
        # Clean up
        os.unlink(file_path)
        await azure.close()

async def test_github_get_artifact_url():
    """Test the get_artifact_url method of the GitHub client."""
    logger.info("Testing GitHub get_artifact_url method...")
    
    # Create mock client
    github = MockGitHubClient()
    await github.initialize({})
    
    # Create a test run
    test_run = await github.create_test_run({
        "name": "GitHub Test Run"
    })
    
    test_run_id = test_run["id"]
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps({"test": "data"}).encode('utf-8'))
        file_path = f.name
    
    try:
        # Upload the artifact
        result = await github.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=file_path,
            artifact_name="test_result.json"
        )
        
        assert result, "Artifact upload failed"
        
        # Get the artifact URL
        url = await github.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="test_result.json"
        )
        
        assert url is not None, "Failed to get artifact URL"
        assert "test_result.json" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"GitHub artifact URL: {url}")
        
        # Test with a pre-cached artifact
        url = await github.get_artifact_url(
            test_run_id="github-mock-org/mock-repo-123",
            artifact_name="test.log"
        )
        
        assert url is not None, "Failed to get cached artifact URL"
        assert "test.log" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"GitHub cached artifact URL: {url}")
        
        # Test with a non-existent artifact
        url = await github.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="non_existent.json"
        )
        
        assert url is None, "Expected None for non-existent artifact"
        
        logger.info("GitHub get_artifact_url test passed")
    
    finally:
        # Clean up
        os.unlink(file_path)
        await github.close()

async def test_bitbucket_get_artifact_url():
    """Test the get_artifact_url method of the Bitbucket client."""
    logger.info("Testing Bitbucket get_artifact_url method...")
    
    # Create mock client
    bitbucket = MockBitbucketClient()
    await bitbucket.initialize({})
    
    # Create a test run
    test_run = await bitbucket.create_test_run({
        "name": "Bitbucket Test Run",
        "workspace": "test_workspace",
        "repository": "test_repo",
        "report_id": "test456"
    })
    
    test_run_id = test_run["id"]
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps({"test": "data"}).encode('utf-8'))
        file_path = f.name
    
    try:
        # Upload the artifact
        result = await bitbucket.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=file_path,
            artifact_name="test_result.json"
        )
        
        assert result, "Artifact upload failed"
        
        # Get the artifact URL
        url = await bitbucket.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="test_result.json"
        )
        
        assert url is not None, "Failed to get artifact URL"
        assert "test_result.json" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"Bitbucket artifact URL: {url}")
        
        # Test with a pre-cached artifact
        url = await bitbucket.get_artifact_url(
            test_run_id="bitbucket-mock_workspace-mock_repo-report123",
            artifact_name="test.log"
        )
        
        assert url is not None, "Failed to get cached artifact URL"
        assert "test.log" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"Bitbucket cached artifact URL: {url}")
        
        # Test with a non-existent artifact
        url = await bitbucket.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="non_existent.json"
        )
        
        assert url is None, "Expected None for non-existent artifact"
        
        logger.info("Bitbucket get_artifact_url test passed")
    
    finally:
        # Clean up
        os.unlink(file_path)
        await bitbucket.close()

async def test_teamcity_get_artifact_url():
    """Test the get_artifact_url method of the TeamCity client."""
    logger.info("Testing TeamCity get_artifact_url method...")
    
    # Create mock client
    teamcity = MockTeamCityClient()
    await teamcity.initialize({})
    
    # Create a test run
    test_run = await teamcity.create_test_run({
        "name": "TeamCity Test Run",
        "build_id": "789"
    })
    
    test_run_id = test_run["id"]
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps({"test": "data"}).encode('utf-8'))
        file_path = f.name
    
    try:
        # Upload the artifact
        result = await teamcity.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=file_path,
            artifact_name="test_result.json"
        )
        
        assert result, "Artifact upload failed"
        
        # Get the artifact URL
        url = await teamcity.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="test_result.json"
        )
        
        assert url is not None, "Failed to get artifact URL"
        assert "test_result.json" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"TeamCity artifact URL: {url}")
        
        # Test with a pre-cached artifact
        url = await teamcity.get_artifact_url(
            test_run_id="teamcity-123-456",
            artifact_name="test.log"
        )
        
        assert url is not None, "Failed to get cached artifact URL"
        assert "test.log" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"TeamCity cached artifact URL: {url}")
        
        # Test with a non-existent artifact
        url = await teamcity.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="non_existent.json"
        )
        
        assert url is None, "Expected None for non-existent artifact"
        
        logger.info("TeamCity get_artifact_url test passed")
    
    finally:
        # Clean up
        os.unlink(file_path)
        await teamcity.close()

async def test_travis_get_artifact_url():
    """Test the get_artifact_url method of the Travis CI client."""
    logger.info("Testing Travis CI get_artifact_url method...")
    
    # Create mock client
    travis = MockTravisClient()
    await travis.initialize({})
    
    # Create a test run
    test_run = await travis.create_test_run({
        "name": "Travis CI Test Run",
        "repository": "test-org/test-repo",
        "build_id": "789"
    })
    
    test_run_id = test_run["id"]
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps({"test": "data"}).encode('utf-8'))
        file_path = f.name
    
    try:
        # Upload the artifact
        result = await travis.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=file_path,
            artifact_name="test_result.json"
        )
        
        assert result, "Artifact upload failed"
        
        # Get the artifact URL
        url = await travis.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="test_result.json"
        )
        
        assert url is not None, "Failed to get artifact URL"
        assert "test_result.json" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"Travis CI artifact URL: {url}")
        
        # Test with a pre-cached artifact
        url = await travis.get_artifact_url(
            test_run_id="travis-mock-org/mock-repo-123",
            artifact_name="test.log"
        )
        
        assert url is not None, "Failed to get cached artifact URL"
        assert "test.log" in url, f"URL doesn't contain artifact name: {url}"
        
        logger.info(f"Travis CI cached artifact URL: {url}")
        
        # Test with a non-existent artifact
        url = await travis.get_artifact_url(
            test_run_id=test_run_id,
            artifact_name="non_existent.json"
        )
        
        assert url is None, "Expected None for non-existent artifact"
        
        logger.info("Travis CI get_artifact_url test passed")
    
    finally:
        # Clean up
        os.unlink(file_path)
        await travis.close()

async def run_all_tests():
    """Run all artifact URL retrieval tests."""
    logger.info("Running all artifact URL retrieval tests...")
    
    # Test Jenkins
    await test_jenkins_get_artifact_url()
    
    # Test CircleCI
    await test_circleci_get_artifact_url()
    
    # Test Azure DevOps
    await test_azure_get_artifact_url()
    
    # Test GitHub
    await test_github_get_artifact_url()
    
    # Test Bitbucket
    await test_bitbucket_get_artifact_url()
    
    # Test TeamCity
    await test_teamcity_get_artifact_url()
    
    # Test Travis CI
    await test_travis_get_artifact_url()
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    anyio.run(run_all_tests())