"""Minimal Kubernetes backend for IPFS Accelerate.

This module intentionally keeps the implementation small and dependency-light:
- generate Kubernetes Job specs for model execution
- track submitted jobs in memory when a cluster client is unavailable
- expose a Docker-compatible execution result envelope
- keep the surface ready for a real client-backed implementation later
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...docker_executor import DockerExecutionResult

logger = logging.getLogger(__name__)


class KubernetesJobStatus(Enum):
	PENDING = "pending"
	RUNNING = "running"
	SUCCEEDED = "succeeded"
	FAILED = "failed"
	UNKNOWN = "unknown"


@dataclass
class KubernetesExecutionConfig:
	image: str
	command: Optional[List[str]] = None
	args: Optional[List[str]] = None
	namespace: str = "default"
	job_name: Optional[str] = None
	service_account_name: Optional[str] = None
	working_dir: Optional[str] = None
	environment: Dict[str, str] = field(default_factory=dict)
	volumes: Dict[str, str] = field(default_factory=dict)
	labels: Dict[str, str] = field(default_factory=dict)
	annotations: Dict[str, str] = field(default_factory=dict)
	model_artifact_cid: Optional[str] = None
	model_artifact_mount_path: Optional[str] = "/workspace/model_artifact"
	timeout: int = 300
	backoff_limit: int = 0
	restart_policy: str = "Never"


@dataclass
class KubernetesJobRecord:
	job_id: str
	config: KubernetesExecutionConfig
	job_spec: Dict[str, Any]
	status: KubernetesJobStatus = KubernetesJobStatus.PENDING
	created_at: float = field(default_factory=time.time)
	started_at: Optional[float] = None
	completed_at: Optional[float] = None
	pod_name: Optional[str] = None
	node_name: Optional[str] = None
	output_cid: Optional[str] = None
	provenance_cid: Optional[str] = None
	stdout: str = ""
	stderr: str = ""
	exit_code: int = 0
	metadata: Dict[str, Any] = field(default_factory=dict)


class KubernetesBackend:
	"""Minimal Kubernetes backend wrapper.

	When the real kubernetes client is unavailable, this still provides:
	- spec generation
	- job submission bookkeeping
	- status polling hooks
	- Docker-like result envelopes
	"""

	def __init__(
		self,
		namespace: str = "default",
		*,
		cluster_context: Optional[str] = None,
		storage: Any | None = None,
		datasets_manager: Any | None = None,
		provenance_logger: Any | None = None,
		persist_results: bool = True,
	):
		self.namespace = namespace
		self.cluster_context = cluster_context
		self._artifact_storage = storage
		self._datasets_manager = datasets_manager
		self._provenance_logger = provenance_logger
		self._persist_results = bool(persist_results)
		self._jobs: Dict[str, KubernetesJobRecord] = {}
		self._client_available = self._try_initialize_client(cluster_context=cluster_context)

	def _record_execution_artifacts(self, record: KubernetesJobRecord) -> None:
		if not self._persist_results:
			return

		artifact_payload = {
			"image": record.config.image,
			"command": list(record.config.command or []),
			"entrypoint": [],
			"container_id": record.pod_name,
			"execution_time": (record.completed_at or time.time()) - record.started_at if record.started_at else 0.0,
			"exit_code": record.exit_code,
			"success": record.status == KubernetesJobStatus.SUCCEEDED,
			"stdout": record.stdout,
			"stderr": record.stderr,
			"error_message": None if record.status == KubernetesJobStatus.SUCCEEDED else (record.stderr or f"Kubernetes job {record.job_id} failed"),
			"model_artifact": dict(record.metadata.get("model_artifact") or {}),
		}

		output_cid = record.output_cid
		provenance_cid = record.provenance_cid

		if self._artifact_storage is not None and not output_cid:
			try:
				output_payload = json.dumps(artifact_payload, sort_keys=True, indent=2)
				output_cid = self._artifact_storage.store(
					output_payload,
					filename=f"kubernetes-execution-{record.job_id}.json",
					pin=False,
				)
			except Exception as exc:
				logger.debug("Failed to persist Kubernetes execution output: %s", exc)

		if self._datasets_manager is not None:
			try:
				self._datasets_manager.log_event(
					"container_execution_completed" if record.status == KubernetesJobStatus.SUCCEEDED else "container_execution_failed",
					{
						**artifact_payload,
						"job_id": record.job_id,
						"namespace": record.config.namespace,
						"node_name": record.node_name,
						"output_cid": output_cid,
					},
					level="INFO" if record.status == KubernetesJobStatus.SUCCEEDED else "ERROR",
					category="PERFORMANCE",
				)
			except Exception as exc:
				logger.debug("Failed to log Kubernetes execution event: %s", exc)

		if self._provenance_logger is not None and not provenance_cid:
			try:
				provenance_cid = self._provenance_logger.log_transformation(
					"kubernetes_execution",
					artifact_payload,
					output_cid=output_cid,
				)
			except Exception as exc:
				logger.debug("Failed to record Kubernetes execution provenance: %s", exc)

		record.output_cid = output_cid
		record.provenance_cid = provenance_cid
		record.metadata.update({
			"image": record.config.image,
			"command": list(record.config.command or []),
			"entrypoint": [],
			"container_id": record.pod_name,
			"execution_time": artifact_payload["execution_time"],
			"exit_code": record.exit_code,
			"success": record.status == KubernetesJobStatus.SUCCEEDED,
			"stdout": record.stdout,
			"stderr": record.stderr,
			"error_message": artifact_payload["error_message"],
			"output_cid": output_cid,
			"provenance_cid": provenance_cid,
			"job_id": record.job_id,
			"namespace": record.config.namespace,
			"pod_name": record.pod_name,
			"node_name": record.node_name,
		})

	def _materialize_model_artifact(self, config: KubernetesExecutionConfig) -> tuple[KubernetesExecutionConfig, Dict[str, Any]]:
		metadata: Dict[str, Any] = {
			"requested": bool(config.model_artifact_cid),
			"cid": config.model_artifact_cid,
			"retrieved": False,
			"mount_path": config.model_artifact_mount_path,
			"host_path": None,
			"error": None,
		}

		artifact_cid = str(config.model_artifact_cid or "").strip()
		if not artifact_cid:
			return config, metadata

		if self._artifact_storage is None:
			metadata["error"] = "artifact storage unavailable"
			return config, metadata

		try:
			artifact = self._artifact_storage.retrieve(artifact_cid)
			if artifact is None:
				metadata["error"] = "artifact not found"
				return config, metadata

			if isinstance(artifact, (bytes, bytearray)):
				suffix = Path(config.model_artifact_mount_path or "/workspace/model_artifact").name or "model_artifact"
				fd, temp_path = tempfile.mkstemp(prefix="ipfs_k8s_model_artifact_", suffix=f"_{suffix}")
				os.close(fd)
				with open(temp_path, "wb") as handle:
					handle.write(bytes(artifact))
				host_path = temp_path
			else:
				host_path = str(artifact)

			updated_volumes = dict(config.volumes)
			updated_volumes[host_path] = str(config.model_artifact_mount_path or "/workspace/model_artifact")
			updated = KubernetesExecutionConfig(
				image=config.image,
				command=list(config.command or []) or None,
				args=list(config.args or []) or None,
				namespace=config.namespace,
				job_name=config.job_name,
				service_account_name=config.service_account_name,
				working_dir=config.working_dir,
				environment=dict(config.environment),
				volumes=updated_volumes,
				labels=dict(config.labels),
				annotations=dict(config.annotations),
				model_artifact_cid=config.model_artifact_cid,
				model_artifact_mount_path=config.model_artifact_mount_path,
				timeout=config.timeout,
				backoff_limit=config.backoff_limit,
				restart_policy=config.restart_policy,
			)
			metadata["retrieved"] = True
			metadata["host_path"] = host_path
			return updated, metadata
		except Exception as exc:
			metadata["error"] = str(exc)
			return config, metadata

	def _try_initialize_client(self, *, cluster_context: Optional[str]) -> bool:
		try:
			from kubernetes import client, config  # type: ignore

			try:
				if cluster_context:
					config.load_kube_config(context=cluster_context)
				else:
					config.load_kube_config()
			except Exception:
				try:
					config.load_incluster_config()
				except Exception:
					return False

			self._api_client = client.ApiClient()
			self._batch_v1 = client.BatchV1Api(self._api_client)
			self._core_v1 = client.CoreV1Api(self._api_client)
			return True
		except Exception:
			self._api_client = None
			self._batch_v1 = None
			self._core_v1 = None
			return False

	def build_job_spec(self, config: KubernetesExecutionConfig) -> Dict[str, Any]:
		job_name = config.job_name or f"ipfs-accel-{uuid.uuid4().hex[:12]}"
		container: Dict[str, Any] = {
			"name": job_name,
			"image": config.image,
			"imagePullPolicy": "IfNotPresent",
		}
		if config.command:
			container["command"] = list(config.command)
		if config.args:
			container["args"] = list(config.args)
		if config.working_dir:
			container["workingDir"] = config.working_dir
		if config.environment:
			container["env"] = [{"name": key, "value": value} for key, value in sorted(config.environment.items())]

		template: Dict[str, Any] = {
			"metadata": {
				"labels": {
					"app": "ipfs-accelerate",
					**config.labels,
				},
				"annotations": dict(config.annotations),
			},
			"spec": {
				"restartPolicy": config.restart_policy,
				"containers": [container],
			},
		}

		if config.service_account_name:
			template["spec"]["serviceAccountName"] = config.service_account_name

		if config.volumes:
			volumes = []
			mounts = []
			for idx, (host_path, container_path) in enumerate(sorted(config.volumes.items())):
				volume_name = f"volume-{idx}"
				volumes.append({"name": volume_name, "hostPath": {"path": host_path}})
				mounts.append({"name": volume_name, "mountPath": container_path})
			if mounts:
				template["spec"]["containers"][0]["volumeMounts"] = mounts
				template["spec"]["volumes"] = volumes

		return {
			"apiVersion": "batch/v1",
			"kind": "Job",
			"metadata": {
				"name": job_name,
				"namespace": config.namespace,
			},
			"spec": {
				"backoffLimit": config.backoff_limit,
				"template": template,
			},
		}

	def submit_job(self, config: KubernetesExecutionConfig) -> str:
		config, artifact_metadata = self._materialize_model_artifact(config)
		job_spec = self.build_job_spec(config)
		job_id = job_spec["metadata"]["name"]
		record = KubernetesJobRecord(job_id=job_id, config=config, job_spec=job_spec)
		record.metadata["model_artifact"] = artifact_metadata
		record.status = KubernetesJobStatus.RUNNING
		record.started_at = time.time()
		record.pod_name = f"{job_id}-pod"
		record.node_name = "local-simulated-node" if not self._client_available else None
		self._jobs[job_id] = record

		if self._client_available:
			try:
				self._batch_v1.create_namespaced_job(namespace=config.namespace, body=job_spec)
			except Exception as exc:
				record.status = KubernetesJobStatus.FAILED
				record.stderr = str(exc)
				record.exit_code = 1
				record.completed_at = time.time()
				logger.debug("Failed to create Kubernetes job %s: %s", job_id, exc)
		else:
			logger.info("Kubernetes client unavailable; recorded simulated job %s", job_id)

		return job_id

	def get_job_status(self, job_id: str) -> KubernetesJobStatus:
		record = self._jobs.get(job_id)
		if record is None:
			return KubernetesJobStatus.UNKNOWN

		if self._client_available and record.status in {KubernetesJobStatus.RUNNING, KubernetesJobStatus.PENDING}:
			try:
				job = self._batch_v1.read_namespaced_job(name=job_id, namespace=record.config.namespace)
				status = getattr(job, "status", None)
				if status is not None and getattr(status, "succeeded", 0):
					record.status = KubernetesJobStatus.SUCCEEDED
					record.completed_at = time.time()
				elif status is not None and getattr(status, "failed", 0):
					record.status = KubernetesJobStatus.FAILED
					record.completed_at = time.time()
				elif status is not None and getattr(status, "active", 0):
					record.status = KubernetesJobStatus.RUNNING
				else:
					record.status = KubernetesJobStatus.PENDING
			except Exception as exc:
				record.status = KubernetesJobStatus.FAILED
				record.stderr = str(exc)
				record.exit_code = 1
				record.completed_at = time.time()

		return record.status

	def collect_result(self, job_id: str) -> DockerExecutionResult:
		record = self._jobs.get(job_id)
		if record is None:
			return DockerExecutionResult(
				success=False,
				exit_code=1,
				stdout="",
				stderr=f"Job not found: {job_id}",
				execution_time=0.0,
				error_message="job not found",
			)

		status = self.get_job_status(job_id)
		self._record_execution_artifacts(record)
		execution_time = 0.0
		if record.started_at:
			execution_time = (record.completed_at or time.time()) - record.started_at

		success = status == KubernetesJobStatus.SUCCEEDED or (status == KubernetesJobStatus.RUNNING and not record.stderr)
		exit_code = 0 if success else (record.exit_code or 1)

		return DockerExecutionResult(
			success=success,
			exit_code=exit_code,
			stdout=record.stdout,
			stderr=record.stderr,
			container_id=record.pod_name,
			execution_time=execution_time,
			error_message=None if success else (record.stderr or f"Kubernetes job {job_id} failed"),
			output_cid=record.output_cid,
			provenance_cid=record.provenance_cid,
			metadata=dict(record.metadata),
		)

	def execute_job(self, config: KubernetesExecutionConfig) -> DockerExecutionResult:
		job_id = self.submit_job(config)
		return self.collect_result(job_id)

	def list_jobs(self) -> List[Dict[str, Any]]:
		return [
			{
				"job_id": record.job_id,
				"namespace": record.config.namespace,
				"status": record.status.value,
				"pod_name": record.pod_name,
				"node_name": record.node_name,
				"created_at": record.created_at,
				"started_at": record.started_at,
				"completed_at": record.completed_at,
				"job_spec": record.job_spec,
				"metadata": record.metadata,
			}
			for record in self._jobs.values()
		]

	def record_job_artifacts(
		self,
		job_id: str,
		*,
		stdout: Optional[str] = None,
		stderr: Optional[str] = None,
		output_cid: Optional[str] = None,
		provenance_cid: Optional[str] = None,
		exit_code: Optional[int] = None,
	) -> None:
		record = self._jobs.get(job_id)
		if record is None:
			return

		if stdout is not None:
			record.stdout = stdout
		if stderr is not None:
			record.stderr = stderr
		if output_cid is not None:
			record.output_cid = output_cid
		if provenance_cid is not None:
			record.provenance_cid = provenance_cid
		if exit_code is not None:
			record.exit_code = exit_code

		if record.completed_at is None:
			record.completed_at = time.time()
		if exit_code == 0 or exit_code is None:
			record.status = KubernetesJobStatus.SUCCEEDED
		else:
			record.status = KubernetesJobStatus.FAILED

	def to_json(self) -> str:
		return json.dumps(self.list_jobs(), indent=2, sort_keys=True, default=str)


__all__ = [
	"KubernetesBackend",
	"KubernetesExecutionConfig",
	"KubernetesJobRecord",
	"KubernetesJobStatus",
]
