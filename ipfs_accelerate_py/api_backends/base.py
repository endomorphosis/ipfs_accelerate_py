"""
BaseAPIBackend – shared mixin for every API backend.

Provides a common, tested implementation of:
  • Priority queue  (_init_queue / _process_queue / queue_with_priority)
  • Circuit breaker (_init_circuit_breaker / check_circuit_breaker / track_request_result)

All concrete backends should inherit from this class so that queue and security
logic is maintained in a single place.
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)


class BaseAPIBackend:
    """
    Mixin that supplies a shared priority-queue and circuit-breaker to every
    API backend.  It intentionally does *not* define ``__init__`` so that
    subclass ``__init__`` methods can call the two helpers below at the right
    point in their own initialisation sequence.

    Usage in a subclass::

        class MyBackend(BaseAPIBackend):
            def __init__(self, resources=None, metadata=None):
                self.resources = resources or {}
                self.metadata = metadata or {}
                # … backend-specific setup …
                self._init_queue(
                    queue_size=int(self.metadata.get("queue_size", 100)),
                    max_concurrent_requests=int(self.metadata.get("max_concurrent_requests", 5)),
                )
                self._init_circuit_breaker()
    """

    # Priority constants ─ available on every subclass without redeclaring them.
    PRIORITY_HIGH = 0
    PRIORITY_NORMAL = 1
    PRIORITY_LOW = 2

    # ------------------------------------------------------------------ #
    #  Initialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def _init_queue(self, queue_size: int = 100, max_concurrent_requests: int = 5) -> None:
        """Set up the shared priority-queue state.

        Call once from the subclass ``__init__``.  Starts a background
        worker thread so requests are processed as soon as they are queued.
        """
        self.queue_size = queue_size
        self.max_concurrent_requests = max_concurrent_requests
        # List of (priority, request_info_dict) tuples, sorted ascending by priority.
        self.request_queue: list = []
        self.active_requests: int = 0
        self.queue_lock = threading.RLock()
        self.queue_processing: bool = False
        self.queue_enabled: bool = True

        # Kick off the background processor.  It exits immediately when the
        # queue is empty and restarts on demand via queue_with_priority.
        self.queue_processor = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_processor.start()

    def _init_circuit_breaker(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
    ) -> None:
        """Set up the shared circuit-breaker state.

        Call once from the subclass ``__init__``.
        """
        self.circuit_state: str = "CLOSED"   # CLOSED | OPEN | HALF_OPEN
        self.failure_threshold: int = failure_threshold
        self.reset_timeout: float = reset_timeout
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.circuit_lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Priority queue                                                      #
    # ------------------------------------------------------------------ #

    def _process_queue(self) -> None:
        """Background worker: dequeue requests and dispatch them.

        Items in ``self.request_queue`` may be either:
          • A plain ``dict`` with keys ``future``, ``func`` / ``endpoint_url``, …
          • A ``(priority, request_info_dict)`` tuple as inserted by
            :meth:`queue_with_priority`.

        Both formats are handled transparently.
        """
        with self.queue_lock:
            if self.queue_processing:
                return  # Another thread is already processing
            self.queue_processing = True

        try:
            while True:
                request_info = None

                with self.queue_lock:
                    if not self.request_queue:
                        self.queue_processing = False
                        break

                    # Respect concurrency limit
                    if self.active_requests >= self.max_concurrent_requests:
                        time.sleep(0.1)
                        continue

                    # Pop the highest-priority (lowest numeric value) item.
                    raw_item = self.request_queue.pop(0)
                    self.active_requests += 1

                # Unwrap (priority, dict) tuple if needed.
                if isinstance(raw_item, tuple) and len(raw_item) == 2 and isinstance(raw_item[1], dict):
                    _, request_info = raw_item
                elif isinstance(raw_item, dict):
                    request_info = raw_item
                else:
                    # Unrecognised format – skip silently to avoid blocking.
                    logger.warning("_process_queue: unrecognised queue item type %s, skipping", type(raw_item))
                    with self.queue_lock:
                        self.active_requests = max(0, self.active_requests - 1)
                    continue

                # Dispatch outside the lock.
                if request_info:
                    try:
                        future = request_info.get("future")
                        func = request_info.get("func")
                        args = request_info.get("args", [])
                        kwargs = request_info.get("kwargs", {})

                        if func and callable(func):
                            # Function-callback path.
                            try:
                                result = func(*args, **kwargs)
                                if future:
                                    future["result"] = result
                                    future["completed"] = True
                            except Exception as exc:
                                if future:
                                    future["error"] = exc
                                    future["completed"] = True
                                logger.error("Error executing queued function: %s", exc)
                        else:
                            # Direct API-request-info path.
                            endpoint_url = request_info.get("endpoint_url")
                            data = request_info.get("data")
                            api_key = request_info.get("api_key")
                            request_id = request_info.get("request_id")

                            if hasattr(self, "make_request"):
                                method = self.make_request          # type: ignore[attr-defined]
                            elif hasattr(self, "make_post_request"):
                                method = self.make_post_request     # type: ignore[attr-defined]
                            else:
                                raise AttributeError(
                                    "Backend has neither make_request nor make_post_request"
                                )

                            # Temporarily disable queueing to prevent recursion.
                            original_queue_enabled = getattr(self, "queue_enabled", True)
                            setattr(self, "queue_enabled", False)
                            try:
                                result = method(
                                    endpoint_url=endpoint_url,
                                    data=data,
                                    api_key=api_key,
                                    request_id=request_id,
                                )
                                if future:
                                    future["result"] = result
                                    future["completed"] = True
                            except Exception as exc:
                                if future:
                                    future["error"] = exc
                                    future["completed"] = True
                                logger.error("Error processing queued request: %s", exc)
                            finally:
                                setattr(self, "queue_enabled", original_queue_enabled)

                    finally:
                        with self.queue_lock:
                            self.active_requests = max(0, self.active_requests - 1)

                time.sleep(0.01)

        except Exception as exc:
            logger.error("Error in queue processing thread: %s", exc)
        finally:
            with self.queue_lock:
                self.queue_processing = False

    def queue_with_priority(self, request_info: dict, priority: int = None) -> dict:
        """Enqueue *request_info* and return a future dict for the caller to poll.

        ``request_info`` should contain either:
          • ``func`` (callable) + optional ``args`` / ``kwargs``  – function-callback style
          • ``endpoint_url``, ``data``, ``api_key``, ``request_id``  – direct API style

        If ``request_info`` already contains a ``"future"`` key it is reused;
        otherwise a new future dict is created and attached.

        Parameters
        ----------
        request_info:
            Dict describing the work to be done.
        priority:
            Integer priority; lower = higher precedence.  Defaults to
            :attr:`PRIORITY_NORMAL`.

        Returns
        -------
        dict
            Future dict with keys ``result``, ``error``, ``completed``.
        """
        if priority is None:
            priority = self.PRIORITY_NORMAL

        with self.queue_lock:
            if len(self.request_queue) >= self.queue_size:
                raise ValueError(
                    f"Request queue is full ({self.queue_size} items). Try again later."
                )

            # Record when this request entered the queue (useful for metrics).
            request_info["queue_entry_time"] = time.time()

            # Reuse an existing future if the caller supplied one.
            if "future" not in request_info or request_info["future"] is None:
                future: dict = {"result": None, "error": None, "completed": False}
                request_info["future"] = future
            else:
                future = request_info["future"]

            # Add to queue and keep it sorted by priority.
            self.request_queue.append((priority, request_info))
            self.request_queue.sort(key=lambda x: x[0])

            logger.debug(
                "Request queued with priority %s. Queue size: %s",
                priority,
                len(self.request_queue),
            )

            # Ensure the background processor is running.
            if not self.queue_processing:
                t = threading.Thread(target=self._process_queue, daemon=True)
                t.start()

        return future

    # ------------------------------------------------------------------ #
    #  Circuit breaker                                                     #
    # ------------------------------------------------------------------ #

    def check_circuit_breaker(self) -> bool:
        """Return ``True`` if a request may proceed, ``False`` if the circuit is open."""
        with self.circuit_lock:
            now = time.time()

            if self.circuit_state == "OPEN":
                if now - self.last_failure_time > self.reset_timeout:
                    logger.info("Circuit breaker transitioning from OPEN to HALF-OPEN")
                    self.circuit_state = "HALF_OPEN"
                    return True
                return False

            # HALF_OPEN or CLOSED – allow the request.
            return True

    def track_request_result(self, success: bool, error_type: str = None, **_kwargs) -> None:
        """Update circuit-breaker state based on the outcome of a request.

        Parameters
        ----------
        success:
            ``True`` if the request succeeded, ``False`` otherwise.
        error_type:
            Optional string identifying the type of error (used for stats).
        **_kwargs:
            Accepted but ignored; allows subclasses with additional keyword
            parameters (e.g. ``operation``, ``latency``) to call
            ``super().track_request_result(...)`` safely.
        """
        with self.circuit_lock:
            if success:
                if self.circuit_state == "HALF_OPEN":
                    logger.info("Circuit breaker transitioning from HALF-OPEN to CLOSED")
                    self.circuit_state = "CLOSED"
                    self.failure_count = 0
                elif self.circuit_state == "CLOSED":
                    self.failure_count = 0
            else:
                self.failure_count += 1
                self.last_failure_time = time.time()

                # Optionally record per-error-type statistics.
                if error_type and hasattr(self, "collect_metrics") and self.collect_metrics:  # type: ignore[attr-defined]
                    if hasattr(self, "stats_lock") and hasattr(self, "request_stats"):
                        with self.stats_lock:  # type: ignore[attr-defined]
                            errors = self.request_stats.get("errors_by_type", {})  # type: ignore[attr-defined]
                            errors[error_type] = errors.get(error_type, 0) + 1
                            self.request_stats["errors_by_type"] = errors  # type: ignore[attr-defined]

                if self.circuit_state == "CLOSED" and self.failure_count >= self.failure_threshold:
                    logger.warning(
                        "Circuit breaker transitioning from CLOSED to OPEN after %s failures",
                        self.failure_count,
                    )
                    self.circuit_state = "OPEN"

                    if hasattr(self, "stats_lock") and hasattr(self, "request_stats"):
                        with self.stats_lock:  # type: ignore[attr-defined]
                            self.request_stats["circuit_breaker_trips"] = (  # type: ignore[attr-defined]
                                self.request_stats.get("circuit_breaker_trips", 0) + 1  # type: ignore[attr-defined]
                            )

                elif self.circuit_state == "HALF_OPEN":
                    logger.warning(
                        "Circuit breaker transitioning from HALF-OPEN to OPEN after test failure"
                    )
                    self.circuit_state = "OPEN"
