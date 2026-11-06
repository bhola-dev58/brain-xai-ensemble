"""
src/monitor.py
==============
Automated Render Keep-Alive and Health Monitoring Module for BrainTumorXAI.

Responsibilities:
  - Periodically verify application health and process liveness via HTTP GET.
  - Record request timestamp, HTTP status code, response latency, and failure details.
  - Implement exponential backoff retry strategy for failed requests.
  - Ensure single-instance execution to avoid duplicate scheduling across workers.
  - Provide both synchronous (thread/CLI) and asynchronous (FastAPI lifespan) execution loops.
  - Support graceful shutdown on SIGINT/SIGTERM or cancellation token.
"""

import asyncio
import logging
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Setup dedicated monitor logger with fallback if not configured
logger = logging.getLogger("BrainTumorXAI.Monitor")


class MonitorConfig:
    """
    Validates and encapsulates health monitor configuration parameters
    sourced from environment variables or explicit keyword arguments.
    """

    def __init__(
        self,
        app_url: Optional[str] = None,
        health_endpoint: Optional[str] = None,
        check_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        request_timeout: Optional[float] = None,
        retry_base_delay: Optional[float] = None,
        log_level: Optional[str] = None,
    ):
        raw_url = (
            app_url
            or os.getenv("APP_URL", "https://brain-tumor-xai-jf4z.onrender.com")
        ).strip().rstrip("/")
        if not raw_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid APP_URL: '{raw_url}'. Must begin with 'http://' or 'https://'.")
        self.app_url = raw_url

        endpoint = (
            health_endpoint
            or os.getenv("HEALTH_ENDPOINT", "/health")
        ).strip()
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        self.health_endpoint = endpoint

        raw_interval = check_interval if check_interval is not None else os.getenv("CHECK_INTERVAL", "300")
        try:
            self.check_interval = int(raw_interval)
            if self.check_interval <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"CHECK_INTERVAL must be a positive integer, got: {raw_interval}")

        raw_retries = max_retries if max_retries is not None else os.getenv("MAX_RETRIES", "3")
        try:
            self.max_retries = int(raw_retries)
            if self.max_retries < 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"MAX_RETRIES must be a non-negative integer, got: {raw_retries}")

        raw_timeout = request_timeout if request_timeout is not None else os.getenv("REQUEST_TIMEOUT", "30")
        try:
            self.request_timeout = float(raw_timeout)
            if self.request_timeout <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"REQUEST_TIMEOUT must be a positive number, got: {raw_timeout}")

        raw_delay = retry_base_delay if retry_base_delay is not None else os.getenv("RETRY_BASE_DELAY", "2")
        try:
            self.retry_base_delay = float(raw_delay)
            if self.retry_base_delay < 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"RETRY_BASE_DELAY must be a non-negative number, got: {raw_delay}")

        self.log_level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()

    @property
    def target_url(self) -> str:
        """Returns the full target health check URL."""
        return f"{self.app_url}{self.health_endpoint}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_url": self.target_url,
            "check_interval": self.check_interval,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
            "retry_base_delay": self.retry_base_delay,
            "log_level": self.log_level,
        }


class HealthMonitor:
    """
    Automated health monitor that sends periodic HTTP GET requests to keep the
    deployed service active, verify process health, and log response telemetry.
    """

    _global_running_lock = threading.Lock()
    _is_running_globally = False

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self._stop_event = threading.Event()
        self._is_active = False

    def calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculates exponential backoff with slight jitter to prevent thundering herds.
        Formula: base_delay * (2 ** attempt) + random_jitter(0, 0.5)
        """
        delay = self.config.retry_base_delay * (2 ** attempt)
        jitter = random.uniform(0.0, 0.5)
        return round(delay + jitter, 2)

    def ping_once(self) -> Dict[str, Any]:
        """
        Executes a single HTTP GET request against the configured health endpoint.
        Returns a structured dictionary of execution telemetry.
        """
        target = self.config.target_url
        timestamp = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        req = urllib.request.Request(
            target,
            headers={"User-Agent": "BrainTumorXAI-HealthMonitor/3.0", "Accept": "application/json"},
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                status_code = response.getcode()
                elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                is_success = 200 <= status_code < 300

                return {
                    "timestamp": timestamp,
                    "target_url": target,
                    "status_code": status_code,
                    "duration_ms": elapsed_ms,
                    "success": is_success,
                    "error": None,
                }
        except urllib.error.HTTPError as http_err:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "timestamp": timestamp,
                "target_url": target,
                "status_code": http_err.code,
                "duration_ms": elapsed_ms,
                "success": False,
                "error": f"HTTP {http_err.code}: {http_err.reason}",
            }
        except urllib.error.URLError as url_err:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "timestamp": timestamp,
                "target_url": target,
                "status_code": 0,
                "duration_ms": elapsed_ms,
                "success": False,
                "error": f"Connection error: {url_err.reason}",
            }
        except TimeoutError:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "timestamp": timestamp,
                "target_url": target,
                "status_code": 0,
                "duration_ms": elapsed_ms,
                "success": False,
                "error": f"Request timed out after {self.config.request_timeout}s",
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "timestamp": timestamp,
                "target_url": target,
                "status_code": 0,
                "duration_ms": elapsed_ms,
                "success": False,
                "error": f"Unexpected error: {str(exc)}",
            }

    def check_with_retries(self) -> Dict[str, Any]:
        """
        Executes health check with exponential backoff retries on failure up to max_retries.
        """
        total_attempts = 1 + self.config.max_retries
        last_result: Dict[str, Any] = {}

        for attempt in range(total_attempts):
            result = self.ping_once()
            last_result = result
            result["attempt"] = attempt + 1
            result["max_retries"] = self.config.max_retries

            if result["success"]:
                logger.info(
                    f"[HealthMonitor] SUCCESS | target={result['target_url']} | "
                    f"status={result['status_code']} | latency={result['duration_ms']}ms | attempt={attempt + 1}"
                )
                return result

            # If failed and attempts remain, backoff and retry
            remaining_retries = total_attempts - (attempt + 1)
            if remaining_retries > 0 and not self._stop_event.is_set():
                delay = self.calculate_backoff_delay(attempt)
                logger.warning(
                    f"[HealthMonitor] RETRY | target={result['target_url']} | "
                    f"error={result['error']} | attempt={attempt + 1}/{total_attempts} | backing off {delay}s"
                )
                # Sleep in short slices to support immediate cancellation
                self._interruptible_sleep(delay)
            else:
                logger.error(
                    f"[HealthMonitor] FINAL_FAILURE | target={result['target_url']} | "
                    f"status={result['status_code']} | error={result['error']} | exhausted {attempt + 1} attempts"
                )

        return last_result

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleeps in small increments allowing quick shutdown when stop_event is set."""
        step = 0.25
        elapsed = 0.0
        while elapsed < seconds and not self._stop_event.is_set():
            time.sleep(min(step, seconds - elapsed))
            elapsed += step

    def stop(self) -> None:
        """Signals the monitoring loop to terminate gracefully."""
        self._stop_event.set()
        with HealthMonitor._global_running_lock:
            HealthMonitor._is_running_globally = False
        self._is_active = False
        logger.info("[HealthMonitor] Shutdown signal received. Monitor stopped.")

    def run_loop(self) -> None:
        """
        Blocking execution loop for standalone CLI execution or background thread.
        Guarded against duplicate concurrent monitor instances.
        """
        with HealthMonitor._global_running_lock:
            if HealthMonitor._is_running_globally:
                logger.warning("[HealthMonitor] A monitor instance is already active. Duplicate start aborted.")
                return
            HealthMonitor._is_running_globally = True

        self._is_active = True
        self._stop_event.clear()

        logger.info(
            f"[HealthMonitor] Started periodic monitor -> target={self.config.target_url} "
            f"interval={self.config.check_interval}s timeout={self.config.request_timeout}s"
        )

        try:
            while not self._stop_event.is_set():
                self.check_with_retries()
                self._interruptible_sleep(float(self.config.check_interval))
        finally:
            with HealthMonitor._global_running_lock:
                HealthMonitor._is_running_globally = False
            self._is_active = False
            logger.info("[HealthMonitor] Loop exited cleanly.")

    async def async_run_loop(self) -> None:
        """
        Non-blocking execution loop for seamless integration with FastAPI lifespan.
        """
        with HealthMonitor._global_running_lock:
            if HealthMonitor._is_running_globally:
                logger.warning("[HealthMonitor] A monitor instance is already active globally. Skipping.")
                return
            HealthMonitor._is_running_globally = True

        self._is_active = True
        self._stop_event.clear()
        logger.info(f"[HealthMonitor] Embedded lifespan monitor active for {self.config.target_url}")

        loop = asyncio.get_running_loop()
        try:
            while not self._stop_event.is_set():
                # Run the blocking check_with_retries in threadpool executor to avoid blocking ASGI loop
                await loop.run_in_executor(None, self.check_with_retries)
                # Async wait with cancellation check
                for _ in range(int(self.config.check_interval)):
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("[HealthMonitor] Async monitor task cancelled.")
        finally:
            with HealthMonitor._global_running_lock:
                HealthMonitor._is_running_globally = False
            self._is_active = False
            logger.info("[HealthMonitor] Embedded async loop shut down.")

