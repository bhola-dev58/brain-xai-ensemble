"""
tests/test_health_and_monitor.py
================================
Comprehensive test suite for the Render Keep-Alive and Health Monitoring system:
1. GET /health endpoint returns 200 with process liveness and readiness status.
2. GET /health does not run ML inference or reload models.
3. Configuration parsing and boundary validation.
4. Successful health ping execution.
5. Failed HTTP responses (500, 503).
6. Connection / network error handling.
7. Timeout error handling.
8. Exponential backoff retry behavior and maximum retry limit enforcement.
9. Graceful shutdown on stop signal.
10. Single-instance concurrency guard preventing duplicate scheduler runs.
11. Embedded lifespan monitor task execution and cleanup.
"""

import asyncio
import io
import urllib.error
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.monitor import HealthMonitor, MonitorConfig


@pytest.fixture
def client():
    return TestClient(app)


# ------------------------------------------------------------------ #
# 1 & 2. GET /health Endpoint Tests
# ------------------------------------------------------------------ #
class TestHealthEndpoint:
    def test_root_health_endpoint_returns_200(self, client):
        """GET /health must return HTTP 200 with standard liveness payload."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "BrainTumorXAI"
        assert data["liveness"] is True
        assert "status" in data
        assert "ready" in data
        assert "timestamp" in data

    def test_root_health_does_not_invoke_model_inference(self, client):
        """GET /health must never trigger predict_tumor_logic or tf.keras inference."""
        with patch("src.inference.predict_tumor_logic") as mock_infer:
            response = client.get("/health")
            assert response.status_code == 200
            mock_infer.assert_not_called()

    def test_legacy_api_health_remains_intact(self, client):
        """GET /api/health must remain operational for backward compatibility."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "models_active" in data
        assert "pipeline_version" in data


# ------------------------------------------------------------------ #
# 3. MonitorConfig Validation Tests
# ------------------------------------------------------------------ #
class TestMonitorConfigValidation:
    def test_valid_custom_config(self):
        """Custom configuration parameters parse accurately."""
        cfg = MonitorConfig(
            app_url="https://custom-app.onrender.com/",
            health_endpoint="/custom-health",
            check_interval=120,
            max_retries=5,
            request_timeout=15.0,
            retry_base_delay=1.5,
        )
        assert cfg.app_url == "https://custom-app.onrender.com"
        assert cfg.health_endpoint == "/custom-health"
        assert cfg.target_url == "https://custom-app.onrender.com/custom-health"
        assert cfg.check_interval == 120
        assert cfg.max_retries == 5
        assert cfg.request_timeout == 15.0
        assert cfg.retry_base_delay == 1.5

    def test_invalid_app_url_scheme_raises(self):
        """URL missing http:// or https:// must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid APP_URL"):
            MonitorConfig(app_url="ftp://invalid-url.com")

    def test_invalid_check_interval_raises(self):
        """Non-positive check interval must raise ValueError."""
        with pytest.raises(ValueError, match="CHECK_INTERVAL"):
            MonitorConfig(check_interval=0)
        with pytest.raises(ValueError, match="CHECK_INTERVAL"):
            MonitorConfig(check_interval=-10)

    def test_invalid_max_retries_raises(self):
        """Negative max retries must raise ValueError."""
        with pytest.raises(ValueError, match="MAX_RETRIES"):
            MonitorConfig(max_retries=-1)

    def test_invalid_request_timeout_raises(self):
        """Non-positive timeout must raise ValueError."""
        with pytest.raises(ValueError, match="REQUEST_TIMEOUT"):
            MonitorConfig(request_timeout=0)

    def test_invalid_retry_base_delay_raises(self):
        """Negative base delay must raise ValueError."""
        with pytest.raises(ValueError, match="RETRY_BASE_DELAY"):
            MonitorConfig(retry_base_delay=-2)

    def test_missing_leading_slash_in_endpoint_is_normalized(self):
        """Endpoint missing leading slash must be normalized cleanly."""
        cfg = MonitorConfig(app_url="https://app.com", health_endpoint="status")
        assert cfg.health_endpoint == "/status"
        assert cfg.target_url == "https://app.com/status"


# ------------------------------------------------------------------ #
# 4, 5, 6, 7. HealthMonitor Single Ping Tests (Mocked)
# ------------------------------------------------------------------ #
class TestHealthMonitorPingOnce:
    def test_successful_ping_records_telemetry(self):
        """Successful HTTP 200 GET records status_code and positive duration."""
        cfg = MonitorConfig(app_url="https://test.render.com", health_endpoint="/health")
        monitor = HealthMonitor(config=cfg)

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = monitor.ping_once()
            assert res["success"] is True
            assert res["status_code"] == 200
            assert res["duration_ms"] >= 0.0
            assert res["target_url"] == "https://test.render.com/health"
            assert res["error"] is None

    def test_failed_http_status_code(self):
        """HTTP error responses (e.g. 503 Service Unavailable) are handled cleanly."""
        cfg = MonitorConfig(app_url="https://test.render.com", health_endpoint="/health")
        monitor = HealthMonitor(config=cfg)

        http_err = urllib.error.HTTPError(
            url="https://test.render.com/health",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b"Unavailable"),
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            res = monitor.ping_once()
            assert res["success"] is False
            assert res["status_code"] == 503
            assert "HTTP 503" in res["error"]

    def test_connection_error_handling(self):
        """Network unreachable or DNS failure is captured without crashing."""
        cfg = MonitorConfig(app_url="https://nonexistent.invalid", health_endpoint="/health")
        monitor = HealthMonitor(config=cfg)

        url_err = urllib.error.URLError(reason="Name or service not known")

        with patch("urllib.request.urlopen", side_effect=url_err):
            res = monitor.ping_once()
            assert res["success"] is False
            assert res["status_code"] == 0
            assert "Connection error" in res["error"]

    def test_timeout_error_handling(self):
        """Request timeout is captured with clear error reason."""
        cfg = MonitorConfig(app_url="https://test.render.com", request_timeout=10)
        monitor = HealthMonitor(config=cfg)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            res = monitor.ping_once()
            assert res["success"] is False
            assert "timed out" in res["error"].lower()


# ------------------------------------------------------------------ #
# 8. Retry Logic and Exponential Backoff Tests
# ------------------------------------------------------------------ #
class TestRetryAndBackoffLogic:
    def test_exponential_backoff_increases_with_attempts(self):
        """Backoff delay must scale exponentially with attempt count."""
        cfg = MonitorConfig(retry_base_delay=2.0)
        monitor = HealthMonitor(config=cfg)

        delay_0 = monitor.calculate_backoff_delay(0)  # ~2.0s + jitter
        delay_1 = monitor.calculate_backoff_delay(1)  # ~4.0s + jitter
        delay_2 = monitor.calculate_backoff_delay(2)  # ~8.0s + jitter

        assert 2.0 <= delay_0 < 3.0
        assert 4.0 <= delay_1 < 5.0
        assert 8.0 <= delay_2 < 9.0
        assert delay_0 < delay_1 < delay_2

    def test_retry_succeeds_on_second_attempt(self):
        """Transient failure retries and returns success when next attempt passes."""
        cfg = MonitorConfig(max_retries=3, retry_base_delay=0.01)
        monitor = HealthMonitor(config=cfg)

        fail_result = {
            "timestamp": "2026-11-06T00:00:00Z",
            "target_url": cfg.target_url,
            "status_code": 500,
            "duration_ms": 10.0,
            "success": False,
            "error": "HTTP 500: Server Error",
        }
        success_result = {
            "timestamp": "2026-11-06T00:00:01Z",
            "target_url": cfg.target_url,
            "status_code": 200,
            "duration_ms": 12.0,
            "success": True,
            "error": None,
        }

        with patch.object(monitor, "ping_once", side_effect=[fail_result, success_result]):
            final = monitor.check_with_retries()
            assert final["success"] is True
            assert final["attempt"] == 2

    def test_exhausted_retries_returns_final_failure(self):
        """When all retries fail, check_with_retries returns failed result after max_retries + 1."""
        cfg = MonitorConfig(max_retries=2, retry_base_delay=0.01)
        monitor = HealthMonitor(config=cfg)

        fail_result = {
            "timestamp": "2026-11-06T00:00:00Z",
            "target_url": cfg.target_url,
            "status_code": 503,
            "duration_ms": 5.0,
            "success": False,
            "error": "HTTP 503: Unavailable",
        }

        with patch.object(monitor, "ping_once", return_value=fail_result) as mock_ping:
            final = monitor.check_with_retries()
            assert final["success"] is False
            assert final["attempt"] == 3  # 1 initial + 2 retries
            assert mock_ping.call_count == 3


# ------------------------------------------------------------------ #
# 9 & 10. Concurrency Guard and Graceful Shutdown Tests
# ------------------------------------------------------------------ #
class TestLifecycleAndConcurrencyGuard:
    def test_graceful_shutdown_stops_loop(self):
        """Calling stop() signals the execution loop to terminate promptly."""
        cfg = MonitorConfig(check_interval=100)
        monitor = HealthMonitor(config=cfg)

        # Trigger stop immediately after first iteration
        def _mock_check():
            monitor.stop()
            return {"success": True}

        with patch.object(monitor, "check_with_retries", side_effect=_mock_check):
            monitor.run_loop()

        assert monitor._stop_event.is_set()
        assert monitor._is_active is False

    def test_single_instance_guard_prevents_duplicate_runs(self):
        """If a monitor instance is already active globally, another instance aborts."""
        cfg = MonitorConfig(check_interval=10)
        m1 = HealthMonitor(config=cfg)
        m2 = HealthMonitor(config=cfg)

        # Mark global lock as running
        with HealthMonitor._global_running_lock:
            HealthMonitor._is_running_globally = True

        try:
            # m2.run_loop should return immediately without executing check_with_retries
            with patch.object(m2, "check_with_retries") as mock_check:
                m2.run_loop()
                mock_check.assert_not_called()
        finally:
            with HealthMonitor._global_running_lock:
                HealthMonitor._is_running_globally = False

    def test_async_run_loop_cancellation(self):
        """Async execution loop terminates cleanly when cancelled."""
        async def _run():
            cfg = MonitorConfig(check_interval=5)
            monitor = HealthMonitor(config=cfg)

            with patch.object(monitor, "check_with_retries", return_value={"success": True}):
                task = asyncio.create_task(monitor.async_run_loop())
                # Let it initialize
                await asyncio.sleep(0.05)
                assert monitor._is_active is True

                # Stop and cancel
                monitor.stop()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                assert monitor._is_active is False

        asyncio.run(_run())
