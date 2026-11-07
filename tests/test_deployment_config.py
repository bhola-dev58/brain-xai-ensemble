"""
tests/test_deployment_config.py
===============================
Tests verifying deployment configuration parsing, storage paths,
model artifact discovery, persistence isolation, and container configs.
"""

import os
import tempfile
from unittest.mock import patch
import pytest

try:
    import yaml
except ImportError:
    yaml = None

import src.config as config
from src.database import init_db, log_scan, get_scan_by_analysis_id
from scripts.init_models import verify_models


class TestConfigurationAndPaths:
    """Verifies that paths, ports, and runtime settings are properly configurable."""

    def test_default_paths_exist_under_base_dir(self):
        """Default paths are derived cleanly from project root."""
        assert os.path.isabs(config.BASE_DIR)
        assert config.MODELS_DIR.startswith(config.BASE_DIR)
        assert config.DATA_DIR.startswith(config.BASE_DIR)
        assert config.REPORTS_DIR.startswith(config.BASE_DIR)
        assert config.TEMP_REPORTS_DIR.startswith(config.REPORTS_DIR)
        assert config.HOST == "0.0.0.0"
        assert config.PORT == 8000

    def test_custom_env_paths_override_defaults(self, monkeypatch):
        """Custom environment variables override default storage paths."""
        custom_data = "/tmp/custom_brain_data"
        custom_port = "9090"
        monkeypatch.setenv("DATA_DIR", custom_data)
        monkeypatch.setenv("PORT", custom_port)

        import importlib
        import src.config as fresh_config
        importlib.reload(fresh_config)

        assert fresh_config.DATA_DIR == custom_data
        assert fresh_config.PORT == 9090

        # Reload back to clean state
        monkeypatch.undo()
        importlib.reload(fresh_config)


class TestModelArtifactDiscovery:
    """Verifies model artifact discovery behavior."""

    def test_verify_models_reports_cleanly_without_synthesizing_weights(self, monkeypatch):
        """verify_models inspects model files without synthesizing fake weights."""
        with tempfile.TemporaryDirectory() as empty_dir:
            monkeypatch.setattr(config, "MODELS_DIR", empty_dir)
            monkeypatch.setattr(config, "CONVNEXT_PATH", os.path.join(empty_dir, "convnext_best.h5"))
            monkeypatch.setattr(config, "INCEPTION_PATH", os.path.join(empty_dir, "inception_best.h5"))
            monkeypatch.setattr(config, "DENSENET_PATH", os.path.join(empty_dir, "densenet_best.h5"))

            # Calling verify_models should report False and NOT create files
            ready = verify_models()
            assert ready is False
            # Ensure no fake weights were written to the directory
            assert len(os.listdir(empty_dir)) == 0


class TestStoragePersistence:
    """Verifies database initialization and persistence in custom paths."""

    def test_database_persistence_in_custom_directory(self, monkeypatch):
        """Database can be initialized and read from an explicitly mounted storage directory."""
        with tempfile.TemporaryDirectory() as custom_dir:
            custom_db = os.path.join(custom_dir, "patient_history.db")
            monkeypatch.setattr("src.database.DB_PATH", custom_db)

            # Initialize database in custom directory
            init_db()
            assert os.path.exists(custom_db)

            # Insert scan
            anl_id = "ANL-PERSIST-1234"
            row_id = log_scan(
                predicted_class="Pituitary Tumor",
                confidence_pct=98.5,
                is_tumor=True,
                severity="Mild",
                tumor_area_pct=1.2,
                location="Sellar Region",
                inference_time=0.045,
                patient_id="PT-PERSIST",
                analysis_id=anl_id,
            )
            assert row_id > 0

            # Retrieve scan
            scan = get_scan_by_analysis_id(anl_id)
            assert scan is not None
            assert scan["predicted_class"] == "Pituitary Tumor"
            assert scan["analysis_id"] == anl_id


class TestContainerDeploymentConfigurations:
    """Verifies syntax and safety parameters of container deployment manifests."""

    def test_dockerfile_contains_security_and_healthcheck(self):
        """Dockerfile enforces non-root user, healthcheck, and dynamic binding."""
        dockerfile_path = os.path.join(config.BASE_DIR, "Dockerfile")
        assert os.path.exists(dockerfile_path)

        with open(dockerfile_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Non-root user
        assert "appuser" in content
        assert "USER appuser" in content

        # Healthcheck
        assert "HEALTHCHECK" in content
        assert "/api/health" in content

        # Storage volumes
        assert "VOLUME" in content
        assert "/app/data" in content
        assert "/app/reports" in content

        # Dynamic binding
        assert "HOST" in content
        assert "PORT" in content

    def test_docker_compose_valid_and_mounts_persistent_volumes(self):
        """docker-compose.yml is valid YAML and defines persistent volume mounts."""
        if yaml is None:
            pytest.skip("PyYAML is not installed")
        compose_path = os.path.join(config.BASE_DIR, "docker-compose.yml")
        assert os.path.exists(compose_path)

        with open(compose_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "services" in data
        assert "brain-tumor-xai" in data["services"]
        svc = data["services"]["brain-tumor-xai"]

        # Volumes for data, reports, and models
        volumes = svc.get("volumes", [])
        assert any("./data:/app/data" in v for v in volumes)
        assert any("./reports:/app/reports" in v for v in volumes)
        assert any("models" in v for v in volumes)

        # Health check
        assert "healthcheck" in svc

    def test_render_yaml_valid_and_specifies_disk(self):
        """render.yaml is valid YAML and specifies persistent disk."""
        if yaml is None:
            pytest.skip("PyYAML is not installed")
        render_path = os.path.join(config.BASE_DIR, "render.yaml")
        assert os.path.exists(render_path)

        with open(render_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "services" in data
        svc = data["services"][0]
        assert svc["type"] == "web"
        assert svc["healthCheckPath"] == "/api/health"
        assert "disk" in svc
        assert svc["disk"]["mountPath"] == "/app/data"
