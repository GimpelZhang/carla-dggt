# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for DGGT Configuration Module

Tests:
- DggtConfig dataclass validation
- DggtConfigLoader.from_yaml() YAML loading
- DggtConfigLoader.from_env() environment variable override
- DggtConfigLoader.default() default values
- DggtConfigLoader.from_yaml_with_env() combined loading
"""

import pytest
import os
import tempfile
from pathlib import Path
import numpy as np

from dggt_config import DggtConfig, DggtConfigLoader


class TestDggtConfig:
    """Tests for DggtConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = DggtConfig()
        assert config.server_host == "localhost"
        assert config.server_port == 50051
        assert config.scene_base_path == ""
        assert config.default_scene_id == ""
        assert config.max_message_length == 10 * 1024 * 1024
        assert config.request_timeout == 30.0
        assert config.image_format == "JPEG"
        assert config.image_quality == 95.0
        assert config.coordinate_transform_enabled is True
        assert config.max_dynamic_objects == 100
        assert config.t_scenario_dggt is None

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = DggtConfig(
            server_host="192.168.1.100",
            server_port=8080,
            scene_base_path="/data/scenes",
            default_scene_id="scene_001",
            image_format="PNG",
            image_quality=80.0,
        )
        assert config.server_host == "192.168.1.100"
        assert config.server_port == 8080
        assert config.scene_base_path == "/data/scenes"
        assert config.default_scene_id == "scene_001"
        assert config.image_format == "PNG"
        assert config.image_quality == 80.0

    def test_transform_matrix(self):
        """Test setting transform matrix."""
        matrix = np.eye(4, dtype=np.float32)
        config = DggtConfig(t_scenario_dggt=matrix)
        assert config.t_scenario_dggt.shape == (4, 4)
        assert np.allclose(config.t_scenario_dggt, matrix)

    def test_validate_success(self):
        """Test validation passes with valid config."""
        config = DggtConfig(scene_base_path="/valid/path")
        # Should not raise
        config.validate()

    def test_validate_missing_scene_base_path(self):
        """Test validation fails without scene_base_path."""
        config = DggtConfig()
        with pytest.raises(ValueError, match="scene_base_path is required"):
            config.validate()

    def test_validate_invalid_image_quality_low(self):
        """Test validation fails with image_quality < 0."""
        config = DggtConfig(
            scene_base_path="/valid/path",
            image_quality=-10.0,
        )
        with pytest.raises(ValueError, match="image_quality must be in"):
            config.validate()

    def test_validate_invalid_image_quality_high(self):
        """Test validation fails with image_quality > 100."""
        config = DggtConfig(
            scene_base_path="/valid/path",
            image_quality=150.0,
        )
        with pytest.raises(ValueError, match="image_quality must be in"):
            config.validate()

    def test_validate_invalid_image_format(self):
        """Test validation fails with unsupported image format."""
        config = DggtConfig(
            scene_base_path="/valid/path",
            image_format="BMP",
        )
        with pytest.raises(ValueError, match="Unsupported image format"):
            config.validate()

    def test_validate_invalid_server_port(self):
        """Test validation fails with invalid server port."""
        config = DggtConfig(
            scene_base_path="/valid/path",
            server_port=-1,
        )
        with pytest.raises(ValueError, match="Invalid server_port"):
            config.validate()

    def test_validate_invalid_request_timeout(self):
        """Test validation fails with invalid timeout."""
        config = DggtConfig(
            scene_base_path="/valid/path",
            request_timeout=-5.0,
        )
        with pytest.raises(ValueError, match="Invalid request_timeout"):
            config.validate()

    def test_validate_invalid_transform_matrix_shape(self):
        """Test validation fails with wrong matrix shape."""
        config = DggtConfig(
            scene_base_path="/valid/path",
            t_scenario_dggt=np.zeros((3, 3), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="t_scenario_dggt must be 4x4"):
            config.validate()


class TestDggtConfigLoaderYAML:
    """Tests for DggtConfigLoader.from_yaml()."""

    def test_load_from_yaml_success(self):
        """Test successful YAML loading."""
        yaml_content = """
server:
  host: myserver.local
  port: 9090
  timeout: 60.0
scene:
  base_path: /test/scenes
  default_id: scene_001
image:
  format: PNG
  quality: 80
coordinate:
  enabled: false
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml(f.name)
            os.unlink(f.name)

        assert config.server_host == "myserver.local"
        assert config.server_port == 9090
        assert config.request_timeout == 60.0
        assert config.scene_base_path == "/test/scenes"
        assert config.default_scene_id == "scene_001"
        assert config.image_format == "PNG"
        assert config.image_quality == 80.0
        assert config.coordinate_transform_enabled is False

    def test_load_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            DggtConfigLoader.from_yaml("/nonexistent/config.yaml")

    def test_load_from_yaml_partial_config(self):
        """Test loading YAML with only partial config."""
        yaml_content = """
server:
  host: partial.local
scene:
  base_path: /partial/scenes
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml(f.name)
            os.unlink(f.name)

        # Override values from YAML
        assert config.server_host == "partial.local"
        assert config.scene_base_path == "/partial/scenes"
        # Default values for unspecified fields
        assert config.server_port == 50051  # default
        assert config.image_format == "JPEG"  # default
        assert config.image_quality == 95.0  # default

    def test_load_from_yaml_with_transform_matrix(self):
        """Test loading YAML with transform matrix."""
        yaml_content = """
scene:
  base_path: /test/scenes
transform:
  t_scenario_dggt:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml(f.name)
            os.unlink(f.name)

        assert config.t_scenario_dggt is not None
        assert np.allclose(config.t_scenario_dggt, np.eye(4))

    def test_load_from_yaml_with_grpc_section(self):
        """Test loading YAML with gRPC section."""
        yaml_content = """
scene:
  base_path: /test/scenes
grpc:
  max_message_length: 20971520
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml(f.name)
            os.unlink(f.name)

        assert config.max_message_length == 20971520  # 20MB

    def test_load_from_yaml_with_dynamic_objects_section(self):
        """Test loading YAML with dynamic objects section."""
        yaml_content = """
scene:
  base_path: /test/scenes
dynamic_objects:
  max_count: 50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml(f.name)
            os.unlink(f.name)

        assert config.max_dynamic_objects == 50


class TestDggtConfigLoaderEnv:
    """Tests for DggtConfigLoader.from_env()."""

    def test_from_env_override_server_host(self):
        """Test environment variable override for server host."""
        os.environ['DGGT_SERVER_HOST'] = 'env-server.local'
        config = DggtConfigLoader.from_env()
        assert config.server_host == 'env-server.local'
        del os.environ['DGGT_SERVER_HOST']

    def test_from_env_override_server_port(self):
        """Test environment variable override for server port."""
        os.environ['DGGT_SERVER_PORT'] = '9999'
        config = DggtConfigLoader.from_env()
        assert config.server_port == 9999
        del os.environ['DGGT_SERVER_PORT']

    def test_from_env_override_scene_base_path(self):
        """Test environment variable override for scene base path."""
        os.environ['DGGT_SCENE_BASE_PATH'] = '/env/scenes'
        config = DggtConfigLoader.from_env()
        assert config.scene_base_path == '/env/scenes'
        del os.environ['DGGT_SCENE_BASE_PATH']

    def test_from_env_override_scene_id(self):
        """Test environment variable override for scene ID."""
        os.environ['DGGT_SCENE_ID'] = 'env_scene_001'
        config = DggtConfigLoader.from_env()
        assert config.default_scene_id == 'env_scene_001'
        del os.environ['DGGT_SCENE_ID']

    def test_from_env_override_base_config(self):
        """Test that env vars override base config values."""
        base_config = DggtConfig(
            scene_base_path="/base/path",
            server_host="base.local",
            server_port=5000,
        )
        os.environ['DGGT_SERVER_HOST'] = 'override.local'
        os.environ['DGGT_SERVER_PORT'] = '6000'
        config = DggtConfigLoader.from_env(base_config)
        assert config.server_host == 'override.local'
        assert config.server_port == 6000
        assert config.scene_base_path == '/base/path'  # not overridden
        del os.environ['DGGT_SERVER_HOST']
        del os.environ['DGGT_SERVER_PORT']

    def test_from_env_no_overrides(self):
        """Test from_env without any env vars set."""
        # Clear any existing env vars
        for key in ['DGGT_SERVER_HOST', 'DGGT_SERVER_PORT', 'DGGT_SCENE_BASE_PATH', 'DGGT_SCENE_ID']:
            if key in os.environ:
                del os.environ[key]

        base_config = DggtConfig(scene_base_path="/test/path")
        config = DggtConfigLoader.from_env(base_config)
        assert config.scene_base_path == "/test/path"

    def test_from_env_additional_overrides(self):
        """Test additional optional environment variable overrides."""
        os.environ['DGGT_REQUEST_TIMEOUT'] = '45.0'
        os.environ['DGGT_IMAGE_FORMAT'] = 'PNG'
        os.environ['DGGT_IMAGE_QUALITY'] = '85.0'
        os.environ['DGGT_MAX_DYNAMIC_OBJECTS'] = '75'

        config = DggtConfigLoader.from_env(DggtConfig(scene_base_path="/test"))
        assert config.request_timeout == 45.0
        assert config.image_format == "PNG"
        assert config.image_quality == 85.0
        assert config.max_dynamic_objects == 75

        for key in ['DGGT_REQUEST_TIMEOUT', 'DGGT_IMAGE_FORMAT', 'DGGT_IMAGE_QUALITY', 'DGGT_MAX_DYNAMIC_OBJECTS']:
            del os.environ[key]


class TestDggtConfigLoaderDefault:
    """Tests for DggtConfigLoader.default()."""

    def test_default_returns_default_config(self):
        """Test that default() returns a config with all default values."""
        config = DggtConfigLoader.default()
        assert config.server_host == "localhost"
        assert config.server_port == 50051
        assert config.scene_base_path == ""
        assert config.image_format == "JPEG"
        assert config.image_quality == 95.0


class TestDggtConfigLoaderCombined:
    """Tests for DggtConfigLoader.from_yaml_with_env()."""

    def test_yaml_with_env_override(self):
        """Test combined YAML + env loading with override."""
        yaml_content = """
scene:
  base_path: /yaml/scenes
server:
  host: yaml.local
  port: 5000
"""
        os.environ['DGGT_SERVER_HOST'] = 'env.local'
        os.environ['DGGT_SERVER_PORT'] = '7000'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml_with_env(f.name)
            os.unlink(f.name)

        # Env overrides YAML
        assert config.server_host == 'env.local'
        assert config.server_port == 7000
        # YAML value not overridden
        assert config.scene_base_path == '/yaml/scenes'

        del os.environ['DGGT_SERVER_HOST']
        del os.environ['DGGT_SERVER_PORT']

    def test_yaml_with_env_no_override(self):
        """Test combined loading without env overrides."""
        yaml_content = """
scene:
  base_path: /yaml/scenes
server:
  host: yaml.local
"""
        # Clear env vars
        for key in ['DGGT_SERVER_HOST', 'DGGT_SERVER_PORT']:
            if key in os.environ:
                del os.environ[key]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml_with_env(f.name)
            os.unlink(f.name)

        assert config.server_host == 'yaml.local'
        assert config.scene_base_path == '/yaml/scenes'


class TestIntegrationWithRealYAML:
    """Integration tests using real config file."""

    @pytest.fixture
    def real_config_path(self):
        """Path to the real dggt_config.yaml file."""
        return Path(__file__).parent.parent.parent / "configs" / "dggt_config.yaml"

    def test_load_real_config_file(self, real_config_path):
        """Test loading the actual config file."""
        if not real_config_path.exists():
            pytest.skip("Real config file not found")

        config = DggtConfigLoader.from_yaml(str(real_config_path))
        assert config.server_host == "localhost"
        assert config.server_port == 50051
        assert config.request_timeout == 30.0
        assert config.scene_base_path != ""  # Should have a real path
        assert config.image_format == "JPEG"
        assert config.image_quality == 95.0
        assert config.coordinate_transform_enabled is True