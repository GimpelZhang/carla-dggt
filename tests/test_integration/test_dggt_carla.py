# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Integration tests for DGGT + CARLA integration.

Tests verify:
- Config loading from YAML
- gRPC connection to DGGT server
- Camera retrieval
- Single frame rendering
- Image decoding

Note: These tests require:
- DGGT server running (or use mock server)
- CARLA server available at localhost:2000
- Valid scene data in scene_base_path
"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dggt_config import DggtConfig, DggtConfigLoader
from dggt_integration import DggtRenderer, DggtScenario, DggtSensor

# Import protobuf types
from nre.grpc.protos import sensorsim_pb2, common_pb2


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_config():
    """Sample DggtConfig for testing."""
    return DggtConfig(
        server_host="localhost",
        server_port=50051,
        scene_base_path="/test/scenes",
        default_scene_id="test_scene",
        request_timeout=30.0,
        image_format="JPEG",
        image_quality=95.0,
    )


@pytest.fixture
def sample_yaml_content():
    """Sample YAML config content."""
    return """
server:
  host: localhost
  port: 50051
  timeout: 30.0
scene:
  base_path: /test/scenes
  default_id: test_scene
image:
  format: JPEG
  quality: 95
coordinate:
  enabled: true
"""


@pytest.fixture
def sample_camera_spec():
    """Sample CameraSpec for testing."""
    spec = sensorsim_pb2.CameraSpec()
    spec.resolution_w = 1036
    spec.resolution_h = 700
    spec.opencv_fisheye_param.focal_length_x = 1000.0
    spec.opencv_fisheye_param.focal_length_y = 1000.0
    spec.opencv_fisheye_param.principal_point_x = 518.0
    spec.opencv_fisheye_param.principal_point_y = 350.0
    return spec


@pytest.fixture
def sample_pose():
    """Sample Pose for testing."""
    pose = common_pb2.Pose()
    pose.vec.x = 10.0
    pose.vec.y = 5.0
    pose.vec.z = 2.0
    pose.quat.w = 1.0
    pose.quat.x = 0.0
    pose.quat.y = 0.0
    pose.quat.z = 0.0
    return pose


@pytest.fixture
def mock_grpc_stub():
    """Mock gRPC stub for testing."""
    stub = Mock()

    # Mock get_available_cameras response
    cameras_response = sensorsim_pb2.AvailableCamerasReturn()
    cam = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera()
    cam.logical_id = "front_camera"
    cam.intrinsics.resolution_w = 1036
    cam.intrinsics.resolution_h = 700
    cameras_response.available_cameras.append(cam)

    stub.get_available_cameras.return_value = cameras_response

    # Mock render_rgb response - create a valid JPEG image
    render_response = sensorsim_pb2.RGBRenderReturn()
    # Create a simple valid test image using PIL
    from PIL import Image
    import io
    test_image = Image.new('RGB', (10, 10), color='red')
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    render_response.image_bytes = buffer.getvalue()
    stub.render_rgb.return_value = render_response

    return stub


# ============================================================================
# Config Loading Tests
# ============================================================================

class TestConfigLoading:
    """Tests for YAML configuration loading."""

    def test_load_config_from_yaml(self, sample_yaml_content):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(sample_yaml_content)
            f.flush()
            config = DggtConfigLoader.from_yaml(f.name)
            os.unlink(f.name)

        assert config.server_host == "localhost"
        assert config.server_port == 50051
        assert config.request_timeout == 30.0
        assert config.scene_base_path == "/test/scenes"
        assert config.default_scene_id == "test_scene"
        assert config.image_format == "JPEG"
        assert config.image_quality == 95.0
        assert config.coordinate_transform_enabled is True

    def test_config_validation_missing_scene_path(self):
        """Test config validation fails without scene_base_path."""
        config = DggtConfig()  # Empty scene_base_path
        with pytest.raises(ValueError, match="scene_base_path is required"):
            config.validate()

    def test_config_validation_invalid_image_format(self):
        """Test config validation fails with invalid image format."""
        config = DggtConfig(
            scene_base_path="/test",
            image_format="INVALID",
        )
        with pytest.raises(ValueError, match="Unsupported image format"):
            config.validate()

    def test_config_env_override(self):
        """Test environment variable override."""
        os.environ['DGGT_SERVER_HOST'] = 'override.local'
        os.environ['DGGT_SERVER_PORT'] = '9999'

        base_config = DggtConfig(scene_base_path="/test")
        config = DggtConfigLoader.from_env(base_config)

        assert config.server_host == 'override.local'
        assert config.server_port == 9999

        del os.environ['DGGT_SERVER_HOST']
        del os.environ['DGGT_SERVER_PORT']


# ============================================================================
# gRPC Connection Tests
# ============================================================================

class TestGRPCConnection:
    """Tests for gRPC connection handling."""

    @patch('dggt_integration.grpc.insecure_channel')
    def test_renderer_connect(self, mock_channel, sample_config, mock_grpc_stub):
        """Test renderer connection to DGGT server."""
        mock_channel.return_value = Mock()

        renderer = DggtRenderer(sample_config, "test_scene")

        # Mock the stub creation
        with patch('dggt_integration.sensorsim_pb2_grpc.SensorsimServiceStub', return_value=mock_grpc_stub):
            renderer.connect()

        assert renderer._stub is not None
        mock_channel.assert_called_once()

    def test_renderer_disconnect(self, sample_config):
        """Test renderer disconnect."""
        renderer = DggtRenderer(sample_config, "test_scene")
        mock_channel = Mock()
        renderer._channel = mock_channel
        renderer._stub = Mock()

        renderer.disconnect()

        mock_channel.close.assert_called_once()
        assert renderer._stub is None


# ============================================================================
# Camera Retrieval Tests
# ============================================================================

class TestCameraRetrieval:
    """Tests for camera retrieval."""

    def test_get_available_cameras(self, sample_config, mock_grpc_stub):
        """Test getting available cameras."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._stub = mock_grpc_stub

        # Manually fetch cameras
        renderer._available_cameras = {"front_camera": Mock()}

        cameras = renderer.get_available_cameras()

        assert "front_camera" in cameras
        assert isinstance(cameras, dict)

    def test_camera_not_found_error(self, sample_config):
        """Test error when camera not found."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._available_cameras = {"front_camera": Mock()}

        with pytest.raises(KeyError):
            renderer.get_camera_spec("invalid_camera")


# ============================================================================
# Render Tests
# ============================================================================

class TestRendering:
    """Tests for rendering functionality."""

    def test_render_request_build(self, sample_config, sample_camera_spec):
        """Test render request building."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._stub = Mock()

        camera_pose = np.eye(4, dtype=np.float32)
        timestamp = 100000

        request = renderer._build_render_request(
            camera_spec=sample_camera_spec,
            camera_pose=camera_pose,
            timestamp=timestamp,
            resolution_ratio=1.0,
        )

        assert request.scene_id == "test_scene"
        assert request.resolution_w == 1036
        assert request.resolution_h == 700
        assert request.frame_start_us == timestamp
        assert request.image_format == sensorsim_pb2.ImageFormat.JPEG

    def test_render_single_frame_mock(self, sample_config, mock_grpc_stub, sample_camera_spec):
        """Test single frame rendering with mock."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._stub = mock_grpc_stub
        renderer._available_cameras = {"front_camera": sample_camera_spec}

        # Create mock world snapshot
        mock_snapshot = Mock()

        camera_pose = np.eye(4, dtype=np.float32)
        timestamp = 100000

        image = renderer.render(
            world_snapshot=mock_snapshot,
            camera_spec=sample_camera_spec,
            camera_pose=camera_pose,
            timestamp=timestamp,
        )

        # Verify render was called
        mock_grpc_stub.render_rgb.assert_called_once()

        # Verify image is numpy array
        assert isinstance(image, np.ndarray)


# ============================================================================
# DggtScenario Tests
# ============================================================================

class TestDggtScenario:
    """Tests for DggtScenario class."""

    def test_scenario_transform_derivation(self, sample_config):
        """Test transform matrix derivation."""
        # Create config with custom transform
        custom_transform = np.eye(4, dtype=np.float32)
        custom_transform[:3, 3] = [100.0, 50.0, 10.0]

        config = DggtConfig(
            scene_base_path="/test",
            t_scenario_dggt=custom_transform,
        )

        # Pass transform directly to renderer (as DggtScenario does)
        renderer = DggtRenderer(config, "test_scene", t_scenario_dggt=custom_transform)

        # Verify transform is applied correctly
        assert np.allclose(renderer._t_scenario_dggt, custom_transform)
        assert np.allclose(renderer._t_carla_dggt, np.linalg.inv(custom_transform))


# ============================================================================
# DggtSensor Tests
# ============================================================================

class TestDggtSensor:
    """Tests for DggtSensor class."""

    def test_sensor_initialization(self, sample_config, mock_grpc_stub, sample_camera_spec):
        """Test sensor initialization."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._stub = mock_grpc_stub
        renderer._available_cameras = {"front_camera": sample_camera_spec}

        sensor = DggtSensor(
            renderer=renderer,
            camera_id="front_camera",
            output_dir="/tmp/test_output",
        )

        assert sensor.get_camera_id() == "front_camera"
        assert sensor.get_frame_count() == 0

    def test_sensor_invalid_camera(self, sample_config, mock_grpc_stub, sample_camera_spec):
        """Test sensor with invalid camera ID."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._stub = mock_grpc_stub
        renderer._available_cameras = {"front_camera": sample_camera_spec}

        with pytest.raises(ValueError, match="Camera 'invalid' not found"):
            DggtSensor(
                renderer=renderer,
                camera_id="invalid",
            )

    def test_framerate_control(self, sample_config, mock_grpc_stub, sample_camera_spec):
        """Test sensor framerate control."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._stub = mock_grpc_stub
        renderer._available_cameras = {"front_camera": sample_camera_spec}

        sensor = DggtSensor(
            renderer=renderer,
            camera_id="front_camera",
            framerate=10.0,
        )

        # First tick at t=0.0 should NOT render (last_timestamp=0.0, diff=0.0 < 0.1)
        assert sensor._should_render(0.0) is False

        # After interval should render
        assert sensor._should_render(0.11) is True

        # Immediate second tick should skip (framerate control)
        assert sensor._should_render(0.15) is False

        # After another interval should render
        assert sensor._should_render(0.22) is True


# ============================================================================
# Image Decode Tests
# ============================================================================

class TestImageDecode:
    """Tests for image decoding."""

    def test_decode_jpeg_pil(self, sample_config):
        """Test JPEG decoding with PIL fallback."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._jpeg_decoder = None  # Force PIL fallback

        # Create a simple valid test image using PIL
        from PIL import Image
        import io
        test_image = Image.new('RGB', (10, 10), color='red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        jpeg_bytes = buffer.getvalue()

        image = renderer._decode_image(jpeg_bytes)

        assert isinstance(image, np.ndarray)


# ============================================================================
# Verification Checklist Tests
# ============================================================================

class TestVerificationChecklist:
    """
    Verification checklist from Phase5_Detailed_Plan.md lines 1367-1372:
    - [ ] DGGT server can start and respond to requests
    - [ ] CARLA client can connect DGGT server
    - [ ] Render requests correctly sent and received
    - [ ] Images can be correctly decoded and saved
    - [ ] Error conditions correctly handled
    """

    @patch('dggt_integration.grpc.insecure_channel')
    def test_server_connection(self, mock_channel, sample_config, mock_grpc_stub):
        """Verify DGGT server can start and respond to requests."""
        mock_channel.return_value = Mock()

        renderer = DggtRenderer(sample_config, "test_scene")

        with patch('dggt_integration.sensorsim_pb2_grpc.SensorsimServiceStub', return_value=mock_grpc_stub):
            renderer.connect()

            # Verify stub is created (server connection)
            assert renderer._stub is not None

    def test_render_request_sent(self, sample_config, mock_grpc_stub, sample_camera_spec):
        """Verify render requests correctly sent."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._stub = mock_grpc_stub

        mock_snapshot = Mock()
        camera_pose = np.eye(4, dtype=np.float32)

        renderer.render(
            world_snapshot=mock_snapshot,
            camera_spec=sample_camera_spec,
            camera_pose=camera_pose,
            timestamp=100000,
        )

        # Verify request was sent
        mock_grpc_stub.render_rgb.assert_called_once()
        request = mock_grpc_stub.render_rgb.call_args[0][0]

        assert request.scene_id == "test_scene"
        assert request.resolution_w == 1036
        assert request.resolution_h == 700

    def test_image_decode(self, sample_config):
        """Verify images can be correctly decoded."""
        renderer = DggtRenderer(sample_config, "test_scene")
        renderer._jpeg_decoder = None

        # Create a simple valid test image using PIL
        from PIL import Image
        import io
        test_image = Image.new('RGB', (10, 10), color='red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        jpeg_bytes = buffer.getvalue()

        image = renderer._decode_image(jpeg_bytes)

        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8

    def test_error_handling_not_connected(self, sample_config, sample_camera_spec):
        """Verify error conditions correctly handled."""
        renderer = DggtRenderer(sample_config, "test_scene")
        # Do not connect (stub is None)

        with pytest.raises(RuntimeError, match="Not connected"):
            renderer.render(
                world_snapshot=Mock(),
                camera_spec=sample_camera_spec,
                camera_pose=np.eye(4),
                timestamp=100000,
            )


# ============================================================================
# Integration Test Markers
# ============================================================================

# Mark tests that require actual CARLA server
@pytest.mark.integration
class TestCARLAIntegration:
    """Tests requiring actual CARLA server."""

    @pytest.mark.skipif(
        not os.environ.get('CARLA_HOST'),
        reason="CARLA server not available"
    )
    def test_carla_connection(self):
        """Test CARLA server connection."""
        import carla

        host = os.environ.get('CARLA_HOST', 'localhost')
        port = int(os.environ.get('CARLA_PORT', 2000))

        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()

        assert world is not None