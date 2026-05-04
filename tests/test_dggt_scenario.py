# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for DggtScenario and DggtSensor classes

Tests cover:
- DggtScenario initialization, transform computation, scene loading
- DggtSensor initialization, tick callback, transform chain
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import os

# Import test targets
from dggt_integration import DggtScenario, DggtSensor, DggtRenderer
from dggt_config import DggtConfig
from dggt_server.scene_metadata import DGGTSceneMetadata


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock DggtConfig"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DggtConfig(
            server_host="localhost",
            server_port=50051,
            scene_base_path=tmpdir,
            default_scene_id="test_scene",
        )
        yield config


@pytest.fixture
def mock_scene_metadata():
    """Create mock DGGTSceneMetadata"""
    return DGGTSceneMetadata(
        scene_id="test_scene",
        scene_path="/tmp/test_scene",
        num_frames=100,
        fps=10.0,
        start_timestamp_us=0,
        end_timestamp_us=10_000_000,
        camera_width=1920,
        camera_height=1080,
        intrinsic_matrix=np.eye(3),
        intrinsics_vary=False,
        has_static_scene=True,
        has_sky_scene=True,
        dynamic_object_ids=[],
        static_scene_size_mb=100.0,
        total_size_mb=150.0,
        georeference=None,
        t_scenario_dggt=None,
    )


@pytest.fixture
def mock_renderer():
    """Create mock DggtRenderer"""
    renderer = Mock(spec=DggtRenderer)
    renderer.get_available_cameras.return_value = {
        "front_wide": Mock(logical_id="front_wide"),
        "rear": Mock(logical_id="rear"),
    }
    renderer.render.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return renderer


# ============================================================================
# DggtScenario Tests
# ============================================================================

class TestDggtScenario:
    """Tests for DggtScenario class"""

    def test_initialization_success(self, mock_config, mock_scene_metadata):
        """Test successful scenario initialization"""
        # Create scene directory
        scene_path = Path(mock_config.scene_base_path) / mock_config.default_scene_id
        scene_path.mkdir(parents=True, exist_ok=True)

        with patch('dggt_server.scene_manager.DGGTSceneManager') as MockSceneManager:
            mock_manager = Mock()
            mock_manager.get_scene.return_value = mock_scene_metadata
            MockSceneManager.return_value = mock_manager

            scenario = DggtScenario(mock_config)

            assert scenario.get_scene_id() == "test_scene"
            assert scenario.get_timestamp_range() == (0, 10_000_000)
            assert scenario.get_scene_metadata() == mock_scene_metadata

    def test_initialization_scene_not_found(self, mock_config):
        """Test initialization with non-existent scene path"""
        config = DggtConfig(
            server_host="localhost",
            server_port=50051,
            scene_base_path="/nonexistent",
            default_scene_id="missing_scene",
        )

        with pytest.raises(ValueError, match="Scene path not found"):
            DggtScenario(config)

    def test_transform_from_config(self, mock_config, mock_scene_metadata):
        """Test transform from config override"""
        scene_path = Path(mock_config.scene_base_path) / mock_config.default_scene_id
        scene_path.mkdir(parents=True, exist_ok=True)

        # Set config override transform
        custom_transform = np.array([
            [1, 0, 0, 10],
            [0, 1, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        mock_config.t_scenario_dggt = custom_transform

        with patch('dggt_server.scene_manager.DGGTSceneManager') as MockSceneManager:
            mock_manager = Mock()
            mock_manager.get_scene.return_value = mock_scene_metadata
            MockSceneManager.return_value = mock_manager

            scenario = DggtScenario(mock_config)

            # t_scenario_dggt should come from config
            np.testing.assert_array_equal(scenario._t_scenario_dggt, custom_transform)
            # t_carla_dggt should be inverse
            np.testing.assert_array_equal(
                scenario.get_t_carla_dggt(),
                np.linalg.inv(custom_transform)
            )

    def test_transform_from_metadata(self, mock_config, mock_scene_metadata):
        """Test transform from scene metadata"""
        scene_path = Path(mock_config.scene_base_path) / mock_config.default_scene_id
        scene_path.mkdir(parents=True, exist_ok=True)

        # Set metadata transform
        custom_transform = np.eye(4)
        custom_transform[0, 3] = 5.0
        mock_scene_metadata.t_scenario_dggt = custom_transform

        with patch('dggt_server.scene_manager.DGGTSceneManager') as MockSceneManager:
            mock_manager = Mock()
            mock_manager.get_scene.return_value = mock_scene_metadata
            MockSceneManager.return_value = mock_manager

            scenario = DggtScenario(mock_config)

            np.testing.assert_array_equal(scenario._t_scenario_dggt, custom_transform)

    def test_transform_default_identity(self, mock_config, mock_scene_metadata):
        """Test default identity transform when no source available"""
        scene_path = Path(mock_config.scene_base_path) / mock_config.default_scene_id
        scene_path.mkdir(parents=True, exist_ok=True)

        with patch('dggt_server.scene_manager.DGGTSceneManager') as MockSceneManager:
            mock_manager = Mock()
            mock_manager.get_scene.return_value = mock_scene_metadata
            MockSceneManager.return_value = mock_manager

            scenario = DggtScenario(mock_config)

            np.testing.assert_array_equal(scenario._t_scenario_dggt, np.eye(4))
            np.testing.assert_array_equal(scenario.get_t_carla_dggt(), np.eye(4))

    def test_load_scene(self, mock_config, mock_scene_metadata):
        """Test scene loading with renderer initialization"""
        scene_path = Path(mock_config.scene_base_path) / mock_config.default_scene_id
        scene_path.mkdir(parents=True, exist_ok=True)

        with patch('dggt_server.scene_manager.DGGTSceneManager') as MockSceneManager, \
             patch('dggt_integration.DggtRenderer') as MockRenderer:
            mock_manager = Mock()
            mock_manager.get_scene.return_value = mock_scene_metadata
            MockSceneManager.return_value = mock_manager

            mock_renderer = Mock()
            MockRenderer.return_value = mock_renderer

            scenario = DggtScenario(mock_config)
            scenario.load_scene()

            # Renderer should be created with correct params
            MockRenderer.assert_called_once()
            mock_renderer.connect.assert_called_once()

    def test_get_renderer_before_load(self, mock_config, mock_scene_metadata):
        """Test get_renderer raises error before load_scene"""
        scene_path = Path(mock_config.scene_base_path) / mock_config.default_scene_id
        scene_path.mkdir(parents=True, exist_ok=True)

        with patch('dggt_server.scene_manager.DGGTSceneManager') as MockSceneManager:
            mock_manager = Mock()
            mock_manager.get_scene.return_value = mock_scene_metadata
            MockSceneManager.return_value = mock_manager

            scenario = DggtScenario(mock_config)

            with pytest.raises(RuntimeError, match="Scene not loaded"):
                scenario.get_renderer()


# ============================================================================
# DggtSensor Tests
# ============================================================================

class TestDggtSensor:
    """Tests for DggtSensor class"""

    def test_initialization_success(self, mock_renderer):
        """Test successful sensor initialization"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
            resolution_ratio=0.5,
        )

        assert sensor.get_camera_id() == "front_wide"
        assert sensor.get_frame_count() == 0

    def test_initialization_invalid_camera(self, mock_renderer):
        """Test initialization with invalid camera ID"""
        with pytest.raises(ValueError, match="Camera 'invalid' not found"):
            DggtSensor(
                renderer=mock_renderer,
                camera_id="invalid",
            )

    def test_register_tick_callback(self, mock_renderer):
        """Test tick callback registration"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
        )

        mock_world = Mock()
        mock_world.on_tick.return_value = 42

        sensor.register_tick_callback(mock_world)

        mock_world.on_tick.assert_called_once_with(sensor.on_tick)
        assert sensor._callback_id == 42

    def test_unregister_tick_callback(self, mock_renderer):
        """Test tick callback unregistration"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
        )

        mock_world = Mock()
        mock_world.on_tick.return_value = 42

        sensor.register_tick_callback(mock_world)
        sensor.unregister_tick_callback(mock_world)

        mock_world.remove_on_tick.assert_called_once_with(42)
        assert sensor._callback_id is None

    def test_on_tick_mounted_camera(self, mock_renderer):
        """Test on_tick with mounted camera (parent actor)"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
            resolution_ratio=1.0,
            parent_actor=Mock(id=1),
            sensor_offset=np.eye(4),
        )

        # Mock world snapshot
        mock_snapshot = Mock()
        mock_snapshot.timestamp.elapsed_seconds = 1.0

        # Mock actor
        mock_actor = Mock()
        mock_actor.get_transform.return_value.get_matrix.return_value = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        mock_snapshot.find.return_value = mock_actor

        sensor.on_tick(mock_snapshot)

        # Should have rendered one frame
        mock_renderer.render.assert_called_once()
        assert sensor.get_frame_count() == 1

    def test_on_tick_free_camera(self, mock_renderer):
        """Test on_tick with free camera (preset pose)"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
            preset_pose=np.eye(4),
        )

        mock_snapshot = Mock()
        mock_snapshot.timestamp.elapsed_seconds = 1.0

        sensor.on_tick(mock_snapshot)

        mock_renderer.render.assert_called_once()
        assert sensor.get_frame_count() == 1

    def test_on_tick_framerate_control(self, mock_renderer):
        """Test framerate control in on_tick"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
            preset_pose=np.eye(4),
            framerate=10.0,  # 10 fps
        )

        mock_snapshot = Mock()

        # First tick at t=0.1 (after initial frame interval)
        mock_snapshot.timestamp.elapsed_seconds = 0.1
        sensor.on_tick(mock_snapshot)
        assert sensor.get_frame_count() == 1

        # Second tick at t=0.15 (too early, should skip)
        mock_snapshot.timestamp.elapsed_seconds = 0.15
        sensor.on_tick(mock_snapshot)
        assert sensor.get_frame_count() == 1  # No new frame

        # Third tick at t=0.2 (should render)
        mock_snapshot.timestamp.elapsed_seconds = 0.2
        sensor.on_tick(mock_snapshot)
        assert sensor.get_frame_count() == 2  # New frame

    def test_on_tick_parent_actor_not_found(self, mock_renderer):
        """Test on_tick when parent actor not found in snapshot"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
            parent_actor=Mock(id=999),
        )

        mock_snapshot = Mock()
        mock_snapshot.timestamp.elapsed_seconds = 1.0
        mock_snapshot.find.return_value = None  # Actor not found

        sensor.on_tick(mock_snapshot)

        # Should not render
        mock_renderer.render.assert_not_called()
        assert sensor.get_frame_count() == 0

    def test_on_tick_no_preset_pose(self, mock_renderer):
        """Test on_tick with free camera but no preset pose"""
        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
        )

        mock_snapshot = Mock()
        mock_snapshot.timestamp.elapsed_seconds = 1.0

        sensor.on_tick(mock_snapshot)

        # Should not render
        mock_renderer.render.assert_not_called()

    def test_save_image(self, mock_renderer):
        """Test image saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sensor = DggtSensor(
                renderer=mock_renderer,
                camera_id="front_wide",
                preset_pose=np.eye(4),
                output_dir=tmpdir,
            )

            mock_snapshot = Mock()
            mock_snapshot.timestamp.elapsed_seconds = 1.0

            sensor.on_tick(mock_snapshot)

            # Check image was saved (frame count 0 -> first save uses 0)
            expected_path = os.path.join(tmpdir, "frame_000000.jpg")
            assert os.path.exists(expected_path)

    def test_callback_invoked(self, mock_renderer):
        """Test custom callback is invoked"""
        callback_images = []

        def my_callback(image):
            callback_images.append(image)

        sensor = DggtSensor(
            renderer=mock_renderer,
            camera_id="front_wide",
            preset_pose=np.eye(4),
            callback=my_callback,
        )

        mock_snapshot = Mock()
        mock_snapshot.timestamp.elapsed_seconds = 1.0

        sensor.on_tick(mock_snapshot)

        assert len(callback_images) == 1
        assert callback_images[0].shape == (1080, 1920, 3)


# ============================================================================
# Transform Chain Tests (CRITICAL)
# ============================================================================

class TestTransformChain:
    """
    Critical tests for transform chain correctness

    The transform chain must match NuRec exactly:
    1. DggtSensor.on_tick: camera_pose = undo_carla_coordinate_transform(actor_pose) @ sensor_offset
    2. DggtRenderer.render: dggt_pose = t_carla_dggt @ camera_pose
    """

    def test_undo_is_applied_in_on_tick(self, mock_renderer):
        """Test that undo_carla_coordinate_transform is applied in on_tick"""
        with patch('dggt_integration.undo_carla_coordinate_transform') as mock_undo:
            # Setup: undo returns identity (simplified test)
            mock_undo.return_value = np.eye(4)

            sensor = DggtSensor(
                renderer=mock_renderer,
                camera_id="front_wide",
                preset_pose=np.eye(4),
            )

            mock_snapshot = Mock()
            mock_snapshot.timestamp.elapsed_seconds = 1.0

            sensor.on_tick(mock_snapshot)

            # Verify undo was called
            mock_undo.assert_called_once()

    def test_transform_chain_order(self, mock_renderer):
        """
        Test that transform chain follows NuRec pattern

        Expected order:
        1. Get actor transform
        2. Apply undo_carla_coordinate_transform
        3. Apply sensor_offset
        4. Pass to renderer (which applies t_carla_dggt)
        """
        # Create test transforms
        actor_transform = np.eye(4)
        actor_transform[0, 3] = 10.0  # Translation in X

        sensor_offset = np.eye(4)
        sensor_offset[1, 3] = 2.0  # Offset in Y

        expected_undo_result = np.array([
            [1, 0, 0, 10],
            [0, -1, 0, -0],  # Y flipped
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        with patch('dggt_integration.undo_carla_coordinate_transform') as mock_undo:
            mock_undo.return_value = expected_undo_result

            sensor = DggtSensor(
                renderer=mock_renderer,
                camera_id="front_wide",
                parent_actor=Mock(id=1),
                sensor_offset=sensor_offset,
            )

            # Mock actor
            mock_actor = Mock()
            mock_actor.get_transform.return_value.get_matrix.return_value = \
                actor_transform.flatten().tolist()

            mock_snapshot = Mock()
            mock_snapshot.timestamp.elapsed_seconds = 1.0
            mock_snapshot.find.return_value = mock_actor

            sensor.on_tick(mock_snapshot)

            # Get the camera_pose passed to render
            call_args = mock_renderer.render.call_args
            camera_pose = call_args.kwargs['camera_pose']

            # camera_pose should be: undo(actor_transform) @ sensor_offset
            expected_pose = expected_undo_result @ sensor_offset
            np.testing.assert_array_almost_equal(camera_pose, expected_pose)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for DggtScenario + DggtSensor"""

    def test_scenario_sensor_integration(self, mock_config, mock_scene_metadata):
        """Test scenario and sensor work together"""
        scene_path = Path(mock_config.scene_base_path) / mock_config.default_scene_id
        scene_path.mkdir(parents=True, exist_ok=True)

        with patch('dggt_server.scene_manager.DGGTSceneManager') as MockSceneManager:
            mock_manager = Mock()
            mock_manager.get_scene.return_value = mock_scene_metadata
            MockSceneManager.return_value = mock_manager

            # Create scenario
            scenario = DggtScenario(mock_config)

            # Mock renderer
            mock_renderer = Mock()
            mock_renderer.get_available_cameras.return_value = {
                "front": Mock(),
            }

            # Create sensor
            sensor = DggtSensor(
                renderer=mock_renderer,
                camera_id="front",
                preset_pose=np.eye(4),
            )

            # Verify sensor works
            mock_snapshot = Mock()
            mock_snapshot.timestamp.elapsed_seconds = 1.0
            sensor.on_tick(mock_snapshot)

            assert sensor.get_frame_count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])