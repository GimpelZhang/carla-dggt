# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for DGGT service

Mock-based tests for DGGT gRPC service functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import grpc

from dggt_server.scene_metadata import DGGTSceneMetadata, FrameMetadata, ObjectMetadata
from dggt_server.scene_manager import DGGTSceneManager
from dggt_server.frame_index_mapper import FrameIndexMapper
from dggt_server.config import DGGTServerConfig


class MockContext:
    """Mock gRPC context for testing error handling"""

    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class TestDGGTServiceMock:
    """Mock-based tests for DGGT service"""

    @pytest.fixture
    def mock_scene_metadata(self):
        """Create mock scene metadata"""
        return DGGTSceneMetadata(
            scene_id="test_scene",
            scene_path="/path/to/scene",
            num_frames=100,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=9_900_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[0, 1],
            static_scene_size_mb=100.0,
            total_size_mb=120.0
        )

    @pytest.fixture
    def mock_frame_metadata(self):
        """Create mock frame metadata"""
        return FrameMetadata(
            frame_idx=50,
            timestamp_us=5_000_000,
            c2w_matrix=np.eye(4),
            intrinsic_matrix=np.array([[1000, 0, 259], [0, 1000, 175], [0, 0, 1]]),
            width=518,
            height=350,
            objects=[]
        )

    @pytest.fixture
    def mock_scene_manager(self, mock_scene_metadata):
        """Create mock scene manager"""
        manager = Mock(spec=DGGTSceneManager)
        manager.get_scene.return_value = mock_scene_metadata
        manager.get_frame_metadata.return_value = FrameMetadata(
            frame_idx=0,
            timestamp_us=0,
            c2w_matrix=np.eye(4),
            intrinsic_matrix=np.eye(3),
            width=518,
            height=350,
            objects=[]
        )
        manager.list_scenes.return_value = ["test_scene"]
        return manager

    @pytest.fixture
    def mock_config(self):
        """Create mock server config"""
        return DGGTServerConfig(
            grpc_host="localhost",
            grpc_port=50052,
            max_workers=10,
            device="cuda",
            scene_base_path="/test/path"
        )

    def test_scene_manager_get_scene(self, mock_scene_manager, mock_scene_metadata):
        """Test scene manager get_scene method"""
        scene = mock_scene_manager.get_scene("test_scene")

        assert scene.scene_id == "test_scene"
        assert scene.num_frames == 100
        mock_scene_manager.get_scene.assert_called_once_with("test_scene")

    def test_scene_manager_list_scenes(self, mock_scene_manager):
        """Test scene manager list_scenes method"""
        scenes = mock_scene_manager.list_scenes()

        assert scenes == ["test_scene"]
        mock_scene_manager.list_scenes.assert_called_once()

    def test_frame_index_mapper_integration(self, mock_scene_metadata):
        """Test frame index mapper with real metadata"""
        mapper = FrameIndexMapper(mock_scene_metadata)

        # Test timestamp to frame conversion
        frame_idx = mapper.get_frame_index(timestamp_us=5_000_000)
        assert frame_idx == 50

        # Test frame range
        frames = mapper.get_frame_range(duration_us=1_000_000)
        assert len(frames) == 11  # frames 0-10

    def test_config_initialization(self, mock_config):
        """Test server config initialization"""
        assert mock_config.grpc_host == "localhost"
        assert mock_config.grpc_port == 50052
        assert mock_config.max_workers == 10
        assert mock_config.device == "cuda"

    def test_config_from_yaml_mock(self):
        """Test config loading from YAML (mocked)"""
        yaml_content = {
            'grpc': {'host': 'testhost', 'port': 12345, 'max_workers': 5},
            'device': 'cpu',
            'scene': {'base_path': '/test/path'},
            'render': {'default_format': 'PNG'},
            'cache': {'max_cached_scenes': 20}
        }

        with patch('dggt_server.config.DGGTServerConfig.from_yaml') as mock_from_yaml:
            mock_from_yaml.return_value = DGGTServerConfig(
                grpc_host='testhost',
                grpc_port=12345,
                max_workers=5,
                device='cpu'
            )

            config = DGGTServerConfig.from_yaml('/fake/path.yaml')
            assert config.grpc_host == 'testhost'
            assert config.grpc_port == 12345

    def test_scene_metadata_frame_conversion(self, mock_scene_metadata):
        """Test scene metadata frame conversion methods"""
        # Timestamp to frame
        frame = mock_scene_metadata.get_frame_index_from_timestamp(5_000_000)
        assert frame == 50

        # Frame to timestamp
        timestamp = mock_scene_metadata.get_timestamp_from_frame_index(50)
        assert timestamp == 5_000_000

    def test_frame_metadata_structure(self, mock_frame_metadata):
        """Test frame metadata structure"""
        assert mock_frame_metadata.frame_idx == 50
        assert mock_frame_metadata.timestamp_us == 5_000_000
        assert mock_frame_metadata.c2w_matrix.shape == (4, 4)
        assert mock_frame_metadata.intrinsic_matrix.shape == (3, 3)

    def test_dynamic_objects_handling(self):
        """Test dynamic object metadata handling"""
        objects = [
            ObjectMetadata(
                object_id=0,
                pose_world=np.eye(4),
                dimensions=np.array([4.0, 2.0, 1.5])
            ),
            ObjectMetadata(
                object_id=1,
                pose_world=np.eye(4),
                dimensions=np.array([3.0, 1.5, 1.0])
            )
        ]

        frame = FrameMetadata(
            frame_idx=0,
            timestamp_us=0,
            c2w_matrix=np.eye(4),
            intrinsic_matrix=np.eye(3),
            width=518,
            height=350,
            objects=objects
        )

        assert len(frame.objects) == 2
        assert frame.objects[0].object_id == 0
        assert np.allclose(frame.objects[1].dimensions, np.array([3.0, 1.5, 1.0]))


class TestDGGTServiceErrorHandling:
    """Tests for error handling in DGGT service"""

    def test_scene_not_found_handling(self):
        """Test handling of scene not found error"""
        from dggt_server.exceptions import SceneNotFoundError

        with pytest.raises(SceneNotFoundError):
            raise SceneNotFoundError("missing_scene", "/path/to/missing")

    def test_invalid_pose_error(self):
        """Test handling of invalid pose error"""
        from dggt_server.exceptions import InvalidPoseError

        with pytest.raises(InvalidPoseError):
            raise InvalidPoseError("Rotation matrix is not orthogonal")

    def test_invalid_camera_spec_error(self):
        """Test handling of invalid camera spec error"""
        from dggt_server.exceptions import InvalidCameraSpecError

        with pytest.raises(InvalidCameraSpecError):
            raise InvalidCameraSpecError("fx", -100, "Focal length must be positive")

    def test_track_id_mapping_error(self):
        """Test handling of track ID mapping error"""
        from dggt_server.exceptions import TrackIDMappingError

        with pytest.raises(TrackIDMappingError):
            raise TrackIDMappingError("unknown_id", "No mapping found")

    def test_rendering_error(self):
        """Test handling of rendering error"""
        from dggt_server.exceptions import RenderingError

        with pytest.raises(RenderingError):
            raise RenderingError("gaussian_splat", "CUDA out of memory")


class TestDGGTServiceIntegration:
    """Integration tests using mock scene manager"""

    @pytest.fixture
    def full_mock_setup(self):
        """Create full mock setup for integration testing"""
        metadata = DGGTSceneMetadata(
            scene_id="integration_test",
            scene_path="/test/integration",
            num_frames=200,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=19_900_000,
            camera_width=1024,
            camera_height=768,
            intrinsic_matrix=np.array([[2000, 0, 512], [0, 2000, 384], [0, 0, 1]]),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[0, 1, 2],
            static_scene_size_mb=500.0,
            total_size_mb=600.0
        )

        mapper = FrameIndexMapper(metadata)

        return {
            'metadata': metadata,
            'mapper': mapper
        }

    def test_full_workflow(self, full_mock_setup):
        """Test full workflow from timestamp to frame"""
        mapper = full_mock_setup['mapper']
        metadata = full_mock_setup['metadata']

        # 1. Get frame from timestamp
        frame_idx = mapper.get_frame_index(timestamp_us=10_000_000)
        assert frame_idx == 100

        # 2. Get timestamp from frame
        timestamp = metadata.get_timestamp_from_frame_index(frame_idx)
        assert timestamp == 10_000_000

        # 3. Get frame range for batch processing
        frames = mapper.get_frame_range(duration_us=2_000_000, start_timestamp_us=10_000_000)
        assert frames[0] == 100
        assert frames[-1] == 120

        # 4. Validate timestamp
        is_valid, clamped = mapper.validate_timestamp(10_000_000)
        assert is_valid == True


class TestDGGTServicerender_rgbWorkflow:
    """Tests for render_rgb() workflow with mock scene_manager"""

    @pytest.fixture
    def mock_scene_metadata(self):
        """Create mock scene metadata for render tests"""
        return DGGTSceneMetadata(
            scene_id="test_scene",
            scene_path="/path/to/scene",
            num_frames=100,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=9_900_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[0, 1],
            static_scene_size_mb=100.0,
            total_size_mb=120.0
        )

    @pytest.fixture
    def mock_frame_metadata(self):
        """Create mock frame metadata for render tests"""
        return FrameMetadata(
            frame_idx=50,
            timestamp_us=5_000_000,
            c2w_matrix=np.eye(4),
            intrinsic_matrix=np.array([[1000, 0, 259], [0, 1000, 175], [0, 0, 1]]),
            width=518,
            height=350,
            objects=[]
        )

    @pytest.fixture
    def mock_scene_manager(self, mock_scene_metadata, mock_frame_metadata):
        """Create mock scene manager with proper returns"""
        manager = Mock(spec=DGGTSceneManager)
        manager.get_scene.return_value = mock_scene_metadata
        manager.get_frame_metadata.return_value = mock_frame_metadata
        manager.list_scenes.return_value = ["test_scene"]
        return manager

    @pytest.fixture
    def mock_config(self):
        """Create mock server config"""
        return DGGTServerConfig(
            grpc_host="localhost",
            grpc_port=50052,
            max_workers=10,
            device="cuda",
            scene_base_path="/test/path"
        )

    @pytest.fixture
    def mock_context(self):
        """Create mock gRPC context"""
        return MockContext()

    @pytest.fixture
    def mock_rgb_request(self):
        """Create mock RGBRenderRequest"""
        from nre.grpc.protos import sensorsim_pb2 as sensorsim_pb
        from nre.grpc.protos import common_pb2 as common_pb

        request = sensorsim_pb.RGBRenderRequest(
            scene_id="test_scene",
            frame_start_us=5_000_000,
            frame_end_us=5_100_000,
            resolution_w=518,
            resolution_h=350,
            image_format=sensorsim_pb.ImageFormat.JPEG,
            image_quality=90,
            # Add valid sensor_pose with two poses for interpolation
            sensor_pose=sensorsim_pb.PosePair(
                start_pose=common_pb.Pose(
                    vec=common_pb.Vec3(x=0.0, y=0.0, z=0.0),
                    quat=common_pb.Quat(x=0.0, y=0.0, z=0.0, w=1.0)
                ),
                end_pose=common_pb.Pose(
                    vec=common_pb.Vec3(x=0.0, y=0.0, z=0.0),
                    quat=common_pb.Quat(x=0.0, y=0.0, z=0.0, w=1.0)
                )
            )
        )
        return request

    def test_render_rgb_workflow_success(
        self,
        mock_scene_manager,
        mock_config,
        mock_context,
        mock_rgb_request,
        mock_scene_metadata,
        mock_frame_metadata
    ):
        """Test the full render_rgb flow with mock scene_manager and mock context"""
        from dggt_server.dggt_service import DGGTService
        import dggt_server.dggt_service as svc_module

        # Create mock renderer that returns a valid test image
        mock_renderer = Mock()
        test_image = np.zeros((350, 518, 3), dtype=np.uint8)
        test_image[:] = 128  # Gray image
        mock_renderer.render_frame.return_value = test_image

        # Patch renderer availability and _get_renderer
        original_available = svc_module._RENDERER_AVAILABLE
        svc_module._RENDERER_AVAILABLE = True

        # Create DGGTService instance
        service = DGGTService(
            scene_manager=mock_scene_manager,
            config=mock_config,
            t_carla_dggt=np.eye(4, dtype=np.float32)
        )

        # Inject mock renderer
        service._renderers["test_scene"] = mock_renderer

        # Reset mock call count to isolate render_rgb behavior
        mock_scene_manager.reset_mock()

        # Re-setup return values after reset
        mock_scene_manager.get_scene.return_value = mock_scene_metadata
        mock_scene_manager.get_frame_metadata.return_value = mock_frame_metadata
        mock_scene_manager.list_scenes.return_value = ["test_scene"]

        try:
            # Call render_rgb
            result = service.render_rgb(mock_rgb_request, mock_context)

            # Verify scene_manager methods were called correctly
            mock_scene_manager.get_scene.assert_called_once_with("test_scene")
            mock_scene_manager.get_frame_metadata.assert_called_once_with("test_scene", 50)

            # Verify mock renderer was called
            mock_renderer.render_frame.assert_called_once()

            # Verify result contains image_bytes
            assert result.image_bytes is not None
            assert len(result.image_bytes) > 0

            # Verify context was not set to error state
            assert mock_context.code is None
            assert mock_context.details is None
        finally:
            svc_module._RENDERER_AVAILABLE = original_available

    def test_render_rgb_scene_not_found(
        self,
        mock_scene_manager,
        mock_config,
        mock_context,
        mock_rgb_request
    ):
        """Test that wrong scene_id returns NOT_FOUND status"""
        from dggt_server.dggt_service import DGGTService
        from dggt_server.exceptions import SceneNotFoundError

        # Mock scene_manager to raise SceneNotFoundError for wrong scene
        mock_scene_manager.get_scene.side_effect = SceneNotFoundError("wrong_scene", "/path/to/missing")

        # Update request with wrong scene_id
        mock_rgb_request.scene_id = "wrong_scene"

        # Create DGGTService instance
        service = DGGTService(
            scene_manager=mock_scene_manager,
            config=mock_config,
            t_carla_dggt=np.eye(4, dtype=np.float32)
        )

        # Call render_rgb with wrong scene_id
        result = service.render_rgb(mock_rgb_request, mock_context)

        # Verify context was set to NOT_FOUND
        assert mock_context.code == grpc.StatusCode.NOT_FOUND
        assert "Scene not found" in mock_context.details

        # Verify result is empty
        assert len(result.image_bytes) == 0

    def test_render_rgb_frame_metadata_error(
        self,
        mock_scene_manager,
        mock_config,
        mock_context,
        mock_rgb_request,
        mock_scene_metadata
    ):
        """Test that frame metadata loading error returns INTERNAL status"""
        from dggt_server.dggt_service import DGGTService

        # Mock get_scene to succeed
        mock_scene_manager.get_scene.return_value = mock_scene_metadata

        # Mock get_frame_metadata to fail
        mock_scene_manager.get_frame_metadata.side_effect = Exception("Frame loading failed")

        # Create DGGTService instance
        service = DGGTService(
            scene_manager=mock_scene_manager,
            config=mock_config,
            t_carla_dggt=np.eye(4, dtype=np.float32)
        )

        # Call render_rgb
        result = service.render_rgb(mock_rgb_request, mock_context)

        # Verify context was set to INTERNAL
        assert mock_context.code == grpc.StatusCode.INTERNAL
        assert "Failed to load frame metadata" in mock_context.details

        # Verify result is empty
        assert len(result.image_bytes) == 0

    def test_mock_context_methods(self, mock_context):
        """Test MockContext class has set_code() and set_details() methods"""
        # Test set_code
        mock_context.set_code(grpc.StatusCode.NOT_FOUND)
        assert mock_context.code == grpc.StatusCode.NOT_FOUND

        # Test set_details
        mock_context.set_details("Test details message")
        assert mock_context.details == "Test details message"

        # Test multiple calls
        mock_context.set_code(grpc.StatusCode.INTERNAL)
        mock_context.set_details("Another message")
        assert mock_context.code == grpc.StatusCode.INTERNAL
        assert mock_context.details == "Another message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])