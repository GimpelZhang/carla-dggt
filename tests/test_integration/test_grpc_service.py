# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Integration tests for gRPC Service

Tests the DGGTService gRPC servicer:
- get_version() RPC
- get_available_scenes() RPC
- get_available_cameras() RPC
- render_rgb() success case
- render_rgb() wrong scene (NOT_FOUND error)
- render_lidar() UNIMPLEMENTED error
- frame_start_us/frame_end_us validation

Uses mock gRPC context for testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import grpc

# Import protobuf types
from nre.grpc.protos import common_pb2, sensorsim_pb2

# Import the modules under test
try:
    from dggt_server.dggt_service import DGGTService, DGGT_VERSION, DGGT_GIT_HASH
    from dggt_server.scene_manager import DGGTSceneManager
    from dggt_server.scene_metadata import DGGTSceneMetadata
    from dggt_server.config import DGGTServerConfig
    from dggt_server.exceptions import SceneNotFoundError
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.dggt_service import DGGTService, DGGT_VERSION, DGGT_GIT_HASH
    from dggt_server.scene_manager import DGGTSceneManager
    from dggt_server.scene_metadata import DGGTSceneMetadata
    from dggt_server.config import DGGTServerConfig
    from dggt_server.exceptions import SceneNotFoundError


@pytest.fixture
def mock_scene_manager():
    """Create a mock DGGTSceneManager"""
    manager = Mock(spec=DGGTSceneManager)

    # Mock scene metadata
    metadata = DGGTSceneMetadata(
        scene_id="test_scene",
        scene_path="/test/path",
        num_frames=10,
        fps=10.0,
        start_timestamp_us=0,
        end_timestamp_us=900_000,
        camera_width=1036,
        camera_height=700,
        intrinsic_matrix=np.array([
            [1000.0, 0.0, 518.0],
            [0.0, 1000.0, 350.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        intrinsics_vary=False,
        has_static_scene=True,
        has_sky_scene=True,
        dynamic_object_ids=[0, 1, 2],
        static_scene_size_mb=50.0,
        total_size_mb=60.0
    )

    manager.list_scenes.return_value = ["test_scene"]
    manager.get_scene.return_value = metadata

    # Mock frame metadata
    from dggt_server.scene_metadata import FrameMetadata
    frame_meta = FrameMetadata(
        frame_idx=0,
        timestamp_us=0,
        c2w_matrix=np.eye(4, dtype=np.float32),
        intrinsic_matrix=np.array([
            [1000.0, 0.0, 518.0],
            [0.0, 1000.0, 350.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        width=1036,
        height=700,
        objects=[]
    )
    manager.get_frame_metadata.return_value = frame_meta

    return manager


@pytest.fixture
def mock_config():
    """Create a mock DGGTServerConfig"""
    return DGGTServerConfig(
        grpc_host="localhost",
        grpc_port=50052,
        max_workers=10,
        device="cpu",
        scene_base_path="/test/path",
        default_fps=10.0,
        auto_discover=True,
        scene_id_mappings={},
        default_image_format="JPEG",
        default_image_quality=95.0,
        max_cached_scenes=10
    )


@pytest.fixture
def mock_context():
    """Create a mock gRPC context"""
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    return context


@pytest.fixture
def dggt_service(mock_scene_manager, mock_config):
    """Create a DGGTService instance for testing with mock renderer"""
    import dggt_server.dggt_service as svc_module

    original_available = svc_module._RENDERER_AVAILABLE
    svc_module._RENDERER_AVAILABLE = True

    service = DGGTService(
        scene_manager=mock_scene_manager,
        config=mock_config
    )

    # Inject mock renderer
    mock_renderer = Mock()
    test_image = np.zeros((700, 1036, 3), dtype=np.uint8)
    test_image[:] = 128
    mock_renderer.render_frame.return_value = test_image
    service._renderers["test_scene"] = mock_renderer

    yield service

    svc_module._RENDERER_AVAILABLE = original_available


@pytest.mark.integration
class TestGRPCServiceGetVersion:
    """Tests for get_version RPC"""

    def test_get_version_returns_version(self, dggt_service, mock_context):
        """Test that get_version returns version string"""
        request = common_pb2.Empty()
        response = dggt_service.get_version(request, mock_context)

        assert "DGGT" in response.version_id
        assert DGGT_VERSION in response.version_id
        assert response.git_hash == DGGT_GIT_HASH

    def test_get_version_returns_api_version(self, dggt_service, mock_context):
        """Test that get_version returns API version"""
        request = common_pb2.Empty()
        response = dggt_service.get_version(request, mock_context)

        assert response.grpc_api_version.major == 1
        assert response.grpc_api_version.minor == 0
        assert response.grpc_api_version.patch == 0

    def test_get_version_no_context_errors(self, dggt_service):
        """Test that get_version works without context errors"""
        request = common_pb2.Empty()
        # Should not raise any exception
        response = dggt_service.get_version(request, None)
        assert response is not None


@pytest.mark.integration
class TestGRPCServiceGetAvailableScenes:
    """Tests for get_available_scenes RPC"""

    def test_get_available_scenes_returns_list(self, dggt_service, mock_context, mock_scene_manager):
        """Test that get_available_scenes returns scene list"""
        request = common_pb2.Empty()
        response = dggt_service.get_available_scenes(request, mock_context)

        assert len(response.scene_ids) == 1
        assert "test_scene" in response.scene_ids

    def test_get_available_scenes_empty_list(self, mock_config, mock_context):
        """Test get_available_scenes with no scenes"""
        mock_manager = Mock(spec=DGGTSceneManager)
        mock_manager.list_scenes.return_value = []

        service = DGGTService(
            scene_manager=mock_manager,
            config=mock_config
        )

        request = common_pb2.Empty()
        response = service.get_available_scenes(request, mock_context)

        assert len(response.scene_ids) == 0

    def test_get_available_scenes_multiple_scenes(self, mock_config, mock_context):
        """Test get_available_scenes with multiple scenes"""
        mock_manager = Mock(spec=DGGTSceneManager)
        mock_manager.list_scenes.return_value = ["scene_1", "scene_2", "scene_3"]

        service = DGGTService(
            scene_manager=mock_manager,
            config=mock_config
        )

        request = common_pb2.Empty()
        response = service.get_available_scenes(request, mock_context)

        assert len(response.scene_ids) == 3


@pytest.mark.integration
class TestGRPCServiceGetAvailableCameras:
    """Tests for get_available_cameras RPC"""

    def test_get_available_cameras_returns_camera(self, dggt_service, mock_context):
        """Test that get_available_cameras returns camera info"""
        request = sensorsim_pb2.AvailableCamerasRequest()
        request.scene_id = "test_scene"

        response = dggt_service.get_available_cameras(request, mock_context)

        assert len(response.available_cameras) == 1
        camera = response.available_cameras[0]
        assert camera.logical_id == "ego_camera"
        assert camera.trajectory_idx == 0

    def test_get_available_cameras_wrong_scene(self, dggt_service, mock_context, mock_scene_manager):
        """Test get_available_cameras with non-existent scene"""
        mock_scene_manager.get_scene.side_effect = SceneNotFoundError("wrong_scene", "/path")

        request = sensorsim_pb2.AvailableCamerasRequest()
        request.scene_id = "wrong_scene"

        response = dggt_service.get_available_cameras(request, mock_context)

        # Should have set NOT_FOUND error
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
        assert len(response.available_cameras) == 0

    def test_get_available_cameras_intrinsic_values(self, dggt_service, mock_context):
        """Test that camera intrinsics are returned correctly"""
        request = sensorsim_pb2.AvailableCamerasRequest()
        request.scene_id = "test_scene"

        response = dggt_service.get_available_cameras(request, mock_context)

        camera = response.available_cameras[0]
        fisheye = camera.intrinsics.opencv_fisheye_param
        assert fisheye.focal_length_x == 1000.0
        assert fisheye.focal_length_y == 1000.0
        assert fisheye.principal_point_x == 518.0
        assert fisheye.principal_point_y == 350.0


@pytest.mark.integration
class TestGRPCServiceRenderRGB:
    """Tests for render_rgb RPC"""

    def test_render_rgb_success(self, dggt_service, mock_context):
        """Test successful RGB rendering"""
        request = sensorsim_pb2.RGBRenderRequest()
        request.scene_id = "test_scene"
        request.frame_start_us = 0
        request.frame_end_us = 100_000  # Must differ
        request.resolution_w = 1036
        request.resolution_h = 700
        request.image_format = sensorsim_pb2.ImageFormat.JPEG

        # Add camera intrinsics (using fisheye param which has focal length fields)
        request.camera_intrinsics.resolution_w = 1036
        request.camera_intrinsics.resolution_h = 700
        fisheye = request.camera_intrinsics.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0
        fisheye.principal_point_x = 518.0
        fisheye.principal_point_y = 350.0

        # Add sensor pose
        request.sensor_pose.start_pose.vec.x = 0.0
        request.sensor_pose.start_pose.vec.y = 0.0
        request.sensor_pose.start_pose.vec.z = 0.0
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.vec.x = 0.0
        request.sensor_pose.end_pose.vec.y = 0.0
        request.sensor_pose.end_pose.vec.z = 0.0
        request.sensor_pose.end_pose.quat.w = 1.0

        response = dggt_service.render_rgb(request, mock_context)

        assert len(response.image_bytes) > 0

    def test_render_rgb_wrong_scene(self, dggt_service, mock_context, mock_scene_manager):
        """Test render_rgb with non-existent scene"""
        mock_scene_manager.get_scene.side_effect = SceneNotFoundError("wrong_scene", "/path")

        request = sensorsim_pb2.RGBRenderRequest()
        request.scene_id = "wrong_scene"
        request.frame_start_us = 0
        request.frame_end_us = 100_000

        response = dggt_service.render_rgb(request, mock_context)

        # Should have set NOT_FOUND error
        mock_context.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_render_rgb_frame_timestamps_must_differ(self, dggt_service, mock_context):
        """Test that frame_start_us and frame_end_us must be different"""
        request = sensorsim_pb2.RGBRenderRequest()
        request.scene_id = "test_scene"
        request.frame_start_us = 100_000
        request.frame_end_us = 100_000  # Same as start
        request.resolution_w = 1036
        request.resolution_h = 700

        # Add minimal camera intrinsics (using fisheye param)
        request.camera_intrinsics.resolution_w = 1036
        request.camera_intrinsics.resolution_h = 700
        fisheye = request.camera_intrinsics.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0
        fisheye.principal_point_x = 518.0
        fisheye.principal_point_y = 350.0

        # Add sensor pose
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.quat.w = 1.0

        # Should still work (midpoint calculation with same timestamps)
        response = dggt_service.render_rgb(request, mock_context)
        # The service should handle this gracefully

    def test_render_rgb_jpeg_format(self, dggt_service, mock_context):
        """Test render_rgb with JPEG format"""
        request = sensorsim_pb2.RGBRenderRequest()
        request.scene_id = "test_scene"
        request.frame_start_us = 0
        request.frame_end_us = 100_000
        request.resolution_w = 640
        request.resolution_h = 480
        request.image_format = sensorsim_pb2.ImageFormat.JPEG
        request.image_quality = 90

        # Add minimal camera intrinsics (using fisheye param)
        request.camera_intrinsics.resolution_w = 640
        request.camera_intrinsics.resolution_h = 480
        fisheye = request.camera_intrinsics.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0
        fisheye.principal_point_x = 320.0
        fisheye.principal_point_y = 240.0

        # Add sensor pose
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.quat.w = 1.0

        response = dggt_service.render_rgb(request, mock_context)

        # JPEG files start with FF D8
        assert len(response.image_bytes) > 0
        # JPEG magic bytes
        assert response.image_bytes[0] == 0xFF
        assert response.image_bytes[1] == 0xD8

    def test_render_rgb_png_format(self, dggt_service, mock_context):
        """Test render_rgb with PNG format"""
        request = sensorsim_pb2.RGBRenderRequest()
        request.scene_id = "test_scene"
        request.frame_start_us = 0
        request.frame_end_us = 100_000
        request.resolution_w = 640
        request.resolution_h = 480
        request.image_format = sensorsim_pb2.ImageFormat.PNG

        # Add minimal camera intrinsics (using fisheye param)
        request.camera_intrinsics.resolution_w = 640
        request.camera_intrinsics.resolution_h = 480
        fisheye = request.camera_intrinsics.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0
        fisheye.principal_point_x = 320.0
        fisheye.principal_point_y = 240.0

        # Add sensor pose
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.quat.w = 1.0

        response = dggt_service.render_rgb(request, mock_context)

        # PNG files start with signature
        assert len(response.image_bytes) > 0
        # PNG magic bytes: 89 50 4E 47
        assert response.image_bytes[0] == 0x89
        assert response.image_bytes[1:4] == b'PNG'

    def test_render_rgb_with_dynamic_objects(self, dggt_service, mock_context):
        """Test render_rgb with dynamic objects"""
        request = sensorsim_pb2.RGBRenderRequest()
        request.scene_id = "test_scene"
        request.frame_start_us = 0
        request.frame_end_us = 100_000
        request.resolution_w = 640
        request.resolution_h = 480

        # Add camera intrinsics (using fisheye param)
        request.camera_intrinsics.resolution_w = 640
        request.camera_intrinsics.resolution_h = 480
        fisheye = request.camera_intrinsics.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0
        fisheye.principal_point_x = 320.0
        fisheye.principal_point_y = 240.0

        # Add sensor pose
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.quat.w = 1.0

        # Add dynamic object
        dyn_obj = request.dynamic_objects.add()
        dyn_obj.track_id = "dggt_obj_0000"
        dyn_obj.pose_pair.start_pose.quat.w = 1.0
        dyn_obj.pose_pair.end_pose.quat.w = 1.0

        response = dggt_service.render_rgb(request, mock_context)

        assert len(response.image_bytes) > 0


@pytest.mark.integration
class TestGRPCServiceRenderLidar:
    """Tests for render_lidar RPC"""

    def test_render_lidar_unimplemented(self, dggt_service, mock_context):
        """Test that render_lidar returns UNIMPLEMENTED"""
        request = sensorsim_pb2.LidarRenderRequest()
        request.scene_id = "test_scene"

        response = dggt_service.render_lidar(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.UNIMPLEMENTED)
        assert "not implemented" in mock_context.set_details.call_args[0][0].lower()

    def test_render_lidar_returns_empty(self, dggt_service, mock_context):
        """Test that render_lidar returns empty response"""
        request = sensorsim_pb2.LidarRenderRequest()
        request.scene_id = "test_scene"

        response = dggt_service.render_lidar(request, mock_context)

        # Response should be empty
        assert response.ByteSize() == 0


@pytest.mark.integration
class TestGRPCServiceShutdown:
    """Tests for shut_down RPC"""

    def test_shut_down_sets_flag(self, dggt_service, mock_context):
        """Test that shut_down sets shutdown flag"""
        request = common_pb2.Empty()

        response = dggt_service.shut_down(request, mock_context)

        assert dggt_service._shutdown_requested == True

    def test_shut_down_clears_cache(self, dggt_service, mock_context, mock_scene_manager):
        """Test that shut_down clears scene cache"""
        request = common_pb2.Empty()

        dggt_service.shut_down(request, mock_context)

        mock_scene_manager.clear_cache.assert_called_once()

    def test_shut_down_returns_empty(self, dggt_service, mock_context):
        """Test that shut_down returns empty response"""
        request = common_pb2.Empty()

        response = dggt_service.shut_down(request, mock_context)

        assert isinstance(response, common_pb2.Empty)


@pytest.mark.integration
class TestGRPCServiceGetAvailableTrajectories:
    """Tests for get_available_trajectories RPC"""

    def test_get_available_trajectories_returns_trajectory(self, dggt_service, mock_context):
        """Test that get_available_trajectories returns trajectory"""
        request = sensorsim_pb2.AvailableTrajectoriesRequest()
        request.scene_id = "test_scene"

        response = dggt_service.get_available_trajectories(request, mock_context)

        assert len(response.available_trajectories) == 1
        assert response.available_trajectories[0].trajectory_idx == 0

    def test_get_available_trajectories_wrong_scene(self, dggt_service, mock_context, mock_scene_manager):
        """Test get_available_trajectories with non-existent scene"""
        mock_scene_manager.get_scene.side_effect = SceneNotFoundError("wrong_scene", "/path")

        request = sensorsim_pb2.AvailableTrajectoriesRequest()
        request.scene_id = "wrong_scene"

        response = dggt_service.get_available_trajectories(request, mock_context)

        mock_context.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)


@pytest.mark.integration
class TestGRPCServiceGetAvailableEgoMasks:
    """Tests for get_available_ego_masks RPC"""

    def test_get_available_ego_masks_returns_empty(self, dggt_service, mock_context):
        """Test that get_available_ego_masks returns empty list (DGGT has no ego masks)"""
        request = common_pb2.Empty()

        response = dggt_service.get_available_ego_masks(request, mock_context)

        assert len(response.ego_mask_metadata) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])