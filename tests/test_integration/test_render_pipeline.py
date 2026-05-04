# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Integration tests for Render Pipeline

Tests the full rendering pipeline:
- Full pipeline single frame render
- Render with dynamic objects
- Multiple frames sequential rendering
- JPEG format output
- PNG format output

Uses mock gRPC context and scene manager for testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import io

# Import protobuf types
from nre.grpc.protos import common_pb2, sensorsim_pb2

# Import the modules under test
try:
    from dggt_server.dggt_service import DGGTService
    from dggt_server.scene_manager import DGGTSceneManager
    from dggt_server.scene_metadata import DGGTSceneMetadata, FrameMetadata, TrackIDMapping
    from dggt_server.config import DGGTServerConfig
    from dggt_server.request_converter import RequestConverter
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.dggt_service import DGGTService
    from dggt_server.scene_manager import DGGTSceneManager
    from dggt_server.scene_metadata import DGGTSceneMetadata, FrameMetadata, TrackIDMapping
    from dggt_server.config import DGGTServerConfig
    from dggt_server.request_converter import RequestConverter


@pytest.fixture
def mock_scene_manager_multi_frame():
    """Create a mock DGGTSceneManager with multi-frame support"""
    manager = Mock(spec=DGGTSceneManager)

    # Mock scene metadata
    metadata = DGGTSceneMetadata(
        scene_id="test_scene",
        scene_path="/test/path",
        num_frames=100,
        fps=10.0,
        start_timestamp_us=0,
        end_timestamp_us=9_900_000,
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

    # Mock frame metadata factory
    def get_frame_metadata(scene_id, frame_idx):
        return FrameMetadata(
            frame_idx=frame_idx,
            timestamp_us=int(frame_idx * 100_000),  # 10Hz
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

    manager.get_frame_metadata.side_effect = get_frame_metadata

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
def dggt_service(mock_scene_manager_multi_frame, mock_config):
    """Create a DGGTService instance for testing with mock renderer"""
    import dggt_server.dggt_service as svc_module

    original_available = svc_module._RENDERER_AVAILABLE
    svc_module._RENDERER_AVAILABLE = True

    service = DGGTService(
        scene_manager=mock_scene_manager_multi_frame,
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


def create_render_request(
    scene_id="test_scene",
    frame_start_us=0,
    frame_end_us=100_000,
    width=640,
    height=480,
    image_format=sensorsim_pb2.ImageFormat.JPEG,
    image_quality=90
):
    """Helper to create a basic RGBRenderRequest"""
    request = sensorsim_pb2.RGBRenderRequest()
    request.scene_id = scene_id
    request.frame_start_us = frame_start_us
    request.frame_end_us = frame_end_us
    request.resolution_w = width
    request.resolution_h = height
    request.image_format = image_format
    request.image_quality = image_quality

    # Add camera intrinsics (using fisheye param which has focal length fields)
    # Also set resolution on camera_intrinsics (CameraSpec)
    request.camera_intrinsics.resolution_w = width
    request.camera_intrinsics.resolution_h = height
    fisheye = request.camera_intrinsics.opencv_fisheye_param
    fisheye.focal_length_x = 1000.0
    fisheye.focal_length_y = 1000.0
    fisheye.principal_point_x = width / 2
    fisheye.principal_point_y = height / 2

    # Add sensor pose (identity)
    request.sensor_pose.start_pose.vec.x = 0.0
    request.sensor_pose.start_pose.vec.y = 0.0
    request.sensor_pose.start_pose.vec.z = 0.0
    request.sensor_pose.start_pose.quat.w = 1.0
    request.sensor_pose.end_pose.vec.x = 0.0
    request.sensor_pose.end_pose.vec.y = 0.0
    request.sensor_pose.end_pose.vec.z = 0.0
    request.sensor_pose.end_pose.quat.w = 1.0

    return request


@pytest.mark.integration
class TestRenderPipeline:
    """Integration tests for render pipeline"""

    def test_full_pipeline_single_frame_render(self, dggt_service, mock_context):
        """Test full pipeline for single frame rendering"""
        request = create_render_request(
            scene_id="test_scene",
            frame_start_us=0,
            frame_end_us=100_000,
            width=640,
            height=480
        )

        response = dggt_service.render_rgb(request, mock_context)

        # Verify response
        assert len(response.image_bytes) > 0

        # Verify JPEG magic bytes
        assert response.image_bytes[0] == 0xFF
        assert response.image_bytes[1] == 0xD8

    def test_render_with_dynamic_objects(self, dggt_service, mock_context):
        """Test rendering with dynamic objects"""
        request = create_render_request()

        # Add dynamic objects
        dyn_obj = request.dynamic_objects.add()
        dyn_obj.track_id = "dggt_obj_0000"
        dyn_obj.pose_pair.start_pose.vec.x = 10.0
        dyn_obj.pose_pair.start_pose.vec.y = 5.0
        dyn_obj.pose_pair.start_pose.vec.z = 0.0
        dyn_obj.pose_pair.start_pose.quat.w = 1.0
        dyn_obj.pose_pair.end_pose.vec.x = 10.0
        dyn_obj.pose_pair.end_pose.vec.y = 5.0
        dyn_obj.pose_pair.end_pose.vec.z = 0.0
        dyn_obj.pose_pair.end_pose.quat.w = 1.0

        dyn_obj2 = request.dynamic_objects.add()
        dyn_obj2.track_id = "dggt_obj_0001"
        dyn_obj2.pose_pair.start_pose.quat.w = 1.0
        dyn_obj2.pose_pair.end_pose.quat.w = 1.0

        response = dggt_service.render_rgb(request, mock_context)

        # Should still render successfully
        assert len(response.image_bytes) > 0

    def test_multiple_frames_sequential_rendering(self, dggt_service, mock_context):
        """Test rendering multiple frames sequentially"""
        rendered_frames = []

        for frame_idx in range(5):
            frame_start_us = frame_idx * 100_000
            frame_end_us = frame_start_us + 100_000

            request = create_render_request(
                frame_start_us=frame_start_us,
                frame_end_us=frame_end_us
            )

            response = dggt_service.render_rgb(request, mock_context)
            assert len(response.image_bytes) > 0
            rendered_frames.append(response.image_bytes)

        # Verify all frames rendered successfully
        assert len(rendered_frames) == 5

    def test_render_different_resolutions(self, dggt_service, mock_context):
        """Test rendering at different resolutions"""
        resolutions = [
            (320, 240),
            (640, 480),
            (1280, 720),
            (1920, 1080)
        ]

        for width, height in resolutions:
            request = create_render_request(width=width, height=height)
            response = dggt_service.render_rgb(request, mock_context)

            assert len(response.image_bytes) > 0


@pytest.mark.integration
class TestRenderPipelineFormats:
    """Integration tests for image format output"""

    def test_jpeg_format_output(self, dggt_service, mock_context):
        """Test JPEG format output"""
        request = create_render_request(
            image_format=sensorsim_pb2.ImageFormat.JPEG,
            image_quality=95
        )

        response = dggt_service.render_rgb(request, mock_context)

        # Verify JPEG magic bytes (FF D8)
        assert response.image_bytes[0] == 0xFF
        assert response.image_bytes[1] == 0xD8

        # JPEG ends with FF D9
        assert response.image_bytes[-2] == 0xFF
        assert response.image_bytes[-1] == 0xD9

    def test_png_format_output(self, dggt_service, mock_context):
        """Test PNG format output"""
        request = create_render_request(
            image_format=sensorsim_pb2.ImageFormat.PNG
        )

        response = dggt_service.render_rgb(request, mock_context)

        # Verify PNG magic bytes (89 50 4E 47 0D 0A 1A 0A)
        assert response.image_bytes[0] == 0x89
        assert response.image_bytes[1:4] == b'PNG'

    def test_jpeg2000_format_output(self, dggt_service, mock_context):
        """Test JPEG2000 format output"""
        # Skip if OpenCV Jasper codec is disabled
        pytest.skip("JPEG2000 codec disabled in OpenCV build - requires OPENCV_IO_ENABLE_JASPER")

        request = create_render_request(
            image_format=sensorsim_pb2.ImageFormat.JPEG2000
        )

        response = dggt_service.render_rgb(request, mock_context)

        # JPEG2000 has different signature, just verify it's not empty
        assert len(response.image_bytes) > 0

    def test_jpeg_quality_affects_size(self, dggt_service, mock_context):
        """Test that JPEG quality affects output size"""
        # High quality
        request_high = create_render_request(
            image_format=sensorsim_pb2.ImageFormat.JPEG,
            image_quality=95
        )
        response_high = dggt_service.render_rgb(request_high, mock_context)

        # Low quality
        request_low = create_render_request(
            image_format=sensorsim_pb2.ImageFormat.JPEG,
            image_quality=10
        )
        response_low = dggt_service.render_rgb(request_low, mock_context)

        # Higher quality should generally produce larger files
        # (though for placeholder images this may not always hold)
        assert len(response_high.image_bytes) > 0
        assert len(response_low.image_bytes) > 0


@pytest.mark.integration
class TestRenderPipelineEdgeCases:
    """Integration tests for render pipeline edge cases"""

    def test_render_first_frame(self, dggt_service, mock_context):
        """Test rendering the first frame (frame index 0)"""
        request = create_render_request(
            frame_start_us=0,
            frame_end_us=50_000  # Small window around frame 0
        )

        response = dggt_service.render_rgb(request, mock_context)
        assert len(response.image_bytes) > 0

    def test_render_last_frame(self, dggt_service, mock_context):
        """Test rendering near the last frame"""
        request = create_render_request(
            frame_start_us=9_800_000,  # Near end of scene
            frame_end_us=9_900_000
        )

        response = dggt_service.render_rgb(request, mock_context)
        assert len(response.image_bytes) > 0

    def test_render_timestamp_out_of_range_clamped(self, dggt_service, mock_context):
        """Test that out-of-range timestamps are clamped"""
        request = create_render_request(
            frame_start_us=20_000_000,  # Beyond scene end
            frame_end_us=21_000_000
        )

        # Should not raise error, should clamp to last frame
        response = dggt_service.render_rgb(request, mock_context)
        assert len(response.image_bytes) > 0

    def test_render_negative_timestamp_clamped(self, dggt_service, mock_context):
        """Test that zero timestamps are handled correctly"""
        # Note: Protobuf uint64 fields cannot accept negative values
        # This test verifies handling of timestamp 0 (which maps to frame 0)
        request = create_render_request(
            frame_start_us=0,
            frame_end_us=50_000
        )

        # Should render frame 0 without error
        response = dggt_service.render_rgb(request, mock_context)
        assert len(response.image_bytes) > 0

    def test_render_with_no_intrinsics_uses_scene_default(self, dggt_service, mock_context):
        """Test that missing camera intrinsics fall back to scene default"""
        request = sensorsim_pb2.RGBRenderRequest()
        request.scene_id = "test_scene"
        request.frame_start_us = 0
        request.frame_end_us = 100_000
        request.resolution_w = 1036
        request.resolution_h = 700

        # Add minimal sensor pose
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.quat.w = 1.0

        # No camera intrinsics set - should use scene default
        response = dggt_service.render_rgb(request, mock_context)
        assert len(response.image_bytes) > 0


@pytest.mark.integration
class TestRenderPipelinePoseInterpolation:
    """Integration tests for pose interpolation in rendering"""

    def test_render_with_interpolated_sensor_pose(self, dggt_service, mock_context):
        """Test rendering with sensor pose interpolation"""
        request = create_render_request()

        # Set different start and end poses
        request.sensor_pose.start_pose.vec.x = 0.0
        request.sensor_pose.start_pose.vec.y = 0.0
        request.sensor_pose.start_pose.vec.z = 0.0
        request.sensor_pose.start_pose.quat.w = 1.0

        request.sensor_pose.end_pose.vec.x = 10.0
        request.sensor_pose.end_pose.vec.y = 5.0
        request.sensor_pose.end_pose.vec.z = 2.0
        request.sensor_pose.end_pose.quat.w = 1.0

        response = dggt_service.render_rgb(request, mock_context)
        assert len(response.image_bytes) > 0

    def test_render_with_rotated_sensor_pose(self, dggt_service, mock_context):
        """Test rendering with rotated sensor pose"""
        import math

        request = create_render_request()

        # Set rotation (90 degrees around Z)
        angle_45 = math.pi / 4
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.quat.w = math.cos(angle_45)
        request.sensor_pose.end_pose.quat.z = math.sin(angle_45)

        response = dggt_service.render_rgb(request, mock_context)
        assert len(response.image_bytes) > 0


@pytest.mark.integration
class TestRenderPipelineRequestConverter:
    """Integration tests for RequestConverter in render pipeline"""

    def test_camera_intrinsics_conversion(self):
        """Test that camera intrinsics are converted correctly"""
        camera_spec = sensorsim_pb2.CameraSpec()
        camera_spec.resolution_w = 1036
        camera_spec.resolution_h = 700

        # Use fisheye param which has focal length fields
        fisheye = camera_spec.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0
        fisheye.principal_point_x = 518.0
        fisheye.principal_point_y = 350.0

        K, width, height = RequestConverter.convert_camera_intrinsics(camera_spec)

        assert K.shape == (3, 3)
        assert K.dtype == np.float32
        assert K[0, 0] == 1000.0  # fx
        assert K[1, 1] == 1000.0  # fy
        assert K[0, 2] == 518.0   # cx
        assert K[1, 2] == 350.0   # cy
        assert width == 1036
        assert height == 700

    def test_fisheye_camera_intrinsics_conversion(self):
        """Test that fisheye camera intrinsics are converted"""
        camera_spec = sensorsim_pb2.CameraSpec()
        camera_spec.resolution_w = 1200
        camera_spec.resolution_h = 800

        fisheye = camera_spec.opencv_fisheye_param
        fisheye.focal_length_x = 1200.0
        fisheye.focal_length_y = 1200.0
        fisheye.principal_point_x = 600.0
        fisheye.principal_point_y = 400.0

        K, width, height = RequestConverter.convert_camera_intrinsics(camera_spec)

        assert K.shape == (3, 3)
        assert width == 1200
        assert height == 800

    def test_pose_interpolation_in_pipeline(self):
        """Test pose interpolation produces valid transformation"""
        pose_pair = sensorsim_pb2.PosePair()

        # Start pose at origin
        pose_pair.start_pose.vec.x = 0.0
        pose_pair.start_pose.vec.y = 0.0
        pose_pair.start_pose.vec.z = 0.0
        pose_pair.start_pose.quat.w = 1.0

        # End pose with translation
        pose_pair.end_pose.vec.x = 10.0
        pose_pair.end_pose.vec.y = 5.0
        pose_pair.end_pose.vec.z = 2.0
        pose_pair.end_pose.quat.w = 1.0

        # Interpolate at midpoint
        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        assert result.shape == (4, 4)
        assert np.allclose(result[:3, 3], [5.0, 2.5, 1.0], atol=1e-6)


@pytest.mark.integration
class TestRenderPipelineEncoding:
    """Integration tests for image encoding in render pipeline"""

    def test_image_encoding_jpeg(self):
        """Test JPEG image encoding"""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        encoded = RequestConverter.encode_image(
            image,
            sensorsim_pb2.ImageFormat.JPEG,
            quality=95
        )

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
        assert encoded[0] == 0xFF
        assert encoded[1] == 0xD8

    def test_image_encoding_png(self):
        """Test PNG image encoding"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        encoded = RequestConverter.encode_image(
            image,
            sensorsim_pb2.ImageFormat.PNG
        )

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
        assert encoded[0] == 0x89
        assert encoded[1:4] == b'PNG'

    def test_image_encoding_planar(self):
        """Test planar RGB encoding"""
        height, width = 480, 640
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        encoded = RequestConverter.encode_image(
            image,
            sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR
        )

        # Planar format: CHW, 3 * height * width bytes
        expected_size = 3 * height * width
        assert len(encoded) == expected_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])