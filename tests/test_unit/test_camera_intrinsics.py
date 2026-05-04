# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for Camera Intrinsics Conversion

Tests the convert_camera_intrinsics method from RequestConverter:
- OpenCV pinhole camera model extraction
- Fisheye camera model approximation
- F-theta camera model handling
- Intrinsic matrix structure validation
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Import the module under test
try:
    from dggt_server.request_converter import RequestConverter
    from dggt_server.exceptions import InvalidCameraSpecError
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.request_converter import RequestConverter
    from dggt_server.exceptions import InvalidCameraSpecError

# Import fixtures helper
from tests.conftest import assert_valid_intrinsic_matrix


@pytest.mark.unit
class TestCameraIntrinsics:
    """Tests for RequestConverter.convert_camera_intrinsics()"""

    @pytest.fixture
    def mock_pinhole_param(self):
        """Create mock OpenCV pinhole camera parameters"""
        param = Mock()
        param.focal_length_x = 1000.0
        param.focal_length_y = 1000.0
        param.principal_point_x = 518.0
        param.principal_point_y = 350.0
        return param

    @pytest.fixture
    def mock_fisheye_param(self):
        """Create mock fisheye camera parameters"""
        param = Mock()
        param.focal_length_x = 1200.0
        param.focal_length_y = 1200.0
        param.principal_point_x = 600.0
        param.principal_point_y = 400.0
        param.max_angle = 1.57  # ~90 degrees
        return param

    @pytest.fixture
    def mock_ftheta_param(self):
        """Create mock f-theta camera parameters"""
        param = Mock()
        param.principal_point_x = 960.0
        param.principal_point_y = 540.0
        param.max_angle = 2.09  # ~120 degrees
        param.pixeldist_to_angle_poly = [0.0, 1.0/960.0]
        return param

    @pytest.fixture
    def mock_camera_spec_pinhole(self, mock_pinhole_param):
        """Create mock CameraSpec with pinhole parameters"""
        spec = Mock()
        spec.resolution_w = 1036
        spec.resolution_h = 700
        spec._pinhole_param = mock_pinhole_param
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            if field_name == "opencv_pinhole_param":
                return spec._pinhole_param is not None
            elif field_name == "opencv_fisheye_param":
                return spec._fisheye_param is not None
            elif field_name == "ftheta_param":
                return spec._ftheta_param is not None
            return False

        spec.HasField = has_field
        spec.opencv_pinhole_param = mock_pinhole_param
        return spec

    @pytest.fixture
    def mock_camera_spec_fisheye(self, mock_fisheye_param):
        """Create mock CameraSpec with fisheye parameters"""
        spec = Mock()
        spec.resolution_w = 1200
        spec.resolution_h = 800
        spec._pinhole_param = None
        spec._fisheye_param = mock_fisheye_param
        spec._ftheta_param = None

        def has_field(field_name):
            if field_name == "opencv_pinhole_param":
                return spec._pinhole_param is not None
            elif field_name == "opencv_fisheye_param":
                return spec._fisheye_param is not None
            elif field_name == "ftheta_param":
                return spec._ftheta_param is not None
            return False

        spec.HasField = has_field
        spec.opencv_fisheye_param = mock_fisheye_param
        return spec

    @pytest.fixture
    def mock_camera_spec_ftheta(self, mock_ftheta_param):
        """Create mock CameraSpec with f-theta parameters"""
        spec = Mock()
        spec.resolution_w = 1920
        spec.resolution_h = 1080
        spec._pinhole_param = None
        spec._fisheye_param = None
        spec._ftheta_param = mock_ftheta_param

        def has_field(field_name):
            if field_name == "opencv_pinhole_param":
                return spec._pinhole_param is not None
            elif field_name == "opencv_fisheye_param":
                return spec._fisheye_param is not None
            elif field_name == "ftheta_param":
                return spec._ftheta_param is not None
            return False

        spec.HasField = has_field
        spec.ftheta_param = mock_ftheta_param
        return spec

    @pytest.fixture
    def mock_camera_spec_empty(self):
        """Create mock CameraSpec with no camera model"""
        spec = Mock()
        spec.resolution_w = 800
        spec.resolution_h = 600
        spec._pinhole_param = None
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            return False

        spec.HasField = has_field
        return spec

    def test_pinhole_intrinsic_extraction(self, mock_camera_spec_pinhole):
        """Test extraction of K matrix from opencv_pinhole_param"""
        K, width, height = RequestConverter.convert_camera_intrinsics(mock_camera_spec_pinhole)

        assert width == 1036
        assert height == 700
        assert K.shape == (3, 3)
        assert K.dtype == np.float32

        # Verify intrinsic matrix structure
        assert K[0, 0] == 1000.0  # fx
        assert K[1, 1] == 1000.0  # fy
        assert K[0, 2] == 518.0   # cx (principal_point_x)
        assert K[1, 2] == 350.0   # cy (principal_point_y)
        assert K[2, 2] == 1.0

    def test_pinhole_different_focal_lengths(self, mock_pinhole_param):
        """Test handling of different fx and fy values"""
        mock_pinhole_param.focal_length_x = 800.0
        mock_pinhole_param.focal_length_y = 900.0

        spec = Mock()
        spec.resolution_w = 800
        spec.resolution_h = 600
        spec._pinhole_param = mock_pinhole_param
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            return field_name == "opencv_pinhole_param"

        spec.HasField = has_field
        spec.opencv_pinhole_param = mock_pinhole_param

        K, _, _ = RequestConverter.convert_camera_intrinsics(spec)

        assert K[0, 0] == 800.0
        assert K[1, 1] == 900.0

    def test_fisheye_intrinsic_approximation(self, mock_camera_spec_fisheye):
        """Test fisheye camera param is approximated as pinhole"""
        K, width, height = RequestConverter.convert_camera_intrinsics(mock_camera_spec_fisheye)

        assert width == 1200
        assert height == 800
        assert K[0, 0] == 1200.0
        assert K[1, 1] == 1200.0
        assert K[0, 2] == 600.0
        assert K[1, 2] == 400.0

    def test_ftheta_intrinsic_estimation(self, mock_camera_spec_ftheta):
        """Test f-theta camera intrinsic estimation"""
        K, width, height = RequestConverter.convert_camera_intrinsics(mock_camera_spec_ftheta)

        assert width == 1920
        assert height == 1080
        assert K[0, 2] == 960.0  # cx
        assert K[1, 2] == 540.0  # cy
        # Focal length is estimated from max_angle and width
        assert K[0, 0] > 0  # Should have positive focal length

    def test_no_camera_model_raises_error(self, mock_camera_spec_empty):
        """Test that missing camera model raises InvalidCameraSpecError"""
        with pytest.raises(InvalidCameraSpecError):
            RequestConverter.convert_camera_intrinsics(mock_camera_spec_empty)

    def test_output_dtype_is_float32(self, mock_camera_spec_pinhole):
        """Test that K matrix has float32 dtype"""
        K, _, _ = RequestConverter.convert_camera_intrinsics(mock_camera_spec_pinhole)
        assert K.dtype == np.float32

    def test_intrinsic_matrix_structure(self, mock_camera_spec_pinhole):
        """Test that intrinsic matrix has correct structure"""
        K, _, _ = RequestConverter.convert_camera_intrinsics(mock_camera_spec_pinhole)

        # Verify structure: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        assert K[0, 1] == 0.0  # No skew
        assert K[1, 0] == 0.0
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0
        assert K[2, 2] == 1.0

    def test_principal_point_at_center(self):
        """Test that principal point can be at image center"""
        param = Mock()
        param.focal_length_x = 500.0
        param.focal_length_y = 500.0
        param.principal_point_x = 400.0  # width/2
        param.principal_point_y = 300.0  # height/2

        spec = Mock()
        spec.resolution_w = 800
        spec.resolution_h = 600
        spec._pinhole_param = param
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            return field_name == "opencv_pinhole_param"

        spec.HasField = has_field
        spec.opencv_pinhole_param = param

        K, width, height = RequestConverter.convert_camera_intrinsics(spec)

        assert K[0, 2] == width / 2
        assert K[1, 2] == height / 2

    def test_arbitrary_principal_point(self):
        """Test that principal point can be at arbitrary location"""
        param = Mock()
        param.focal_length_x = 1000.0
        param.focal_length_y = 1000.0
        param.principal_point_x = 200.0  # Not at center
        param.principal_point_y = 150.0

        spec = Mock()
        spec.resolution_w = 800
        spec.resolution_h = 600
        spec._pinhole_param = param
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            return field_name == "opencv_pinhole_param"

        spec.HasField = has_field
        spec.opencv_pinhole_param = param

        K, _, _ = RequestConverter.convert_camera_intrinsics(spec)

        assert K[0, 2] == 200.0
        assert K[1, 2] == 150.0

    def test_positive_focal_lengths_required(self, mock_camera_spec_pinhole):
        """Test that focal lengths are positive"""
        K, _, _ = RequestConverter.convert_camera_intrinsics(mock_camera_spec_pinhole)

        assert K[0, 0] > 0  # fx positive
        assert K[1, 1] > 0  # fy positive


@pytest.mark.unit
class TestCameraIntrinsicsEdgeCases:
    """Edge case tests for camera intrinsics"""

    def test_very_small_focal_length(self):
        """Test handling of very small focal length"""
        param = Mock()
        param.focal_length_x = 10.0
        param.focal_length_y = 10.0
        param.principal_point_x = 50.0
        param.principal_point_y = 50.0

        spec = Mock()
        spec.resolution_w = 100
        spec.resolution_h = 100
        spec._pinhole_param = param
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            return field_name == "opencv_pinhole_param"

        spec.HasField = has_field
        spec.opencv_pinhole_param = param

        K, _, _ = RequestConverter.convert_camera_intrinsics(spec)

        assert K[0, 0] == 10.0
        assert K[1, 1] == 10.0

    def test_very_large_focal_length(self):
        """Test handling of very large focal length"""
        param = Mock()
        param.focal_length_x = 10000.0
        param.focal_length_y = 10000.0
        param.principal_point_x = 1920.0
        param.principal_point_y = 1080.0

        spec = Mock()
        spec.resolution_w = 3840
        spec.resolution_h = 2160
        spec._pinhole_param = param
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            return field_name == "opencv_pinhole_param"

        spec.HasField = has_field
        spec.opencv_pinhole_param = param

        K, _, _ = RequestConverter.convert_camera_intrinsics(spec)

        assert K[0, 0] == 10000.0
        assert K[1, 1] == 10000.0

    def test_ftheta_zero_max_angle(self):
        """Test f-theta with zero max_angle (edge case)"""
        param = Mock()
        param.principal_point_x = 960.0
        param.principal_point_y = 540.0
        param.max_angle = 0.0
        param.pixeldist_to_angle_poly = []

        spec = Mock()
        spec.resolution_w = 1920
        spec.resolution_h = 1080
        spec._pinhole_param = None
        spec._fisheye_param = None
        spec._ftheta_param = param

        def has_field(field_name):
            return field_name == "ftheta_param"

        spec.HasField = has_field
        spec.ftheta_param = param

        K, _, _ = RequestConverter.convert_camera_intrinsics(spec)

        # Should still produce valid K matrix
        assert K.shape == (3, 3)
        assert K[0, 2] == 960.0

    def test_resolution_values(self):
        """Test that resolution values are correctly extracted"""
        param = Mock()
        param.focal_length_x = 500.0
        param.focal_length_y = 500.0
        param.principal_point_x = 256.0
        param.principal_point_y = 256.0

        spec = Mock()
        spec.resolution_w = 512
        spec.resolution_h = 512
        spec._pinhole_param = param
        spec._fisheye_param = None
        spec._ftheta_param = None

        def has_field(field_name):
            return field_name == "opencv_pinhole_param"

        spec.HasField = has_field
        spec.opencv_pinhole_param = param

        K, width, height = RequestConverter.convert_camera_intrinsics(spec)

        assert width == 512
        assert height == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])