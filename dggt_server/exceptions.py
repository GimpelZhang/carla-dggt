# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Exception Classes

Custom exceptions for DGGT scene loading and management.
Uses hybrid error strategy: Critical errors raise, Warning/Info log and continue.
"""


class DGGTError(Exception):
    """DGGT base exception class"""
    pass


class SceneNotFoundError(DGGTError):
    """Scene directory not found"""
    def __init__(self, scene_id: str, path: str):
        self.scene_id = scene_id
        self.path = path
        super().__init__(f"Scene '{scene_id}' not found at path: {path}")


class MissingStaticSceneError(DGGTError):
    """Static scene PLY file missing"""
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        super().__init__(f"Static scene PLY file is missing for scene '{scene_id}'")


class EmptyEgoPoseError(DGGTError):
    """Ego pose directory is empty"""
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        super().__init__(f"No ego pose data found for scene '{scene_id}'")


class InvalidJSONError(DGGTError):
    """JSON parsing failed"""
    def __init__(self, file_path: str, original_error: Exception):
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(f"Failed to parse JSON file '{file_path}': {original_error}")


class FrameIndexOutOfRangeError(DGGTError):
    """Frame index out of range (auto-corrected)"""
    def __init__(self, requested_idx: int, valid_range: tuple, corrected_idx: int):
        self.requested_idx = requested_idx
        self.valid_range = valid_range
        self.corrected_idx = corrected_idx
        super().__init__(
            f"Frame index {requested_idx} out of range {valid_range}, "
            f"corrected to {corrected_idx}"
        )


class InvalidPoseError(DGGTError):
    """Invalid pose matrix provided"""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid pose: {reason}")


class InvalidCameraSpecError(DGGTError):
    """Invalid camera specification"""
    def __init__(self, param_name: str, value: any, reason: str):
        self.param_name = param_name
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid camera parameter '{param_name}'={value}: {reason}")


class TrackIDMappingError(DGGTError):
    """Track ID to object ID mapping failed"""
    def __init__(self, track_id: str, reason: str):
        self.track_id = track_id
        self.reason = reason
        super().__init__(f"Failed to map track_id '{track_id}': {reason}")


class RenderingError(DGGTError):
    """Rendering operation failed"""
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Rendering failed during '{operation}': {reason}")
