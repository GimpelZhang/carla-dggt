# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Track System Module

Provides classes for pose interpolation and track management for DGGT frame-based data.

Key differences from NuRec (track.py):
- Frame indices instead of microsecond timestamps
- Lazy loading of poses from JSON files (ego_pose/, dynamic_objects/)
- Timestamp conversion: timestamp_us = frame_idx * 100000 (10Hz)
- Pose caching for performance

Classes:
- DggtInterpolatedPoses: Base class for pose interpolation with lazy loading
- DggtTrack: Single object trajectory with metadata (extends DggtInterpolatedPoses)

Coordinate Transform Note:
This module handles pose data in DGGT world coordinates.
Coordinate transforms to CARLA are applied externally by DggtScenario.
"""

from scipy.spatial.transform import Rotation
from typing import List, Union, Optional, Tuple, Dict
import numpy as np
import json
import os
import logging

# Reuse constants from NuRec
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    DYNAMIC_FLAG,
    CONTROLLABLE_FLAG,
    EGO_FLAG,
    SPECTATOR_FLAG,
    EGO_TRACK_ID,
)

logger = logging.getLogger(__name__)


class DggtInterpolatedPoses:
    """
    Interpolated poses for DGGT frame-based data.

    Adapted from NuRec InterpolatedPoses (track.py:309-607).

    Key adaptations:
    - Uses frame indices instead of microsecond timestamps
    - Lazy loading of poses from JSON files
    - Internal conversion: timestamp_us = frame_idx * 100000 (10Hz default)

    Attributes:
        timestamps: List of timestamps in microseconds (derived from frame indices)
        _scene_path: Path to DGGT scene directory
        _start_frame: Starting frame index (0-based)
        _end_frame: Ending frame index (inclusive)
        _fps: Frame rate (default 10Hz for Waymo)
        _object_id: Object ID for dynamic object poses (None for ego)
        _is_ego: True for ego vehicle (loads from ego_pose/)
        _pose_cache: Cache of loaded poses by frame index
        transform: 4x4 transform matrix to apply to all poses
        ignore_out_of_bounds: Whether to clamp out-of-bounds timestamps
    """

    def __init__(
        self,
        scene_path: str,
        frame_range: Tuple[int, int],  # (start_frame, end_frame) inclusive
        fps: float = 10.0,  # DGGT default 10Hz
        object_id: Optional[int] = None,  # For dynamic objects
        is_ego: bool = False,  # For ego track
    ):
        """
        Initialize interpolated poses.

        Args:
            scene_path: Path to DGGT scene directory (e.g., scene/0328/001/)
            frame_range: (start_frame, end_frame) inclusive range (0-based indices)
            fps: Frame rate (default 10Hz for Waymo)
            object_id: Object ID for dynamic object poses (None for ego)
            is_ego: True for ego vehicle (loads from ego_pose/)
        """
        self._scene_path = scene_path
        self._start_frame = frame_range[0]
        self._end_frame = frame_range[1]
        self._fps = fps
        self._object_id = object_id
        self._is_ego = is_ego

        # Generate timestamps from frame indices
        # timestamp_us = frame_idx * 1_000_000 / fps
        # For 10Hz: timestamp_us = frame_idx * 100_000
        us_per_frame = int(1_000_000 / fps)
        self.timestamps = [
            frame_idx * us_per_frame
            for frame_idx in range(self._start_frame, self._end_frame + 1)
        ]

        # Lazy loading cache: frame_idx -> pose_matrix (4x4)
        self._pose_cache: Dict[int, np.ndarray] = {}
        self._poses: List[np.ndarray] = []  # Loaded poses (populated on demand)

        # Transform application (same as NuRec)
        self.ignore_out_of_bounds = False
        self.transform = np.eye(4)
        self.post_transform = np.eye(4)

    def _load_pose_for_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load pose from JSON file for given frame.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            4x4 pose matrix in DGGT world coordinates, or None if not found
        """
        if frame_idx in self._pose_cache:
            return self._pose_cache[frame_idx]

        try:
            if self._is_ego:
                # Load from ego_pose/frame_XXXX_ego.json
                pose_file = os.path.join(
                    self._scene_path,
                    "ego_pose",
                    f"frame_{frame_idx:04d}_ego.json"
                )
                with open(pose_file, "r") as f:
                    data = json.load(f)
                # Use camera_extrinsics_world as the pose
                pose = np.array(data["camera_extrinsics_world"], dtype=np.float64)
            else:
                # Load from dynamic_objects/frame_XXXX_objects.json
                objects_file = os.path.join(
                    self._scene_path,
                    "dynamic_objects",
                    f"frame_{frame_idx:04d}_objects.json"
                )
                with open(objects_file, "r") as f:
                    objects = json.load(f)

                # Find object by ID
                pose = None
                for obj in objects:
                    if obj.get("object_id") == self._object_id:
                        pose = np.array(obj["pose_world"], dtype=np.float64)
                        break

                if pose is None:
                    logger.warning(
                        f"Object {self._object_id} not found in frame {frame_idx}"
                    )
                    return None

            # Validate pose shape
            if pose.shape != (4, 4):
                logger.error(
                    f"Invalid pose shape {pose.shape} for frame {frame_idx}"
                )
                return None

            # Cache and return
            self._pose_cache[frame_idx] = pose
            return pose

        except FileNotFoundError:
            logger.warning(f"Pose file not found for frame {frame_idx}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for frame {frame_idx}: {e}")
            return None

    def _ensure_poses_loaded(self) -> None:
        """Load all poses if not already loaded."""
        if len(self._poses) == len(self.timestamps):
            return

        self._poses = []
        for frame_idx in range(self._start_frame, self._end_frame + 1):
            pose = self._load_pose_for_frame(frame_idx)
            if pose is not None:
                self._poses.append(pose)
            else:
                # Use last valid pose as fallback, or identity if first frame
                if len(self._poses) > 0:
                    logger.warning(
                        f"Using fallback pose for frame {frame_idx}"
                    )
                    self._poses.append(self._poses[-1])
                else:
                    logger.warning(
                        f"Using identity pose for frame {frame_idx} (no prior pose)"
                    )
                    self._poses.append(np.eye(4))

    def _get_interpolation_params(
        self, timestamp: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Determine interpolation parameters for a given timestamp.

        Args:
            timestamp: Timestamp in microseconds

        Returns:
            Tuple: (prev_pose, next_pose, t_factor)
            Returns (None, None, 0.0) if timestamp is out of range
        """
        self._ensure_poses_loaded()

        # Check if timestamp is within track's time range
        if self.ignore_out_of_bounds and timestamp > self.timestamps[-1]:
            # Clamp to last pose
            return self._poses[-2], self._poses[-1], 1.0
        elif timestamp < self.timestamps[0] or timestamp > self.timestamps[-1]:
            logger.error(
                f"Timestamp {timestamp} is outside track's time range "
                f"[{self.timestamps[0]}, {self.timestamps[-1]}]"
            )
            return None, None, 0.0

        # Find interpolation bracket
        prev_idx = 0
        for i in range(len(self.timestamps)):
            if self.timestamps[i] > timestamp:
                break
            prev_idx = i

        next_idx = prev_idx + 1

        # If at last timestamp, return last pose
        if next_idx >= len(self.timestamps):
            return self._poses[prev_idx], None, 0.0

        # Calculate interpolation factor (t) between 0 and 1
        t = (timestamp - self.timestamps[prev_idx]) / (
            self.timestamps[next_idx] - self.timestamps[prev_idx]
        )

        return self._poses[prev_idx], self._poses[next_idx], t

    def interpolate_pose_matrix(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Interpolate pose at given timestamp and return as 4x4 matrix.

        Uses SLERP for rotation interpolation and linear interpolation for translation.

        Args:
            timestamp: Timestamp in microseconds

        Returns:
            4x4 transformation matrix, or None if timestamp is out of range
        """
        start_pose, end_pose, t = self._get_interpolation_params(timestamp)

        if start_pose is None:
            logger.warning(
                f"Start pose is None for timestamp {timestamp}, likely out of bounds"
            )
            return None

        if end_pose is None:
            # At last timestamp, return start pose
            result = start_pose.copy()
            result = self.transform @ result @ self.post_transform
            return result

        # Extract rotation matrices and translations
        start_rotation_matrix = start_pose[:3, :3]
        start_translation = start_pose[:3, 3]

        end_rotation_matrix = end_pose[:3, :3]
        end_translation = end_pose[:3, 3]

        # Linear interpolation for translation
        interp_translation = start_translation + t * (
            end_translation - start_translation
        )

        # SLERP for rotation (same algorithm as NuRec)
        start_rotation = Rotation.from_matrix(start_rotation_matrix)
        end_rotation = Rotation.from_matrix(end_rotation_matrix)

        # Compute relative rotation in rotation vector form
        rotvec = (start_rotation.inv() * end_rotation).as_rotvec()
        interp_rotation = start_rotation * Rotation.from_rotvec(rotvec * t)

        # Build result matrix
        result = np.eye(4)
        result[:3, :3] = interp_rotation.as_matrix()
        result[:3, 3] = interp_translation

        # Apply external transform
        result = self.transform @ result @ self.post_transform

        return result

    def interpolate_pose_xyzquat(self, timestamp: float) -> Optional[List[float]]:
        """
        Interpolate pose at given timestamp and return as xyz+quaternion format.

        Args:
            timestamp: Timestamp in microseconds

        Returns:
            List [x, y, z, qx, qy, qz, qw], or None if timestamp is out of range
        """
        matrix = self.interpolate_pose_matrix(timestamp)
        if matrix is None:
            return None

        translation = matrix[:3, 3]
        rotation = Rotation.from_matrix(matrix[:3, :3])
        quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]

        return np.concatenate([translation, quaternion]).tolist()

    def interpolate_pose_euler(self, timestamp: float) -> Optional[List[float]]:
        """
        Interpolate pose at given timestamp and return as xyz+euler format.

        Args:
            timestamp: Timestamp in microseconds

        Returns:
            List [x, y, z, roll, pitch, yaw] (radians), or None if out of range
        """
        matrix = self.interpolate_pose_matrix(timestamp)
        if matrix is None:
            return None

        translation = matrix[:3, 3]
        rotation = Rotation.from_matrix(matrix[:3, :3])
        euler_angles = rotation.as_euler("zyx")

        return np.concatenate([translation, euler_angles]).tolist()

    def interpolate_pose(self, timestamp: float) -> Optional[List[float]]:
        """
        Default interpolation method returning xyz+quaternion format.

        Args:
            timestamp: Timestamp in microseconds

        Returns:
            List [x, y, z, qx, qy, qz, qw], or None if out of range
        """
        return self.interpolate_pose_xyzquat(timestamp)

    def set_ignore_out_of_bounds(self, ignore: bool) -> None:
        """
        Set whether to ignore timestamps outside the available range.

        Args:
            ignore: If True, clamp out-of-bounds timestamps to last pose
        """
        self.ignore_out_of_bounds = ignore

    def set_transform(self, transform: np.ndarray) -> None:
        """
        Set transformation matrix to apply to all interpolated poses.

        Args:
            transform: 4x4 transformation matrix
        """
        if transform.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4, got shape {transform.shape}")
        self.transform = transform

    def set_post_transform(self, transform: np.ndarray) -> None:
        """
        Set post-transformation matrix to apply to all interpolated poses.

        Args:
            transform: 4x4 transformation matrix
        """
        if transform.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4, got shape {transform.shape}")
        self.post_transform = transform

    def get_path(
        self,
        spacing_us: float = 100_000,
        start_time: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Generate path waypoints at regular intervals.

        Args:
            spacing_us: Spacing between points in microseconds (default 100ms)
            start_time: Start timestamp in microseconds (default: first timestamp)

        Returns:
            List of 4x4 pose matrices
        """
        self._ensure_poses_loaded()

        if start_time is None:
            start_time = self.timestamps[0]

        path = []
        for timestamp in range(
            int(start_time),
            int(self.timestamps[-1]) + 1,
            int(spacing_us)
        ):
            mat = self.interpolate_pose_matrix(timestamp)
            if mat is not None:
                path.append(mat)

        return path

    def get_frame_range(self) -> Tuple[int, int]:
        """Get frame range as (start_frame, end_frame)."""
        return (self._start_frame, self._end_frame)


class DggtTrack(DggtInterpolatedPoses):
    """
    Single object trajectory with metadata.

    Adapted from NuRec Track (track.py:609-693).

    Attributes:
        track_id: Unique identifier (e.g., "dggt_obj_0001" or EGO_TRACK_ID)
        object_id: DGGT integer object ID (None for ego)
        dims: Dimensions [length, width, height] in meters
        label: Class label (e.g., "automobile", "person", "unknown")
        dynamic: Whether object is dynamic (moving)
        controllable: Whether object is controllable
        ego: Whether this is the ego vehicle track
        spectator: Whether this is a spectator/camera track
    """

    def __init__(
        self,
        track_id: str,
        scene_path: str,
        frame_range: Tuple[int, int],
        dims: Optional[List[float]],
        label: str,
        flags: List[str],
        fps: float = 10.0,
        object_id: Optional[int] = None,
        is_ego: bool = False,
    ):
        """
        Initialize a track with frame range and metadata.

        Args:
            track_id: Unique identifier (e.g., "dggt_obj_0001")
            scene_path: Path to DGGT scene directory
            frame_range: (start_frame, end_frame) inclusive range
            dims: Dimensions [length, width, height] (None if unknown)
            label: Class label (e.g., "automobile", "person")
            flags: List of flags (DYNAMIC, CONTROLLABLE, EGO, SPECTATOR)
            fps: Frame rate (default 10Hz)
            object_id: DGGT object ID for pose loading (None for ego)
            is_ego: True for ego vehicle track
        """
        super().__init__(scene_path, frame_range, fps, object_id, is_ego)

        self.track_id = track_id
        self.dims = dims if dims is not None else [0.0, 0.0, 0.0]
        self.label = label

        # Parse flags
        self.dynamic = DYNAMIC_FLAG in flags
        self.controllable = CONTROLLABLE_FLAG in flags
        self.ego = EGO_FLAG in flags
        self.spectator = SPECTATOR_FLAG in flags

    def start_time(self) -> float:
        """Get start time in microseconds."""
        return self.timestamps[0]

    def end_time(self) -> float:
        """Get end time in microseconds."""
        return self.timestamps[-1]

    def start_frame(self) -> int:
        """Get start frame index (0-based)."""
        return self._start_frame

    def end_frame(self) -> int:
        """Get end frame index (0-based)."""
        return self._end_frame

    def is_dynamic(self) -> bool:
        """Check if track is dynamic (moving object)."""
        return self.dynamic

    def is_controllable(self) -> bool:
        """Check if track is controllable."""
        return self.controllable

    def is_ego(self) -> bool:
        """Check if this is the ego vehicle track."""
        return self.ego

    def is_spectator(self) -> bool:
        """Check if this is a spectator/camera track."""
        return self.spectator

    def get_class(self) -> str:
        """Get class label."""
        return self.label

    def get_lifetime_frames(self) -> int:
        """Get lifetime in number of frames."""
        return self._end_frame - self._start_frame + 1

    def get_lifetime_seconds(self) -> float:
        """Get lifetime in seconds."""
        return self.get_lifetime_frames() / self._fps

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DggtTrack(track_id='{self.track_id}', "
            f"frames=[{self._start_frame}, {self._end_frame}], "
            f"label='{self.label}', dynamic={self.dynamic})"
        )


def build_ego_track(
    scene_path: str,
    num_frames: int,
    fps: float = 10.0,
) -> DggtTrack:
    """
    Build ego vehicle track from scene's ego_pose data.

    Convenience function to create a DggtTrack for the ego vehicle.

    Args:
        scene_path: Path to DGGT scene directory
        num_frames: Number of frames in the scene
        fps: Frame rate (default 10Hz)

    Returns:
        DggtTrack for ego vehicle
    """
    return DggtTrack(
        track_id=EGO_TRACK_ID,
        scene_path=scene_path,
        frame_range=(0, num_frames - 1),
        dims=None,  # Ego dims are set by blueprint
        label="ego",
        flags=[EGO_FLAG, DYNAMIC_FLAG],
        fps=fps,
        object_id=None,
        is_ego=True,
    )


class DggtTracks:
    """
    Manages collection of DggtTrack objects with lifecycle management.

    Adapted from NuRec Tracks (scenario.py:128-226).

    Key differences:
    - Uses frame indices internally, converts to timestamps for interpolation
    - Track discovery via frame scanning (DggtTrackIndex)
    - Time stepping by frame_step (number of frames to advance)

    Attributes:
        track_data: Sorted list of DggtTrack objects
        start_frame: Starting frame index for replay
        fps: Frame rate (default 10Hz)
        zero_time: Reference start time in microseconds
        current_time: Current playback time in microseconds
        active_tracks: Currently active track objects
        track_index: Index in track_data for next activation check
        min_lifetime: Minimum lifetime filter in microseconds
    """

    def __init__(
        self,
        track_data: List[DggtTrack],
        start_frame: int,
        fps: float = 10.0,
    ):
        """
        Initialize tracks collection.

        Args:
            track_data: List of DggtTrack objects
            start_frame: Starting frame index for replay (0-based)
            fps: Frame rate (default 10Hz)
        """
        # Sort by start_time (same as NuRec)
        track_data.sort(key=lambda x: x.start_time())

        self.track_data = track_data
        self.start_frame = start_frame
        self.fps = fps

        # Current time tracking
        us_per_frame = int(1_000_000 / fps)
        self.zero_time = start_frame * us_per_frame
        self.current_time = self.zero_time

        # Active tracks
        self.active_tracks: List[DggtTrack] = []
        self.track_index = 0

        # Minimum lifetime filter (default: 1 frame)
        self.min_lifetime = us_per_frame

    def reset(self) -> None:
        """Reset to initial state."""
        self.current_time = self.zero_time
        self.active_tracks = []
        self.track_index = 0

    def update(self, frame_step: int = 1) -> Tuple[List[DggtTrack], List[DggtTrack]]:
        """
        Update tracks based on frame step.

        Args:
            frame_step: Number of frames to advance (default 1)

        Returns:
            Tuple of (new_tracks, tracks_to_remove)
        """
        us_per_frame = int(1_000_000 / self.fps)
        time_step = frame_step * us_per_frame
        self.current_time += time_step

        # Remove expired tracks
        tracks_to_remove = []
        for track in self.active_tracks:
            if self.current_time > track.end_time():
                tracks_to_remove.append(track)

        for track in tracks_to_remove:
            self.active_tracks.remove(track)

        # Add newly active tracks
        new_tracks = []
        while (
            self.track_index < len(self.track_data)
            and self.track_data[self.track_index].start_time() <= self.current_time
        ):
            next_track = self.track_data[self.track_index]
            lifetime = next_track.end_time() - next_track.start_time()

            if lifetime > self.min_lifetime:
                self.active_tracks.append(next_track)
                new_tracks.append(next_track)

            self.track_index += 1

        return new_tracks, tracks_to_remove

    def get_current_frame(self) -> int:
        """Get current frame index."""
        us_per_frame = int(1_000_000 / self.fps)
        return int(self.current_time / us_per_frame)

    def get_current_time_seconds(self) -> float:
        """Get current time in seconds relative to zero_time."""
        return (self.current_time - self.zero_time) / 1_000_000

    def set_minimum_lifetime_frames(self, min_frames: int) -> None:
        """Set minimum lifetime in frames."""
        us_per_frame = int(1_000_000 / self.fps)
        self.min_lifetime = min_frames * us_per_frame

    def set_minimum_lifetime_seconds(self, min_seconds: float) -> None:
        """Set minimum lifetime in seconds."""
        self.min_lifetime = int(min_seconds * 1_000_000)

    def get_all_possible_tracks(self) -> List[DggtTrack]:
        """Get all tracks longer than minimum lifetime."""
        return [
            track for track in self.track_data
            if track.end_time() - track.start_time() > self.min_lifetime
        ]

    def set_view_transform(self, transform: np.ndarray) -> None:
        """Apply transform to all tracks."""
        for track in self.track_data:
            track.set_transform(transform)

    def get_active_track_by_id(self, track_id: str) -> Optional[DggtTrack]:
        """Find active track by track_id."""
        for track in self.active_tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_track_by_id(self, track_id: str) -> Optional[DggtTrack]:
        """Find any track by track_id (searches all track_data)."""
        for track in self.track_data:
            if track.track_id == track_id:
                return track
        return None


class DggtTrackIndex:
    """
    Builds track continuity from DGGT per-frame object files.

    Algorithm:
    1. Scan all frame_XXXX_objects.json files
    2. For each object_id, record frames where it appears
    3. Identify continuous segments (gap < threshold)
    4. Create DggtTrack for each continuous segment

    Attributes:
        _scene_path: Path to DGGT scene directory
        _num_frames: Total number of frames to scan
        _fps: Frame rate (default 10Hz)
        _max_gap_frames: Maximum gap to consider continuous (default 5 frames)
        _object_presence: Map of object_id -> list of (frame, pose, dims, label)
    """

    def __init__(
        self,
        scene_path: str,
        num_frames: int,
        fps: float = 10.0,
        max_gap_frames: int = 5,
    ):
        """
        Initialize track index builder.

        Args:
            scene_path: Path to DGGT scene directory
            num_frames: Total number of frames in scene
            fps: Frame rate (default 10Hz)
            max_gap_frames: Maximum gap between appearances to consider continuous
        """
        self._scene_path = scene_path
        self._num_frames = num_frames
        self._fps = fps
        self._max_gap_frames = max_gap_frames

        # Object presence map: object_id -> list of (frame_idx, pose, dims, label)
        self._object_presence: Dict[int, List[Tuple[int, np.ndarray, List[float], str]]] = {}

        # Build index by scanning frames
        self._scan_frames()

    def _scan_frames(self) -> None:
        """Scan all frame files to build object presence map."""
        logger.info(f"Scanning {self._num_frames} frames for track discovery...")

        for frame_idx in range(self._num_frames):
            objects_file = os.path.join(
                self._scene_path,
                "dynamic_objects",
                f"frame_{frame_idx:04d}_objects.json"
            )

            if not os.path.exists(objects_file):
                logger.debug(f"Frame {frame_idx} objects file not found, skipping")
                continue

            try:
                with open(objects_file, "r") as f:
                    objects = json.load(f)

                for obj in objects:
                    obj_id = obj.get("object_id")
                    if obj_id is None:
                        logger.warning(f"Object missing object_id in frame {frame_idx}")
                        continue

                    pose = np.array(obj.get("pose_world", np.eye(4)), dtype=np.float64)
                    dims = obj.get("dimensions", [0.0, 0.0, 0.0])
                    # DGGT JSON may not have label field, use "unknown" as default
                    label = obj.get("label", "unknown")

                    if obj_id not in self._object_presence:
                        self._object_presence[obj_id] = []

                    self._object_presence[obj_id].append(
                        (frame_idx, pose, dims, label)
                    )

            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Error loading frame {frame_idx}: {e}")
                continue

        logger.info(f"Found {len(self._object_presence)} unique objects")

    def _find_continuous_segments(
        self,
        frames_data: List[Tuple[int, np.ndarray, List[float], str]]
    ) -> List[Tuple[int, int, str, List[float]]]:
        """
        Find continuous segments in frame data.

        Args:
            frames_data: List of (frame_idx, pose, dims, label)

        Returns:
            List of (start_frame, end_frame, label, avg_dims)
        """
        if len(frames_data) == 0:
            return []

        segments = []

        # Sort by frame index
        frames_data.sort(key=lambda x: x[0])

        current_start = frames_data[0][0]
        last_frame = frames_data[0][0]
        label = frames_data[0][3]

        # Track dimensions for averaging
        dims_list = [frames_data[0][2]]

        for frame_idx, pose, obj_dims, obj_label in frames_data[1:]:
            if frame_idx - last_frame <= self._max_gap_frames:
                # Continuous - extend segment
                last_frame = frame_idx
                dims_list.append(obj_dims)
                # Use most recent label if changed
                label = obj_label
            else:
                # Gap detected - end current segment, start new
                avg_dims = [
                    sum(d[i] for d in dims_list) / len(dims_list)
                    for i in range(3)
                ]
                segments.append((current_start, last_frame, label, avg_dims))

                # Start new segment
                current_start = frame_idx
                last_frame = frame_idx
                label = obj_label
                dims_list = [obj_dims]

        # Add final segment
        avg_dims = [
            sum(d[i] for d in dims_list) / len(dims_list)
            for i in range(3)
        ]
        segments.append((current_start, last_frame, label, avg_dims))

        return segments

    def get_tracks(self) -> List[DggtTrack]:
        """
        Generate DggtTrack objects from presence map.

        Returns:
            List of DggtTrack objects for all discovered objects
        """
        tracks = []

        for obj_id, frames_data in self._object_presence.items():
            segments = self._find_continuous_segments(frames_data)

            for seg_idx, (start_frame, end_frame, label, dims) in enumerate(segments):
                # Generate track_id
                if len(segments) > 1:
                    # Multiple segments for same object - add segment suffix
                    track_id = f"dggt_obj_{obj_id:04d}_seg{seg_idx}"
                else:
                    track_id = f"dggt_obj_{obj_id:04d}"

                # All discovered dynamic objects are DYNAMIC
                flags = [DYNAMIC_FLAG]

                track = DggtTrack(
                    track_id=track_id,
                    scene_path=self._scene_path,
                    frame_range=(start_frame, end_frame),
                    dims=dims,
                    label=label,
                    flags=flags,
                    fps=self._fps,
                    object_id=obj_id,
                    is_ego=False,
                )
                tracks.append(track)

        logger.info(f"Generated {len(tracks)} tracks from {len(self._object_presence)} objects")
        return tracks

    def get_object_ids(self) -> List[int]:
        """Get list of all discovered object IDs."""
        return list(self._object_presence.keys())

    def get_object_appearance_count(self, object_id: int) -> int:
        """Get number of frames where object appears."""
        return len(self._object_presence.get(object_id, []))


def scan_scene_for_tracks(
    scene_path: str,
    num_frames: int,
    fps: float = 10.0,
) -> Tuple[DggtTracks, DggtTrack]:
    """
    Convenience function to scan scene and build track collection.

    Creates both the DggtTracks collection for dynamic objects
    and the ego vehicle track.

    Args:
        scene_path: Path to DGGT scene directory
        num_frames: Total number of frames
        fps: Frame rate (default 10Hz)

    Returns:
        Tuple of (DggtTracks, ego_track)
    """
    # Build dynamic object tracks
    track_index = DggtTrackIndex(scene_path, num_frames, fps)
    dynamic_tracks = track_index.get_tracks()

    # Build ego track
    ego_track = build_ego_track(scene_path, num_frames, fps)

    # Combine into collection (ego track is handled separately)
    # Note: DggtTracks manages dynamic tracks, ego is added to actor_mapping directly
    tracks_collection = DggtTracks(dynamic_tracks, start_frame=0, fps=fps)

    return tracks_collection, ego_track