# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Frame Index Mapper

Maps timestamps to frame indices for DGGT scenes.
Handles time ranges and batch frame index computation.
"""

import logging
from typing import List, Tuple, Optional

from .scene_metadata import DGGTSceneMetadata

logger = logging.getLogger(__name__)


class FrameIndexMapper:
    """
    Frame Index Mapper

    Maps timestamps to frame indices using scene metadata.
    Supports single frame lookup and batch range computation.

    Strategy:
    - Single frame: use midpoint of start/end timestamps
    - Batch range: compute all frames within duration
    """

    def __init__(self, scene_metadata: DGGTSceneMetadata):
        """
        Initialize frame index mapper

        Args:
            scene_metadata: DGGT scene metadata from Phase 1
        """
        self.metadata = scene_metadata

    def get_frame_index(
        self,
        frame_start_us: Optional[int] = None,
        frame_end_us: Optional[int] = None,
        timestamp_us: Optional[int] = None
    ) -> int:
        """
        Compute frame index from timestamps

        Three modes:
        1. timestamp_us provided: direct lookup
        2. frame_start_us and frame_end_us provided: use midpoint
        3. None provided: return frame 0

        Args:
            frame_start_us: Frame start timestamp in microseconds
            frame_end_us: Frame end timestamp in microseconds
            timestamp_us: Direct timestamp in microseconds (overrides start/end)

        Returns:
            Frame index (clamped to valid range)
        """
        # Mode 1: Direct timestamp
        if timestamp_us is not None:
            return self.metadata.get_frame_index_from_timestamp(timestamp_us)

        # Mode 2: Start/end range - use midpoint
        if frame_start_us is not None and frame_end_us is not None:
            midpoint_us = (frame_start_us + frame_end_us) // 2
            return self.metadata.get_frame_index_from_timestamp(midpoint_us)

        # Mode 3: No timestamp provided - return first frame
        logger.debug("No timestamp provided, returning frame 0")
        return 0

    def get_frame_range(
        self,
        duration_us: int,
        start_timestamp_us: Optional[int] = None
    ) -> List[int]:
        """
        Compute batch frame indices for a duration

        Returns all frames that fall within the time window.

        Args:
            duration_us: Duration in microseconds
            start_timestamp_us: Start timestamp (defaults to scene start)

        Returns:
            List of frame indices within the time window
        """
        # Default to scene start
        if start_timestamp_us is None:
            start_timestamp_us = self.metadata.start_timestamp_us

        end_timestamp_us = start_timestamp_us + duration_us

        # Clamp to valid range
        if start_timestamp_us < self.metadata.start_timestamp_us:
            start_timestamp_us = self.metadata.start_timestamp_us
            logger.warning(
                f"Start timestamp clamped to scene start: {self.metadata.start_timestamp_us}"
            )

        if end_timestamp_us > self.metadata.end_timestamp_us:
            end_timestamp_us = self.metadata.end_timestamp_us
            logger.warning(
                f"End timestamp clamped to scene end: {self.metadata.end_timestamp_us}"
            )

        # Compute frame indices
        start_frame = self.metadata.get_frame_index_from_timestamp(start_timestamp_us)
        end_frame = self.metadata.get_frame_index_from_timestamp(end_timestamp_us)

        # Include all frames in range
        frame_indices = list(range(start_frame, end_frame + 1))

        logger.debug(
            f"Frame range for duration {duration_us}us: "
            f"frames {start_frame} to {end_frame} ({len(frame_indices)} frames)"
        )

        return frame_indices

    def get_timestamp_range(self, frame_indices: List[int]) -> Tuple[int, int]:
        """
        Compute timestamp range from frame indices

        Args:
            frame_indices: List of frame indices

        Returns:
            Tuple of (start_timestamp_us, end_timestamp_us)
        """
        if not frame_indices:
            return (self.metadata.start_timestamp_us, self.metadata.start_timestamp_us)

        # Clamp indices to valid range
        valid_indices = [
            max(0, min(idx, self.metadata.num_frames - 1))
            for idx in frame_indices
        ]

        start_ts = self.metadata.get_timestamp_from_frame_index(min(valid_indices))
        end_ts = self.metadata.get_timestamp_from_frame_index(max(valid_indices))

        return (start_ts, end_ts)

    def validate_timestamp(self, timestamp_us: int) -> Tuple[bool, int]:
        """
        Validate and potentially clamp a timestamp

        Args:
            timestamp_us: Timestamp to validate

        Returns:
            Tuple of (is_valid, clamped_timestamp)
        """
        clamped = timestamp_us

        if timestamp_us < self.metadata.start_timestamp_us:
            clamped = self.metadata.start_timestamp_us
            logger.warning(
                f"Timestamp {timestamp_us} clamped to start: {clamped}"
            )
            return (False, clamped)

        if timestamp_us > self.metadata.end_timestamp_us:
            clamped = self.metadata.end_timestamp_us
            logger.warning(
                f"Timestamp {timestamp_us} clamped to end: {clamped}"
            )
            return (False, clamped)

        return (True, clamped)
