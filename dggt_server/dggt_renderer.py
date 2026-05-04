# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Renderer Module

Wraps gsplat rasterization for DGGT Gaussian Splatting scene rendering.

Key components:
- Load static scene PLY (static_scene.ply)
- Load sky PLY (sky_scene.ply) if exists
- Load dynamic object PLY per frame (frame_{t:04d}_dynamic.ply)
- Apply object pose transformations (local -> world)
- Use gsplat.rasterization for rendering

Reference: DGGT_Engine.md, /home/junchuan/e2e/dggt/dggt_engine.py
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

# gsplat rasterization
from gsplat.rendering import rasterization

logger = logging.getLogger(__name__)

# Coordinate transform utility (for service layer integration)
# CRITICAL: Import undo_carla_coordinate_transform from utils.py, never reimplement!
# Note: This import may fail in environments without carla module (e.g., dggt env)
# The function is used by the service layer, not directly by this renderer.
# When integrating with CARLA, ensure the service imports this from utils.py.
_undo_carla_coordinate_transform = None


def _get_undo_carla_coordinate_transform():
    """
    Lazy import of undo_carla_coordinate_transform from utils.py

    This deferred import avoids module load failures in environments
    that don't have the carla package (like the dggt conda environment).

    Returns:
        undo_carla_coordinate_transform function from utils.py

    Raises:
        ImportError: If utils.py cannot be imported (missing carla dependency)
    """
    global _undo_carla_coordinate_transform
    if _undo_carla_coordinate_transform is None:
        try:
            from ..utils import undo_carla_coordinate_transform
            _undo_carla_coordinate_transform = undo_carla_coordinate_transform
        except ImportError:
            import sys
            import os as _os
            parent_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from utils import undo_carla_coordinate_transform
            _undo_carla_coordinate_transform = undo_carla_coordinate_transform
    return _undo_carla_coordinate_transform


class DGGTRenderer:
    """
    DGGT Gaussian Splatting Renderer

    Wraps gsplat.rasterization for rendering DGGT scenes with:
    - Static scene Gaussians (loaded once)
    - Sky Gaussians (loaded once, optional)
    - Dynamic object Gaussians (loaded per frame, transformed from local to world)

    Supports pose overrides for:
    - Camera poses (frame-specific C2W matrices)
    - Object poses (object-specific world transforms)
    """

    def __init__(self, scene_path: str, device: str = "cuda"):
        """
        Initialize DGGT Renderer

        Args:
            scene_path: Path to DGGT scene directory containing:
                - gaussians/static_scene.ply
                - gaussians/sky_scene.ply (optional)
                - gaussians/frame_{t:04d}_dynamic.ply
                - dynamic_objects/frame_{t:04d}_objects.json
                - ego_pose/frame_{t:04d}_ego.json
            device: Torch device ("cuda" or "cpu")
        """
        self.scene_path = scene_path
        self.device = device

        # Paths to Gaussian data
        self.static_ply = os.path.join(scene_path, "gaussians", "static_scene.ply")
        self.sky_ply = os.path.join(scene_path, "gaussians", "sky_scene.ply")
        self.dynamic_dir = os.path.join(scene_path, "gaussians")
        self.meta_dir = os.path.join(scene_path, "dynamic_objects")
        self.ego_dir = os.path.join(scene_path, "ego_pose")

        # Load static scene (one-time load)
        if not os.path.exists(self.static_ply):
            raise FileNotFoundError(f"Static scene PLY not found: {self.static_ply}")

        self.static_gs = self._load_ply(self.static_ply)

        # Load sky scene (optional)
        if os.path.exists(self.sky_ply):
            self.sky_gs = self._load_ply(self.sky_ply)
        else:
            self.sky_gs = None

        # Pose override storage
        self.cam_overrides: Dict[int, torch.Tensor] = {}  # frame_idx -> C2W 4x4
        self.obj_overrides: Dict[int, Dict[int, torch.Tensor]] = {}  # frame_idx -> {obj_id -> world_pose 4x4}

    def _load_ply(self, path: str) -> Dict[str, torch.Tensor]:
        """
        Load PLY file containing Gaussian splatting data

        Args:
            path: Path to PLY file

        Returns:
            Dict with keys:
                - 'means': (N, 3) positions
                - 'scales': (N, 3) scale parameters
                - 'quats': (N, 4) quaternions (w, x, y, z)
                - 'opacities': (N,) opacity values
                - 'colors': (N, 3) RGB colors (DC coefficients)
                - 'object_ids': (N,) optional, for dynamic objects
        """
        plydata = PlyData.read(path)
        v = plydata['vertex']

        # Extract Gaussian properties
        data = {
            'means': torch.stack([
                torch.tensor(v['x'], dtype=torch.float32),
                torch.tensor(v['y'], dtype=torch.float32),
                torch.tensor(v['z'], dtype=torch.float32)
            ], dim=-1).to(self.device),

            'scales': torch.stack([
                torch.tensor(v['scale_0'], dtype=torch.float32),
                torch.tensor(v['scale_1'], dtype=torch.float32),
                torch.tensor(v['scale_2'], dtype=torch.float32)
            ], dim=-1).to(self.device),

            'quats': torch.stack([
                torch.tensor(v['rot_0'], dtype=torch.float32),  # w
                torch.tensor(v['rot_1'], dtype=torch.float32),  # x
                torch.tensor(v['rot_2'], dtype=torch.float32),  # y
                torch.tensor(v['rot_3'], dtype=torch.float32)   # z
            ], dim=-1).to(self.device),

            'opacities': torch.tensor(v['opacity'], dtype=torch.float32).to(self.device),

            'colors': torch.stack([
                torch.tensor(v['f_dc_0'], dtype=torch.float32),
                torch.tensor(v['f_dc_1'], dtype=torch.float32),
                torch.tensor(v['f_dc_2'], dtype=torch.float32)
            ], dim=-1).to(self.device)
        }

        # Dynamic objects have object_id field
        if 'object_id' in v:
            data['object_ids'] = torch.tensor(v['object_id'], dtype=torch.int32).to(self.device)

        return data

    def set_camera_pose(self, frame_idx: int, pose_matrix: Optional[np.ndarray]) -> None:
        """
        Override camera pose for a specific frame

        Args:
            frame_idx: Frame index
            pose_matrix: 4x4 C2W (Camera-to-World) matrix, or None to clear override
        """
        if pose_matrix is None:
            if frame_idx in self.cam_overrides:
                del self.cam_overrides[frame_idx]
        else:
            if isinstance(pose_matrix, np.ndarray):
                pose_matrix = torch.tensor(pose_matrix, device=self.device, dtype=torch.float32)
            self.cam_overrides[frame_idx] = pose_matrix

    def set_object_pose(self, frame_idx: int, object_id: int, pose_matrix: Optional[np.ndarray]) -> None:
        """
        Override object pose for a specific frame

        Args:
            frame_idx: Frame index
            object_id: Object ID (from DGGT clustering)
            pose_matrix: 4x4 world transform matrix, or None to clear override
        """
        if frame_idx not in self.obj_overrides:
            self.obj_overrides[frame_idx] = {}

        if pose_matrix is None:
            if object_id in self.obj_overrides[frame_idx]:
                del self.obj_overrides[frame_idx][object_id]
        else:
            if isinstance(pose_matrix, np.ndarray):
                pose_matrix = torch.tensor(pose_matrix, device=self.device, dtype=torch.float32)
            self.obj_overrides[frame_idx][object_id] = pose_matrix

    def clear_overrides(self) -> None:
        """Clear all pose overrides"""
        self.cam_overrides.clear()
        self.obj_overrides.clear()

    def _transform_gaussians(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        transform_mat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 4x4 transform to Gaussian means and rotations

        Args:
            means: (N, 3) Gaussian positions
            quats: (N, 4) Gaussian quaternions (w, x, y, z)
            transform_mat: (4, 4) transformation matrix

        Returns:
            new_means: (N, 3) transformed positions
            new_quats: (N, 4) transformed quaternions
        """
        R_mat = transform_mat[:3, :3]
        T_vec = transform_mat[:3, 3]

        # Transform positions: p' = R * p + T
        new_means = (R_mat @ means.T).T + T_vec

        # Transform rotations: q' = q_transform * q_original
        # Convert rotation matrix to quaternion
        r = R.from_matrix(R_mat.cpu().numpy())
        q_trans_np = r.as_quat()  # scipy returns (x, y, z, w)

        # Convert to (w, x, y, z) format for gsplat
        q_trans_wxyz = np.array([q_trans_np[3], q_trans_np[0], q_trans_np[1], q_trans_np[2]])
        q_trans = torch.tensor(q_trans_wxyz, device=self.device, dtype=torch.float32)

        # Quaternion multiplication
        def quat_mult(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
            """Multiply quaternions q1 * q2 (both in wxyz format)"""
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack([w, x, y, z], dim=-1)

        # Broadcast q_trans to match quats shape and multiply
        new_quats = quat_mult(q_trans.unsqueeze(0).expand_as(quats), quats)

        return new_means, new_quats

    def render_frame(
        self,
        frame_idx: int,
        camera_pose_override: Optional[np.ndarray] = None,
        object_pose_overrides: Optional[Dict[int, np.ndarray]] = None,
        use_scene_defaults: bool = True,
        intrinsics_override: Optional[np.ndarray] = None,
        width_override: Optional[int] = None,
        height_override: Optional[int] = None
    ) -> np.ndarray:
        """
        Render a single frame

        Args:
            frame_idx: Frame index to render
            camera_pose_override: Optional C2W matrix override (4x4)
            object_pose_overrides: Optional dict of {obj_id: world_pose 4x4}
            use_scene_defaults: If True, use scene's default poses when no override
            intrinsics_override: Optional 3x3 intrinsic matrix override
            width_override: Optional image width override
            height_override: Optional image height override

        Returns:
            RGB image as numpy array (H, W, 3) uint8
        """
        # Apply temporary overrides
        if camera_pose_override is not None:
            self.set_camera_pose(frame_idx, camera_pose_override)
        if object_pose_overrides is not None:
            for obj_id, pose in object_pose_overrides.items():
                self.set_object_pose(frame_idx, obj_id, pose)

        try:
            image = self._render_frame_internal(
                frame_idx, use_scene_defaults,
                intrinsics_override=intrinsics_override,
                width_override=width_override,
                height_override=height_override
            )
        finally:
            # Clear temporary overrides
            if camera_pose_override is not None:
                self.set_camera_pose(frame_idx, None)
            if object_pose_overrides is not None:
                for obj_id in object_pose_overrides.keys():
                    self.set_object_pose(frame_idx, obj_id, None)

        return image

    def _render_frame_internal(
        self,
        frame_idx: int,
        use_scene_defaults: bool,
        intrinsics_override: Optional[np.ndarray] = None,
        width_override: Optional[int] = None,
        height_override: Optional[int] = None
    ) -> np.ndarray:
        """
        Internal frame rendering logic

        Args:
            frame_idx: Frame index
            use_scene_defaults: Use scene default poses when no override
            intrinsics_override: Optional 3x3 intrinsic matrix override
            width_override: Optional image width override
            height_override: Optional image height override

        Returns:
            RGB image (H, W, 3) uint8
        """
        # 1. Load camera/ego data
        ego_path = os.path.join(self.ego_dir, f"frame_{frame_idx:04d}_ego.json")
        if not os.path.exists(ego_path):
            raise FileNotFoundError(f"Ego pose file not found: {ego_path}")

        with open(ego_path, 'r') as f:
            ego_data = json.load(f)

        # Get camera pose (C2W)
        if frame_idx in self.cam_overrides:
            c2w = self.cam_overrides[frame_idx]
        elif use_scene_defaults:
            c2w_np = np.array(ego_data['camera_extrinsics_world'], dtype=np.float32)
            c2w = torch.tensor(c2w_np, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"No camera pose available for frame {frame_idx}")

        # Pad 3x4 to 4x4 if needed
        if c2w.shape == (3, 4):
            bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]],
                                       device=self.device, dtype=torch.float32)
            c2w = torch.cat([c2w, bottom_row], dim=0)

        # Get intrinsics (use override if provided, otherwise read from JSON)
        if intrinsics_override is not None:
            K_np = np.array(intrinsics_override, dtype=np.float32)
        else:
            K_np = np.array(ego_data['camera_intrinsics'], dtype=np.float32)
        K = torch.tensor(K_np, device=self.device, dtype=torch.float32)
        W = width_override if width_override is not None else ego_data['camera']['width']
        H = height_override if height_override is not None else ego_data['camera']['height']

        # Compute view matrix (W2C) = inverse(C2W)
        viewmat = torch.inverse(c2w)

        # 2. Load and transform dynamic objects
        dyn_ply_path = os.path.join(self.dynamic_dir, f"frame_{frame_idx:04d}_dynamic.ply")
        obj_meta_path = os.path.join(self.meta_dir, f"frame_{frame_idx:04d}_objects.json")

        dyn_means_list: List[torch.Tensor] = []
        dyn_quats_list: List[torch.Tensor] = []
        dyn_scales_list: List[torch.Tensor] = []
        dyn_opac_list: List[torch.Tensor] = []
        dyn_cols_list: List[torch.Tensor] = []

        if os.path.exists(dyn_ply_path) and os.path.exists(obj_meta_path):
            dyn_gs = self._load_ply(dyn_ply_path)

            with open(obj_meta_path, 'r') as f:
                obj_meta = json.load(f)

            # Build default poses dict
            default_poses: Dict[int, torch.Tensor] = {}
            for obj in obj_meta:
                obj_id = obj['object_id']
                pose_np = np.array(obj['pose_world'], dtype=np.float32)
                default_poses[obj_id] = torch.tensor(pose_np, device=self.device, dtype=torch.float32)
                logger.info(f"[RENDERER] Default pose for object_id={obj_id}: [{pose_np[0,3]:.6f}, {pose_np[1,3]:.6f}, {pose_np[2,3]:.6f}]")

            # Process each unique object ID
            if 'object_ids' in dyn_gs:
                unique_ids = torch.unique(dyn_gs['object_ids'])

                for uid in unique_ids:
                    uid_int = int(uid.item())
                    if uid_int == -1:  # Skip noise points
                        continue

                    # Mask for this object
                    mask = (dyn_gs['object_ids'] == uid)

                    # Get local Gaussians
                    means_local = dyn_gs['means'][mask]
                    quats_local = dyn_gs['quats'][mask]

                    # Determine transform matrix (override vs default)
                    if frame_idx in self.obj_overrides and uid_int in self.obj_overrides[frame_idx]:
                        transform = self.obj_overrides[frame_idx][uid_int]
                        transform_np = transform.cpu().numpy()
                        logger.info(f"[RENDERER] Using OVERRIDE pose for object_id={uid_int}: [{transform_np[0,3]:.6f}, {transform_np[1,3]:.6f}, {transform_np[2,3]:.6f}]")
                    elif use_scene_defaults and uid_int in default_poses:
                        transform = default_poses[uid_int]
                        transform_np = transform.cpu().numpy()
                        logger.info(f"[RENDERER] Using DEFAULT pose for object_id={uid_int}: [{transform_np[0,3]:.6f}, {transform_np[1,3]:.6f}, {transform_np[2,3]:.6f}]")
                    else:
                        # No pose available, skip this object
                        logger.warning(f"[RENDERER] No pose available for object_id={uid_int}, skipping")
                        continue

                    # Apply transform (Local -> World)
                    means_world, quats_world = self._transform_gaussians(
                        means_local, quats_local, transform
                    )

                    dyn_means_list.append(means_world)
                    dyn_quats_list.append(quats_world)
                    dyn_scales_list.append(dyn_gs['scales'][mask])
                    dyn_opac_list.append(dyn_gs['opacities'][mask])
                    dyn_cols_list.append(dyn_gs['colors'][mask])

        # 3. Combine all Gaussians
        all_means: List[torch.Tensor] = []
        all_quats: List[torch.Tensor] = []
        all_scales: List[torch.Tensor] = []
        all_opac: List[torch.Tensor] = []
        all_cols: List[torch.Tensor] = []

        # Sky (if exists) - add first so it's rendered behind
        if self.sky_gs is not None:
            all_means.append(self.sky_gs['means'])
            all_quats.append(self.sky_gs['quats'])
            all_scales.append(self.sky_gs['scales'])
            all_opac.append(self.sky_gs['opacities'])
            all_cols.append(self.sky_gs['colors'])

        # Static scene
        all_means.append(self.static_gs['means'])
        all_quats.append(self.static_gs['quats'])
        all_scales.append(self.static_gs['scales'])
        all_opac.append(self.static_gs['opacities'])
        all_cols.append(self.static_gs['colors'])

        # Dynamic objects
        all_means.extend(dyn_means_list)
        all_quats.extend(dyn_quats_list)
        all_scales.extend(dyn_scales_list)
        all_opac.extend(dyn_opac_list)
        all_cols.extend(dyn_cols_list)

        # Concatenate
        final_means = torch.cat(all_means, dim=0)
        final_quats = torch.cat(all_quats, dim=0)
        final_scales = torch.cat(all_scales, dim=0)
        final_opac = torch.cat(all_opac, dim=0)
        final_cols = torch.cat(all_cols, dim=0)

        # 4. Rasterize with gsplat
        renders, _, _ = rasterization(
            means=final_means,
            quats=final_quats,
            scales=final_scales,
            opacities=final_opac,
            colors=final_cols,
            viewmats=viewmat.unsqueeze(0),  # (1, 4, 4)
            Ks=K.unsqueeze(0),              # (1, 3, 3)
            width=W,
            height=H,
            render_mode='RGB'
        )

        # Convert to numpy image
        image_out = renders[0].detach().cpu().clamp(0, 1).numpy()
        image_out = (image_out * 255).astype(np.uint8)

        return image_out

    def get_frame_metadata(self, frame_idx: int) -> Dict:
        """
        Get frame metadata from ego pose JSON

        Args:
            frame_idx: Frame index

        Returns:
            Dict with camera intrinsics, extrinsics, etc.
        """
        ego_path = os.path.join(self.ego_dir, f"frame_{frame_idx:04d}_ego.json")
        if not os.path.exists(ego_path):
            raise FileNotFoundError(f"Ego pose file not found: {ego_path}")

        with open(ego_path, 'r') as f:
            return json.load(f)

    def get_object_metadata(self, frame_idx: int) -> List[Dict]:
        """
        Get dynamic object metadata for a frame

        Args:
            frame_idx: Frame index

        Returns:
            List of object dicts with object_id, pose_world, dimensions
        """
        obj_meta_path = os.path.join(self.meta_dir, f"frame_{frame_idx:04d}_objects.json")
        if not os.path.exists(obj_meta_path):
            return []

        with open(obj_meta_path, 'r') as f:
            return json.load(f)

    def get_object_ids(self, frame_idx: int) -> List[int]:
        """
        Get list of object IDs present in a frame

        Args:
            frame_idx: Frame index

        Returns:
            List of object IDs (excluding -1 noise)
        """
        obj_meta = self.get_object_metadata(frame_idx)
        return [obj['object_id'] for obj in obj_meta if obj['object_id'] != -1]


# Convenience function for quick rendering
def render_dggt_frame(
    scene_path: str,
    frame_idx: int,
    camera_pose: Optional[np.ndarray] = None,
    object_poses: Optional[Dict[int, np.ndarray]] = None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Convenience function to render a single DGGT frame

    Args:
        scene_path: Path to DGGT scene directory
        frame_idx: Frame index to render
        camera_pose: Optional C2W override (4x4)
        object_poses: Optional {obj_id: world_pose} overrides
        device: Torch device

    Returns:
        RGB image (H, W, 3) uint8
    """
    renderer = DGGTRenderer(scene_path, device=device)
    return renderer.render_frame(frame_idx, camera_pose, object_poses)