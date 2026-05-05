import os
import json
import torch
import numpy as np
import cv2
from plyfile import PlyData
import torch.nn.functional as F
from gsplat.rendering import rasterization
from scipy.spatial.transform import Rotation as R

class DGGTRenderer:
    def __init__(self, scene_path, device="cuda"):
        self.scene_path = scene_path
        self.device = device
        
        # Paths
        self.static_ply = os.path.join(scene_path, "gaussians", "static_scene.ply")
        self.sky_ply = os.path.join(scene_path, "gaussians", "sky_scene.ply")
        self.dynamic_dir = os.path.join(scene_path, "gaussians")
        self.meta_dir = os.path.join(scene_path, "dynamic_objects")
        self.ego_dir = os.path.join(scene_path, "ego_pose")
        
        # Load Static Scene (Once)
        print("Loading static scene...")
        self.static_gs = self._load_ply(self.static_ply)
        if os.path.exists(self.sky_ply):
            print("Loading sky scene...")
            self.sky_gs = self._load_ply(self.sky_ply)
        else:
            print("Warning: No sky_scene.ply found.")
            self.sky_gs = None
        
        # State storage for overrides
        self.cam_overrides = {} # frame_idx -> 4x4 matrix
        self.obj_overrides = {} # frame_idx -> {obj_id -> 4x4 matrix}

    def _load_ply(self, path):
        """Loads PLY and returns dict of tensors on device."""
        plydata = PlyData.read(path)
        v = plydata['vertex']
        
        data = {
            'means': torch.stack([torch.tensor(v['x']), torch.tensor(v['y']), torch.tensor(v['z'])], dim=-1).to(self.device),
            'scales': torch.stack([torch.tensor(v['scale_0']), torch.tensor(v['scale_1']), torch.tensor(v['scale_2'])], dim=-1).to(self.device),
            'quats': torch.stack([torch.tensor(v['rot_0']), torch.tensor(v['rot_1']), torch.tensor(v['rot_2']), torch.tensor(v['rot_3'])], dim=-1).to(self.device),
            'opacities': torch.tensor(v['opacity']).to(self.device),
            'colors': torch.stack([torch.tensor(v['f_dc_0']), torch.tensor(v['f_dc_1']), torch.tensor(v['f_dc_2'])], dim=-1).to(self.device)
        }
        
        # Optional Object ID for dynamic files
        if 'object_id' in v:
            data['object_ids'] = torch.tensor(v['object_id']).to(self.device)
            
        return data

    def set_camera_pose(self, frame_idx, pose_matrix):
        """Override camera pose for specific frame. pose_matrix: 4x4 Tensor or Numpy"""
        if pose_matrix is None:
            if frame_idx in self.cam_overrides: del self.cam_overrides[frame_idx]
        else:
            if isinstance(pose_matrix, np.ndarray): pose_matrix = torch.tensor(pose_matrix, device=self.device).float()
            self.cam_overrides[frame_idx] = pose_matrix

    def set_object_pose(self, frame_idx, object_id, pose_matrix):
        """Override object pose. pose_matrix is the new World Transform."""
        if frame_idx not in self.obj_overrides: self.obj_overrides[frame_idx] = {}
        
        if pose_matrix is None:
            if object_id in self.obj_overrides[frame_idx]: del self.obj_overrides[frame_idx][object_id]
        else:
            if isinstance(pose_matrix, np.ndarray): pose_matrix = torch.tensor(pose_matrix, device=self.device).float()
            self.obj_overrides[frame_idx][object_id] = pose_matrix

    def _transform_gaussians(self, means, quats, transform_mat):
        """
        Applies 4x4 transform to Gaussian means and rotations.
        means: (N, 3), quats: (N, 4), transform_mat: (4, 4)
        """
        # 1. Transform Positions: p' = R*p + T
        R_mat = transform_mat[:3, :3]
        T_vec = transform_mat[:3, 3]
        
        new_means = (R_mat @ means.T).T + T_vec
        
        # 2. Transform Rotations: q' = q_transform * q_original
        # Convert matrix R to quaternion [w, x, y, z] (scipy uses x,y,z,w)
        r = R.from_matrix(R_mat.cpu().numpy())
        q_trans_np = r.as_quat() # xyzw
        # Rearrange to wxyz for gsplat consistency if needed (GS usually uses wxyz or xyzw, check implementation)
        # Assuming gsplat uses WXYZ convention standard in pytorch3d/vggt logic
        q_trans_wxyz = np.r_[q_trans_np[3], q_trans_np[:3]] 
        q_trans = torch.tensor(q_trans_wxyz, device=self.device, dtype=torch.float32)
        
        # Quaternion multiplication (batch)
        # Simple approximation: For rendering, we assume the internal rotation logic handles normalization
        # Note: This is a simplified rotation update. 
        # A robust implementation requires standard q_mult logic.
        
        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.stack((w, x, y, z), dim=-1)

        new_quats = quat_mult(q_trans.unsqueeze(0).expand_as(quats), quats)
        return new_means, new_quats

    def _get_bbox_corners_2d(self, pose, dimensions, viewmat, K, W, H):
        """
        Calculates the 2D image coordinates of the 3D bounding box corners.
        """
        # 1. Define 8 corners in Local Space (assuming centroid-centered)
        # dimensions: [length, width, height]
        dx, dy, dz = dimensions[0] / 2.0, dimensions[1] / 2.0, dimensions[2] / 2.0

        # Order: Bottom 4, then Top 4
        corners_local = torch.tensor([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],  # Bottom
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]  # Top
        ], device=self.device, dtype=torch.float32)

        # 2. Transform Local -> World
        # pose is 4x4. We treat corners as vectors with w=1
        ones = torch.ones((8, 1), device=self.device)
        corners_hom = torch.cat([corners_local, ones], dim=1)  # (8, 4)

        # P_world = P_local * Pose^T
        corners_world = corners_hom @ pose.T

        # 3. Transform World -> Camera
        # P_cam = P_world * Viewmat^T (gsplat/opengl style usually)
        corners_cam = corners_world @ viewmat.T

        # 4. Project Camera -> Image
        xyz = corners_cam[:, :3]

        # Check if behind camera (z < 0 convention for OpenGL, z > 0 for OpenCV)
        # DGGT usually uses OpenCV convention where Z+ is forward.
        # Simple clipping check: if z <= 0.1, we can't project validly.
        if (xyz[:, 2] < 0.1).any():
            return None  # Box is partially or fully behind camera

        # Apply Intrinsics: P_img = K * P_cam
        # K is (3, 3). xyz is (8, 3). Need (K @ xyz.T).T
        uv_w = (K @ xyz.T).T

        # Normalize u = x/z, v = y/z
        uv = uv_w[:, :2] / (uv_w[:, 2:3] + 1e-6)

        return uv.detach().cpu().numpy()

    def _draw_bbox(self, image, corners_2d, color=(0, 255, 0), thickness=2):
        """
        Draws lines connecting the 8 corners on the numpy image.
        """
        if corners_2d is None: return image

        pts = corners_2d.astype(np.int32)

        # Edges indices based on the corner definition in _get_bbox_corners_2d
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
        ]

        H, W = image.shape[:2]

        # Draw lines
        for start, end in edges:
            pt1 = tuple(pts[start])
            pt2 = tuple(pts[end])

            # Simple bounds check to prevent drawing massive lines across screen if projection goes wild
            if -W < pt1[0] < 2 * W and -H < pt1[1] < 2 * H:
                cv2.line(image, pt1, pt2, color, thickness)

        return image

    def render_sequence(self, num_frames, output_dir, draw_bboxes=False):
        os.makedirs(output_dir, exist_ok=True)

        for t in range(num_frames):
            print(f"Rendering frame {t}/{num_frames}...")

            # -------------------------------------------------
            # 1. Load Camera & Ego Data
            # -------------------------------------------------
            ego_path = os.path.join(self.ego_dir, f"frame_{t:04d}_ego.json")
            with open(ego_path, 'r') as f: ego_data = json.load(f)

            # Check for Camera Override
            if t in self.cam_overrides:
                c2w = self.cam_overrides[t]
            else:
                c2w = torch.tensor(ego_data['camera_extrinsics_world'], device=self.device).float()

            # If the matrix is 3x4, pad it to 4x4
            if c2w.shape == (3, 4):
                bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=c2w.dtype)
                c2w = torch.cat([c2w, bottom_row], dim=0)  # Now 4x4

            # Intrinsics
            K = torch.tensor(ego_data['camera_intrinsics'], device=self.device).float()
            W, H = ego_data['camera']['width'], ego_data['camera']['height']

            # Invert C2W to W2C (View Matrix) for rasterizer
            # gsplat usually expects W2C.
            viewmat = torch.inverse(c2w)

            # -------------------------------------------------
            # 2. Load & Transform Dynamic Objects
            # -------------------------------------------------
            dyn_ply_path = os.path.join(self.dynamic_dir, f"frame_{t:04d}_dynamic.ply")
            obj_meta_path = os.path.join(self.meta_dir, f"frame_{t:04d}_objects.json")

            dyn_means, dyn_quats, dyn_scales, dyn_opac, dyn_cols = [], [], [], [], []

            if os.path.exists(dyn_ply_path):
                dyn_gs = self._load_ply(dyn_ply_path)
                with open(obj_meta_path, 'r') as f: obj_meta = json.load(f)

                # Map object IDs to default poses
                default_poses = {o['object_id']: torch.tensor(o['pose_world'], device=self.device).float() for o in obj_meta}

                unique_ids = torch.unique(dyn_gs['object_ids'])

                for uid in unique_ids:
                    uid_int = int(uid.item())
                    if uid_int == -1: continue # Skip noise

                    # Mask for this object
                    mask = (dyn_gs['object_ids'] == uid)

                    # Get Local Gaussians
                    p_local = dyn_gs['means'][mask]
                    q_local = dyn_gs['quats'][mask]

                    # Determine Transformation Matrix
                    # Check override first, then default
                    if t in self.obj_overrides and uid_int in self.obj_overrides[t]:
                        transform = self.obj_overrides[t][uid_int]
                    else:
                        transform = default_poses.get(uid_int, torch.eye(4, device=self.device))

                    # Apply Transform (Local -> World)
                    p_world, q_world = self._transform_gaussians(p_local, q_local, transform)

                    dyn_means.append(p_world)
                    dyn_quats.append(q_world)
                    dyn_scales.append(dyn_gs['scales'][mask])
                    dyn_opac.append(dyn_gs['opacities'][mask])
                    dyn_cols.append(dyn_gs['colors'][mask])

            # -------------------------------------------------
            # 3. Combine Scene
            # -------------------------------------------------
            # Start lists
            all_means, all_quats, all_scales, all_opac, all_cols = [], [], [], [], []

            # A. Add Sky (if exists)
            if self.sky_gs is not None:
                all_means.append(self.sky_gs['means'])
                all_quats.append(self.sky_gs['quats'])
                all_scales.append(self.sky_gs['scales'])
                all_opac.append(self.sky_gs['opacities'])
                all_cols.append(self.sky_gs['colors'])

            # B. Add Static Scene
            all_means.append(self.static_gs['means'])
            all_quats.append(self.static_gs['quats'])
            all_scales.append(self.static_gs['scales'])
            all_opac.append(self.static_gs['opacities'])
            all_cols.append(self.static_gs['colors'])

            # Add dynamics if exist
            if len(dyn_means) > 0:
                all_means.extend(dyn_means)
                all_quats.extend(dyn_quats)
                all_scales.extend(dyn_scales)
                all_opac.extend(dyn_opac)
                all_cols.extend(dyn_cols)

            final_means = torch.cat(all_means, dim=0)
            final_quats = torch.cat(all_quats, dim=0)
            final_scales = torch.cat(all_scales, dim=0)
            final_opac = torch.cat(all_opac, dim=0)
            final_cols = torch.cat(all_cols, dim=0)

            # -------------------------------------------------
            # 4. Rasterize
            # -------------------------------------------------
            renders, _, _ = rasterization(
                means=final_means,
                quats=final_quats,
                scales=final_scales,
                opacities=final_opac,
                colors=final_cols,
                viewmats=viewmat.unsqueeze(0), # (1, 4, 4)
                Ks=K.unsqueeze(0),             # (1, 3, 3)
                width=W,
                height=H,
                render_mode='RGB'
            )

            # Save Image
            image_out = renders[0].detach().cpu().clamp(0,1).numpy()
            image_out = (image_out * 255).astype(np.uint8)

            if draw_bboxes:
                # Load metadata again to get dimensions (if not already cached)
                obj_meta_path = os.path.join(self.meta_dir, f"frame_{t:04d}_objects.json")
                if os.path.exists(obj_meta_path):
                    with open(obj_meta_path, 'r') as f:
                        obj_meta = json.load(f)

                    # Convert image to writable if needed (sometimes torch tensor output is read-only)
                    image_out = image_out.copy()

                    # Map object IDs to default poses/dims
                    for obj in obj_meta:
                        uid = obj['object_id']
                        if uid == -1: continue

                        dims = obj['dimensions']

                        # Determine current pose (Default vs Override)
                        if t in self.obj_overrides and uid in self.obj_overrides[t]:
                            pose = self.obj_overrides[t][uid]
                        else:
                            pose = torch.tensor(obj['pose_world'], device=self.device).float()

                        # Calculate Corners & Project
                        corners_2d = self._get_bbox_corners_2d(pose, dims, viewmat, K, W, H)

                        # Draw
                        # Use Red (0, 0, 255) if it's modified, Green (0, 255, 0) otherwise
                        is_modified = (t in self.obj_overrides and uid in self.obj_overrides[t])
                        color = (255, 0, 0) if is_modified else (0, 255, 0)  # RGB

                        self._draw_bbox(image_out, corners_2d, color=color)

            import imageio
            imageio.imwrite(os.path.join(output_dir, f"render_frame_{t:04d}.png"), image_out)

# Usage Example
if __name__ == "__main__":
    engine = DGGTRenderer("/home/junchuan/e2e/dggt/output/waymo/training/scene1/0412/001")
    
    # Example: Shift Object 1 by 2 meters in X axis at frame 10
    # Assume we know object 1 exists
    # 1. Get original pose (helper logic would be needed to fetch this easily, 
    #    here we assume we construct a new matrix)
    # new_pose = torch.tensor([
    #     [1.0, 0.0, 0.0, -0.030429556965827942],
    #     [0.0, 1.0, 0.0, 1.0325412191450596],  # <--- Modified Y translation
    #     [0.0, 0.0, 1.0, 0.6571296453475952],
    #     [0.0, 0.0, 0.0, 1.0]
    # ], dtype=torch.float32, device="cuda")  # Ensure device matches your engine's device
    
    # 2. Modify Pose
    # new_pose = ... (4x4 Tensor)
    # engine.set_object_pose(0, 0, new_pose)
    
    engine.render_sequence(num_frames=20, output_dir="/home/junchuan/e2e/dggt/output/waymo/training/scene1/0412/001/render_test", draw_bboxes=True)
