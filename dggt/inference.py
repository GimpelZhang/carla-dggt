import argparse
import os
import random
import time
import json
import numpy as np
from scipy.spatial import cKDTree
import scipy.spatial.transform
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import open3d as o3d
from sklearn.cluster import DBSCAN  # Added for object clustering

# --- DGGT Modules ---
from third_party.difix.infer import process_images_with_difix
from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, \
    resize_depth_bilinear
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_masked_gs, get_split_gs
from dggt.utils.visual_track import visualize_tracks_on_images
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
from utils.interplation import interp_all
from utils.video_maker import make_comparison_video_quad

# --- Export Utilities ---
from utils.export_utils import save_json, save_ply
from utils.gt_data_loader import load_ego_pose_translations


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0) ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    alpha_ = alpha * conf
    return alpha_.float()


def compute_metrics(img1, img2, loss_fn):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)
        lpips_val = loss_fn(img1[i].unsqueeze(0) * 2 - 1, img2[i].unsqueeze(0) * 2 - 1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val.item())
    return sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list), sum(lpips_list) / len(lpips_list)


def parse_scene_names(scene_names_str):
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        return [str(i).zfill(3) for i in range(int(start), int(end) + 1)]
    else:
        return [str(int(x)).zfill(3) for x in scene_names_str.split()]
    
def calculate_scale_factor(P1, P2):
    """
    Calculate scale factor between predicted and GT trajectories.

    Args:
        P1: Predicted trajectory translations (S, 3)
        P2: GT ego_pose translations (S, 3)

    Returns:
        scale_factor: Physical scale to recover metric distances
    """
    # Compute frame-to-frame movement (not absolute positions)
    movement_P1 = P1[1:] - P1[:-1]  # Movement vectors between consecutive frames
    movement_P2 = P2[1:] - P2[:-1]

    distances_P1 = torch.norm(movement_P1, dim=1)  # Distance moved per frame step
    distances_P2 = torch.norm(movement_P2, dim=1)

    avg_distance_P1 = torch.mean(distances_P1)
    avg_distance_P2 = torch.mean(distances_P2)
    if avg_distance_P1 < 0.001:  # Almost stationary
        return 1
    scale_factor = avg_distance_P2 / avg_distance_P1
    return scale_factor

def save_video(images, path, fps=8):
    images = images.detach().cpu()  # Ensure it's on CPU
    if images.max() <= 1.0:
        images = images * 255.0
    images = images.byte().permute(0, 2, 3, 1).numpy()  # [S, H, W, 3]
    
    imageio.mimwrite(path, images, fps=fps, codec='libx264')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the input images')
    parser.add_argument('--scene_names', type=str, nargs='+', required=True,
                        help='Scene names, supports formats like 3 5 7 or (3,7)')
    parser.add_argument('--input_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], required=True, help='Processing mode')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for results')
    parser.add_argument('-images', action='store_true', help='Whether to output each frame image')
    parser.add_argument('-depth', action='store_true', help='Whether to output each frame depth as .npy')
    parser.add_argument('-metrics', action='store_true', help='Whether to output evaluation metrics')
    parser.add_argument('-diffusion', action='store_true', help='Whether to process images with diffusion model')
    parser.add_argument('--intervals', type=int, default=2, help='Interval for mode=3')
    parser.add_argument('--dataset', type=str, default='waymo', choices=['waymo', 'nuscenes'],
                        help='Dataset type: waymo (default) or nuscenes')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    loss_fn = lpips.LPIPS(net='alex').to(device)

    scene_names_str = ' '.join(args.scene_names)
    scene_names = parse_scene_names(scene_names_str)

    # Initialize Dataset
    if args.mode == 3:
        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=scene_names,
            sequence_length=args.sequence_length,
            start_idx=args.start_idx,
            mode=args.mode,
            views=args.input_views,
            intervals=args.intervals
        )
    else:
        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=scene_names,
            sequence_length=args.sequence_length,
            start_idx=args.start_idx,
            mode=args.mode,
            views=args.input_views
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize Model
    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)

    if args.mode == 3:
        track_ckpt = 'pretrained/tracking_model.pth'  # Ensure this path is correct
        track_model = load_model(track_ckpt)
        track_model.to(device)
        track_model.seq_len = 2
    else:
        track_model = None

    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []
    inference_time_list = []
    scene_idx = 1

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            gt_dy_map = batch['dynamic_mask'].to(device)
            gt_depth = batch['gt_depth'].to(device)

            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            if args.mode == 3:
                target_images = batch['targets'].to(device)
                target_sky_masks = batch['target_masks'].to(device)
            else:
                target_images = None

            start_time = time.time()

            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))

                # Pad extrinsics to 4x4
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(
                    extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                intervals = args.intervals
                views = args.input_views

                # Get Geometry
                depth_map = predictions["depth"][0]
                point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None, ...]
                point_map = torch.from_numpy(point_map).to(device).float()

                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1)  # B,H,W

                # --- INTERPOLATION MODE LOGIC ---
                if args.mode == 3:
                    depth_map = depth_map.unsqueeze(0)
                    if args.input_views == 1:
                        (extrinsic, intrinsic, point_map, gs_map, dy_map,
                         gs_conf, bg_mask, images, pred_flows, flow_masks, depth_interp) = interp_all(
                            extrinsic, intrinsic, point_map, gs_map, dy_map,
                            gs_conf, bg_mask, images, target_images, depth_map, track_model, intervals, views
                        )

                    I = intervals
                    bg_point_map = point_map[:, ::I, ...]
                    bg_bg_mask = bg_mask[:, ::I, ...]
                    bg_gs_map = gs_map[:, ::I, ...]
                    bg_dy_map = dy_map[:, ::I, ...]
                    bg_gs_conf = gs_conf[:, ::I, ...]

                    static_mask = (bg_bg_mask & (bg_dy_map < 0.5))
                    gs_conf = bg_gs_conf[static_mask]
                    static_points = bg_point_map[static_mask].reshape(-1, 3)
                    gs_dynamic_list = bg_dy_map[static_mask].sigmoid()
                    static_rgbs, static_opacity, static_scales, static_rotation = get_split_gs(bg_gs_map, static_mask)
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
                    gs_timestamps = timestamps[frame_idx]
                    static_opacity = static_opacity * (1 - gs_dynamic_list)

                # --- RECONSTRUCTION MODE LOGIC & SETUP ---
                if args.mode == 2:
                    # Prepare Output Directories
                    scene_name = str(scene_idx).zfill(3)
                    scene_out_dir = os.path.join(args.output_path, scene_name)
                    ego_dir = os.path.join(scene_out_dir, "ego_pose")
                    obj_dir = os.path.join(scene_out_dir, "dynamic_objects")
                    gs_dir = os.path.join(scene_out_dir, "gaussians")
                    os.makedirs(ego_dir, exist_ok=True)
                    os.makedirs(obj_dir, exist_ok=True)
                    os.makedirs(gs_dir, exist_ok=True)

                    print(f"Exporting data for scene {scene_name}...")

                    # Accumulators for Static Scene and Sky
                    acc_static_pos, acc_static_scl, acc_static_rot, acc_static_opa, acc_static_col = [], [], [], [], []
                    acc_sky_pos, acc_sky_scl, acc_sky_rot, acc_sky_opa, acc_sky_col = [], [], [], [], []

                    S = extrinsic.shape[0]
                    print(S)

                    # --------------------------------------------------------------------------
                    # SCALE RECOVERY: Load GT ego pose and calculate scale_factor
                    # --------------------------------------------------------------------------
                    scene_dir = os.path.join(args.image_dir, scene_name)
                    scale_factor = 1.0  # Default fallback

                    try:
                        # Load GT ego pose translations
                        # Use start_idx to get correct frame indices (e.g., start_idx=40 means frames 40-59)
                        frame_indices = list(range(args.start_idx, args.start_idx + S))
                        gt_translations = load_ego_pose_translations(scene_dir, frame_indices, dataset=args.dataset)

                        # Extract predicted translations from extrinsic (W2C matrices)
                        # [修复1] 对extrinsic求逆得到C2W，提取相机中心坐标
                        c2w_all = torch.inverse(extrinsic)  # W2C -> C2W
                        pred_translations = c2w_all[:, :3, 3].clone()  # C2W平移才是相机中心

                        # Calculate scale factor
                        scale_factor_tensor = calculate_scale_factor(pred_translations, gt_translations)

                        # Convert to float for consistent use with numpy operations
                        scale_factor = float(scale_factor_tensor)

                        # Validate scale factor
                        if scale_factor < 1.0 or scale_factor > 100.0:
                            print(f"[Warning] Abnormal scale factor {scale_factor:.4f}, fallback to 1.0")
                            scale_factor = 1.0
                        else:
                            print(f"[Scale Recovery] scale_factor = {scale_factor:.4f}")
                    except FileNotFoundError as e:
                        print(f"[Warning] GT ego pose not found: {e}. scale_factor=1.0")
                        scale_factor = 1.0

                    # --------------------------------------------------------------------------
                    # DATA EXPORT LOOP (Per Frame Accumulation)
                    # --------------------------------------------------------------------------
                    for t in range(S):
                        # 1. Export Ego/Camera JSON
                        # Calculate C2W (Camera-to-World) by inverting W2C (extrinsic)
                        ext_t_4x4 = extrinsic[t]  # Already 4x4
                        c2w = torch.inverse(ext_t_4x4)
                        c2w[:3, 3] *= scale_factor  # Apply scaling to C2W translation (camera center)

                        frame_ego_data = {
                            "frame_id": t,
                            "camera": {
                                "width": W,
                                "height": H
                            },
                            "camera_intrinsics": intrinsic[t].cpu().numpy().tolist(),
                            "camera_extrinsics_world": c2w.cpu().numpy().tolist(),  # Scaled C2W
                            "ego_pose_world": c2w[:3, 3].cpu().numpy().tolist(),  # Correct camera position
                            "scale_factor": float(scale_factor)
                        }
                        save_json(frame_ego_data, os.path.join(ego_dir, f"frame_{t:04d}_ego.json"))

                        # 2. Accumulate Static Scene
                        # Mask for static regions
                        mask_t = (bg_mask[:, t] & (dy_map[:, t] < 0.5))

                        if mask_t.sum() > 0:
                            st_pos = point_map[:, t][mask_t].reshape(-1, 3)
                            st_rgb, st_opa, st_scl, st_rot = get_split_gs(gs_map[:, t], mask_t)

                            # Clean up ghosts
                            st_dyn_conf = dy_map[:, t][mask_t].sigmoid().reshape(-1)
                            st_opa = st_opa * (1 - st_dyn_conf)

                            # Apply scale factor to static scene positions and scales (metric recovery)
                            st_pos = st_pos * scale_factor
                            st_scl = st_scl * scale_factor

                            acc_static_pos.append(st_pos.detach().cpu().numpy())
                            acc_static_scl.append(st_scl.detach().cpu().numpy())
                            acc_static_rot.append(st_rot.detach().cpu().numpy())
                            acc_static_opa.append(st_opa.detach().cpu().numpy())
                            acc_static_col.append(st_rgb.detach().cpu().numpy())

                        # 3. Accumulate Sky (Per-frame Query)
                        img_t = images[:, t:t + 1]
                        # Reshape to match SkyModel expectations [1, 4, 4]
                        ext_t_query = ext_t_4x4.unsqueeze(0)
                        int_t_query = torch.eye(4, device=device).unsqueeze(0)
                        int_t_query[:, :3, :3] = intrinsic[t]

                        with torch.no_grad():
                            sky_rgb, sky_mask_val, _ = model.sky_model._get_background_color(
                                img_t, ext_t_query, int_t_query
                            )
                            # Get Geometry
                            sk_pos = model.sky_model.bg_pcd[sky_mask_val]
                            sk_rot = model.sky_model.bg_quat[sky_mask_val]
                            sk_scl = torch.exp(model.sky_model.bg_scales)[sky_mask_val]
                            sk_opa = model.sky_model.bg_opacity.squeeze(-1)[sky_mask_val]
                            sk_col = sky_rgb

                            # Apply scale factor to sky positions and scales (metric recovery)
                            sk_pos = sk_pos * scale_factor
                            sk_scl = sk_scl * scale_factor

                            acc_sky_pos.append(sk_pos.detach().cpu().numpy())
                            acc_sky_scl.append(sk_scl.detach().cpu().numpy())
                            acc_sky_rot.append(sk_rot.detach().cpu().numpy())
                            acc_sky_opa.append(sk_opa.detach().cpu().numpy())
                            acc_sky_col.append(sk_col.detach().cpu().numpy())

                        # 4. Export Dynamic Objects (Local Space PLY + Metadata) with Scale Recovery
                        obj_mask = (bg_mask[:, t] & (dy_map[:, t] > 0.5))

                        if obj_mask.sum() > 0:
                            dy_pos = point_map[:, t][obj_mask].reshape(-1, 3)
                            dy_rgb, dy_opa, dy_scl, dy_rot = get_split_gs(gs_map[:, t], obj_mask)
                            dy_opa = dy_opa * dy_map[:, t][obj_mask].sigmoid().reshape(-1)

                            # Apply scale recovery to dynamic object positions and Gaussian scales
                            dy_pos_scaled = dy_pos * scale_factor
                            dy_scl_scaled = dy_scl * scale_factor
                            pos_np_scaled = dy_pos_scaled.detach().cpu().numpy()

                            # DBSCAN Clustering on scaled (metric) coordinates
                            if len(pos_np_scaled) > 50:  # Only cluster if enough points
                                clustering = DBSCAN(eps=0.5, min_samples=10).fit(pos_np_scaled)
                                labels = clustering.labels_
                                labels_tensor = torch.tensor(labels, device=dy_pos.device, dtype=torch.int32)

                                local_pos = dy_pos_scaled.clone()
                                objects_meta = []
                                unique_labels = set(labels)
                                if -1 in unique_labels: unique_labels.remove(-1)

                                for label in unique_labels:
                                    indices = (labels == label)
                                    cluster_pts_scaled = pos_np_scaled[indices]
                                    centroid_scaled = cluster_pts_scaled.mean(axis=0)  # Real-world metric centroid

                                    # Transform to Local Space (using scaled coordinates for metric local PLY)
                                    local_pos[indices] = dy_pos_scaled[indices] - torch.tensor(centroid_scaled, device=dy_pos.device)

                                    # Calculate dimensions from scaled coordinates (real-world dimensions)
                                    min_pt = cluster_pts_scaled.min(axis=0)
                                    max_pt = cluster_pts_scaled.max(axis=0)
                                    dimensions = max_pt - min_pt

                                    pose_mat = np.eye(4)
                                    pose_mat[:3, 3] = centroid_scaled  # Metric centroid for world pose

                                    objects_meta.append({
                                        "object_id": int(label),
                                        "pose_world": pose_mat.tolist(),
                                        "dimensions": dimensions.tolist()
                                    })

                                # Save PLY with scaled Gaussian sizes for metric physical representation
                                save_ply(
                                    local_pos, dy_scl_scaled, dy_rot, dy_opa, dy_rgb,
                                    os.path.join(gs_dir, f"frame_{t:04d}_dynamic.ply"),
                                    object_ids=labels_tensor
                                )
                                # Save Metadata
                                save_json(objects_meta, os.path.join(obj_dir, f"frame_{t:04d}_objects.json"))

                    # --------------------------------------------------------------------------
                    # MERGE AND SAVE ACCUMULATED SCENES
                    # --------------------------------------------------------------------------
                    def merge(arr_list):
                        return np.concatenate(arr_list, axis=0) if arr_list else np.array([])

                    # Save Merged Static
                    if acc_static_pos:
                        print("Saving Merged Static Scene...")
                        final_st_pos = merge(acc_static_pos)
                        final_st_scl = merge(acc_static_scl)
                        final_st_rot = merge(acc_static_rot)
                        final_st_opa = merge(acc_static_opa)
                        final_st_col = merge(acc_static_col)

                        save_ply(
                            torch.tensor(final_st_pos), torch.tensor(final_st_scl),
                            torch.tensor(final_st_rot), torch.tensor(final_st_opa),
                            torch.tensor(final_st_col),
                            os.path.join(gs_dir, "static_scene.ply")
                        )

                    # Save Merged Sky
                    if acc_sky_pos:
                        print("Saving Merged Sky Scene...")
                        final_sk_pos = merge(acc_sky_pos)
                        final_sk_scl = merge(acc_sky_scl)
                        final_sk_rot = merge(acc_sky_rot)
                        final_sk_opa = merge(acc_sky_opa)
                        final_sk_col = merge(acc_sky_col)

                        save_ply(
                            torch.tensor(final_sk_pos), torch.tensor(final_sk_scl),
                            torch.tensor(final_sk_rot), torch.tensor(final_sk_opa),
                            torch.tensor(final_sk_col),
                            os.path.join(gs_dir, "sky_scene.ply")
                        )

                    # Setup for Rendering (Standard Logic for validation)
                    static_mask = (bg_mask & (dy_map < 0.5))
                    static_points = point_map[static_mask].reshape(-1, 3)
                    gs_dynamic_list = dy_map[static_mask].sigmoid()
                    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                    static_opacity = static_opacity * (1 - gs_dynamic_list)
                    static_gs_conf = gs_conf[static_mask]
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
                    gs_timestamps = timestamps[frame_idx]

                # --- RENDERER SETUP (Original Logic) ---
                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]

                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i],
                                                                                                 bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i

                    dynamic_points.append(dynamic_point)
                    dynamic_rgbs.append(dynamic_rgb)
                    dynamic_opacitys.append(dynamic_opacity)
                    dynamic_scales.append(dynamic_scale)
                    dynamic_rotations.append(dynamic_rotation)

                chunked_renders, chunked_alphas = [], []
                if args.mode == 3:
                    origin_extrinsic = extrinsic
                    origin_intrinsic = intrinsic

                    # Rendering Loop
                for idx in range(dy_map.shape[1]):
                    if args.mode == 3:
                        I = intervals
                        t0 = timestamps[idx // I]
                        static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=gs_conf, gamma1=0.1)

                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            [static_points, static_rgbs, static_opacity_, static_scales, static_rotation],
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx],
                             dynamic_rotations[idx]]
                        )
                        renders_chunk, alphas_chunk, _ = rasterization(
                            means=world_points,
                            quats=rotation,
                            scales=scales,
                            opacities=opacity,
                            colors=rgbs,
                            viewmats=extrinsic[idx:idx + 1],
                            Ks=intrinsic[idx:idx + 1],
                            width=W,
                            height=H,
                            render_mode='RGB+ED',
                        )
                    if args.mode == 2:
                        t0 = timestamps[idx]
                        static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
                        static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                        if dynamic_points:
                            world_points, rgbs, opacity, scales, rotation = concat_list(
                                static_gs_list,
                                [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx],
                                 dynamic_rotations[idx]]
                            )
                        else:
                            world_points, rgbs, opacity, scales, rotation = static_gs_list

                        renders_chunk, alphas_chunk, _ = rasterization(
                            means=world_points,
                            quats=rotation,
                            scales=scales,
                            opacities=opacity,
                            colors=rgbs,
                            viewmats=extrinsic[idx:idx + 1],
                            Ks=intrinsic[idx:idx + 1],
                            width=W,
                            height=H,
                            render_mode='RGB+ED',
                        )
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)

                renders = torch.cat(chunked_renders, dim=0)
                depth_frames = renders[..., -1]  # Save depth for comparison video
                renders = renders[..., :-1]
                alphas = torch.cat(chunked_alphas, dim=0)

                if args.mode == 3:
                    bg_render = model.sky_model.forward_with_new_pose(images, origin_extrinsic, origin_intrinsic,
                                                                      extrinsic, intrinsic)
                if args.mode == 2:
                    bg_render = model.sky_model(images, extrinsic, intrinsic)
                    bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)

                renders = alphas * renders + (1 - alphas) * bg_render
                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]

            # Post-Processing
            scene_name = str(scene_idx).zfill(3)
            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)

            if args.diffusion:
                processed_frames = []
                for i in range(rendered_image.shape[0]):
                    frame = rendered_image[i].detach().cpu().clamp(0, 1)
                    processed_frame = process_images_with_difix(frame, "pretrained/diffusion_model.pth")
                    processed_frames.append(processed_frame)
                rendered_image = torch.stack(processed_frames, dim=0).to(device)

            psnr, ssim, lpip = compute_metrics(rendered_image, target_image, loss_fn)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpip)
            scene_idx += 1

            scene_out_dir = os.path.join(args.output_path, scene_name)
            os.makedirs(scene_out_dir, exist_ok=True)

            if args.images:
                if args.input_views == 1:
                    image_list = []
                    for i in range(rendered_image.shape[0]):
                        rendered = rendered_image[i].detach().cpu().clamp(0, 1)
                        image_path = os.path.join(scene_out_dir, f"view_{i}.png")
                        T.ToPILImage()(rendered).save(image_path)
                        image_list.append(rendered.permute(1, 2, 0).numpy())
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, (np.array(image_list) * 255).astype(np.uint8), fps=8, codec="libx264")
                if args.input_views == 3:
                    T_total = rendered_image.shape[0]
                    groups = T_total // 3
                    video_list = []
                    for g in range(groups):
                        idx_center = 3 * g + 0
                        idx_left = 3 * g + 1
                        idx_right = 3 * g + 2
                        center = rendered_image[idx_center].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        left = rendered_image[idx_left].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        right = rendered_image[idx_right].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        H, W = center.shape[0], center.shape[1]

                        def to_uint8(arr):
                            a = (arr * 255.0).astype(np.uint8)
                            if a.ndim == 2:
                                a = np.stack([a] * 3, axis=-1)
                            if a.shape[2] == 4:
                                a = a[:, :, :3]
                            return a

                        left_u = to_uint8(left)
                        center_u = to_uint8(center)
                        right_u = to_uint8(right)
                        white = np.ones((H, 10, 3), dtype=np.uint8) * 255
                        composed = np.concatenate([left_u, white, center_u, white, right_u], axis=1)
                        image_path = os.path.join(scene_out_dir, f"view_{g:04d}.png")
                        Image.fromarray(composed).save(image_path)
                        video_list.append(composed)
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, np.array(video_list), fps=8, codec="libx264")

            gt_frames = target_image.detach().cpu()
            pred_frames = rendered_image.detach().cpu()
            dyn_frames = dy_map[0].sigmoid().detach().cpu()
            gt_dy_map = gt_dy_map.mean(dim=2)
            gt_dy_map = gt_dy_map[0].sigmoid().detach().cpu()
            if args.mode == 2:
                depth_frames = depth_frames.detach().cpu()  # Use accumulated depth from renders
                gt_depth = gt_depth[..., 0:1]
                gt_depth = gt_depth[0].squeeze(-1).detach().cpu()
                sky_mask = sky_mask.detach().cpu()
            if args.mode == 3:
                depth_frames = depth_interp[0].detach().cpu()
                gt_depth = gt_depth[..., 0:1]
                gt_depth = gt_depth[0].squeeze(-1).detach().cpu()
                sky_mask = target_sky_masks.permute(0, 1, 3, 4, 2).detach().cpu()
            out_video = os.path.join(scene_out_dir, "comparison.mp4")
            make_comparison_video_quad(gt_frames, pred_frames, gt_dy_map, dyn_frames, gt_depth, depth_frames, sky_mask,
                                       out_video, fps=8, views=args.input_views)  #
            print("Saved comparison video:", out_video)

            if args.depth:
                S = depth_frames.shape[0]
                if args.input_views == 1:
                    for i in range(S):
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{i}.npy"), depth_i)
                elif args.input_views == 3:
                    for i in range(S):
                        view_id = i % 3
                        frame_id = i // 3
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{frame_id:04d}_{view_id}.npy"), depth_i)
    if args.metrics:
        print("PSNR:", sum(psnr_list) / len(psnr_list))
        print("SSIM:", sum(ssim_list) / len(ssim_list))
        print("LPIPS:", sum(lpips_list) / len(lpips_list))
        print("Avg Inference Time (s):", sum(inference_time_list) / len(inference_time_list))

if __name__ == "__main__":
    main()