import json
import numpy as np
import torch
from plyfile import PlyData, PlyElement

def save_json(data, path):
    def convert(o):
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, torch.Tensor): return o.detach().cpu().numpy().tolist()
        raise TypeError
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=convert)

def save_ply(positions, scales, rotations, opacities, colors, path, object_ids=None):
    """
    Saves 3D Gaussians to .ply. 
    If object_ids is provided, adds an 'object_id' property to vertices.
    """
    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    xyz = to_np(positions)
    scale = to_np(scales)
    rot = to_np(rotations)
    opac = to_np(opacities)
    col = to_np(colors)
    
    N = xyz.shape[0]
    
    # Base attributes
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    # Add object_id if dynamic
    if object_ids is not None:
        obj_ids = to_np(object_ids)
        dtype_list.append(('object_id', 'i4'))

    elements = np.empty(N, dtype=dtype_list)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = np.zeros(N)
    elements['ny'] = np.zeros(N)
    elements['nz'] = np.zeros(N)
    elements['f_dc_0'] = col[:, 0]
    elements['f_dc_1'] = col[:, 1]
    elements['f_dc_2'] = col[:, 2]
    elements['opacity'] = opac.squeeze()
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    elements['rot_0'] = rot[:, 0]
    elements['rot_1'] = rot[:, 1]
    elements['rot_2'] = rot[:, 2]
    elements['rot_3'] = rot[:, 3]

    if object_ids is not None:
        elements['object_id'] = obj_ids

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def voxel_downsample(positions, scales, rotations, opacities, colors, voxel_size=0.05):
    """
    Simple voxel downsampling to reduce file size of accumulated point clouds.
    Keeps the first point found in each voxel.
    """
    if len(positions) == 0:
        return positions, scales, rotations, opacities, colors

    # Quantize positions to integers
    quantized = np.floor(positions / voxel_size).astype(np.int32)

    # Use a dictionary or unique to find unique voxels
    # np.unique with axis=0 is slow, using a structured array or packing bits is faster,
    # but for simplicity/readability, we use a basic approach.

    # Pack coordinates into a single 64-bit integer for hashing (assuming range fits)
    # This is a fast way to find unique indices in 3D
    key = ((quantized[:, 0] & 0xFFFFF) << 40) | \
          ((quantized[:, 1] & 0xFFFFF) << 20) | \
          ((quantized[:, 2] & 0xFFFFF))

    _, unique_indices = np.unique(key, return_index=True)

    return (positions[unique_indices], scales[unique_indices],
            rotations[unique_indices], opacities[unique_indices],
            colors[unique_indices])