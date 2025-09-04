#!/usr/bin/env python3
"""
ScanNet Gaussian Splatting Loader

Loads all Gaussians from ScanNet folder and processes them for inference.
"""

import os
import numpy as np
import plyfile
from pathlib import Path
from typing import Dict, List


def SH2RGB(sh):
    """Convert SH DC coefficients to RGB [0,1] using viewer's method"""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


class ScanNetGaussianLoader:
    """
    Loads all Gaussians from ScanNet folder and processes them
    """
    
    def __init__(self, scannet_folder: str):
        self.scannet_folder = Path(scannet_folder)
        if not self.scannet_folder.exists():
            raise FileNotFoundError(f"ScanNet folder not found: {self.scannet_folder}")
    
    def get_scene_list(self):
        """Get list of all scenes with PLY files"""
        scenes = []
        for item in self.scannet_folder.iterdir():
            if item.is_dir() and item.name.startswith('scene'):
                ply_path = item / 'ckpts' / 'point_cloud_30000.ply'
                if ply_path.exists():
                    scenes.append(item.name)
        return sorted(scenes)
    
    def load_scene_gaussians(self, scene_name: str):
        """Load Gaussians from a single scene PLY file"""
        ply_path = self.scannet_folder / scene_name / 'ckpts' / 'point_cloud_30000.ply'
        
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        print(f"Loading Gaussians from: {scene_name}")
        
        # Read PLY file
        ply_data = plyfile.PlyData.read(str(ply_path))
        vertex = ply_data["vertex"]
        
        # Extract coordinates (positions)
        coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        
        # Extract DC features (RGB color) and convert properly
        features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        color = SH2RGB(features_dc) * 255.0
        color = np.clip(color, 0, 255).astype(np.float32)
        
        # Extract opacity and apply sigmoid
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        opacity = 1 / (1 + np.exp(-opacity))
        
        # Extract quaternions and normalize
        quat = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)
        quat_norms = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / (quat_norms + 1e-8)
        
        # Extract scales and apply exp + clip
        scale = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
        scale = np.exp(scale)
        scale = np.clip(scale, 0, 1.5)
        
        # Extract normals if available
        normal = None
        try:
            normal = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
            normal_norms = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal / (normal_norms + 1e-8)
        except (ValueError, KeyError):
            normal = None
        
        # Create data dictionary
        data_dict = {
            'coord': coord,
            'color': color,
            'opacity': opacity,
            'quat': quat,
            'scale': scale,
        }
        
        if normal is not None:
            data_dict['normal'] = normal
        
        # Create segmentation (required for transforms)
        num_points = coord.shape[0]
        data_dict['segment'] = np.ones(num_points, dtype=np.int32) * -1
        
        print(f"  Loaded {num_points} Gaussians")
        print(f"  Color range: [{color.min():.2f}, {color.max():.2f}]")
        print(f"  Opacity range: [{opacity.min():.4f}, {opacity.max():.4f}]")
        
        return data_dict
