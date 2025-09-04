#!/usr/bin/env python3
"""
Complete Gaussian Splatting Pipeline Runner
Ready-to-run script for ScanNet GS data processing and inference

Usage:
    # List available scenes (NPY format)
    python run_gs_pipeline.py --npy_folder /path/to/scannet_gs --list_scenes
    
    # Run inference on NPY scene
    python run_gs_pipeline.py --npy_folder /path/to/scannet_gs --scene_name scene_00001
    
    # Run inference on PLY files
    python run_gs_pipeline.py --ply /path/to/scene.ply --save_features
    
    # List available scenes (NPZ format - processed ScanNet)
    python run_gs_pipeline.py --npz_folder /path/to/processed_scannet --list_scenes
    
    # Run inference on NPZ scene
    python run_gs_pipeline.py --npz_folder /path/to/processed_scannet --scene_name scene0000_00
    
    # Process all ScanNet scenes and save Gaussians + features
    python run_gs_pipeline.py --scannet_folder /path/to/scannet_gs --process_all_scenes --save_per_gaussian --save_sparse
    
    # List available ScanNet scenes
    python run_gs_pipeline.py --scannet_folder /path/to/scannet_gs --list_scenes
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import plyfile
import math
from pathlib import Path
from typing import Dict, List, Any, Optional


# Add the project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import required modules
from scenesplat.scenesplat import PointTransformerV3
from scenesplat.serialization import decode
from scenesplat.utils.transform import Compose
from dataloader.scannet_loader import ScanNetGaussianLoader
from tqdm import tqdm
import time

def SH2RGB(sh):
    """Convert SH DC coefficients to RGB [0,1] using viewer's method"""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

class GaussianSplatDataLoader:
    """
    Loads and processes Gaussian Splatting data from PLY, NPY, or NPZ format
    """
    
    def __init__(self, data_path: str, use_normal: bool = True, data_type: str = "npy", sample_num: int = 100_000_000):
        self.data_path = Path(data_path)
        self.use_normal = use_normal
        self.data_type = data_type  # "npy", "ply", or "npz"
        self.sample_num = sample_num
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Initialize transforms based on the config
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        """Get the transform pipeline from config"""
        # Define keys based on normal usage
        keys = ["coord", "color", "opacity", "quat", "scale", "segment"]
        feat_keys = ["color", "opacity", "quat", "scale"]
        if self.use_normal:
            keys.append("normal")
            feat_keys.append("normal")
        
        return Compose([
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",  # Use test mode for inference
                keys=tuple(keys),
                return_inverse=True,
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "inverse"),
                feat_keys=tuple(feat_keys),
            ),
        ])
    
    def list_scenes(self) -> List[str]:
        """List available scenes in the data directory"""
        if self.data_type == "ply":
            return [self.data_path.stem]  # Single PLY file
        elif self.data_type == "npz":
            # List NPZ files directly
            npz_files = list(self.data_path.glob("scene*.npz"))
            return [f.stem for f in sorted(npz_files)]
        else:  # npy format
            scenes = []
            for split in ["train", "val", "test"]:
                split_path = self.data_path / split
                if split_path.exists():
                    for scene_dir in split_path.iterdir():
                        if scene_dir.is_dir():
                            scenes.append(f"{split}/{scene_dir.name}")
            return sorted(scenes)
    
    def _load_ply_data(self) -> Dict[str, np.ndarray]:
        """Load PLY file and extract Gaussian parameters"""
        print(f"Loading PLY file: {self.data_path}")
        
        ply_data = plyfile.PlyData.read(str(self.data_path))
        vertex = ply_data["vertex"]
        
        # Extract coordinates
        coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        
        # Extract DC features (RGB color)
        features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        # Convert SH DC to RGB [0,1] then to [0,255]
        color = SH2RGB(features_dc) * 255.0
        color = np.clip(color, 0, 255)
        
        # Extract opacity
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        
        # Extract quaternion (rot_0, rot_1, rot_2, rot_3 -> wxyz)
        quat = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)
        
        # Extract scales
        scale = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
        if self.sample_num is not None and self.sample_num < coord.shape[0]:
            random_idx = np.random.choice(coord.shape[0], size=self.sample_num, replace=False)
        else:
            random_idx = np.arange(coord.shape[0])
        coord = coord[random_idx]
        color = color[random_idx]
        opacity = opacity[random_idx]
        quat = quat[random_idx]
        scale = scale[random_idx]
        
        data_dict = {
            "coord": coord,
            "color": color,
            "opacity": opacity,
            "quat": quat,
            "scale": scale,
        }
        
        # Extract normals if available and requested
        if self.use_normal:
            if all(f"n{axis}" in vertex for axis in ["x", "y", "z"]):
                normal = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
                data_dict["normal"] = normal
                print(f"  Loaded normals: {normal.shape}")
            else:
                # Create dummy normals
                num_points = coord.shape[0]
                data_dict["normal"] = np.zeros((num_points, 3), dtype=np.float32)
                print(f"  Created dummy normals: {data_dict['normal'].shape}")
        
        # Create dummy segmentation for consistency
        num_points = coord.shape[0]
        data_dict["segment"] = np.ones(num_points, dtype=np.int32) * -1
        
        print(f"  Loaded coord: {coord.shape}")
        print(f"  Loaded color: {color.shape}")
        print(f"  Loaded opacity: {opacity.shape}")
        print(f"  Loaded quat: {quat.shape}")
        print(f"  Loaded scale: {scale.shape}")
        
        return data_dict
    
    def _load_npy_data(self, scene_path: str) -> Dict[str, np.ndarray]:
        """Load all .npy files for a scene"""
        scene_dir = self.data_path / scene_path
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory {scene_dir} not found")
        
        print(f"Loading scene: {scene_path}")
        
        # Required GS attributes
        required_files = ["coord.npy", "color.npy", "opacity.npy", "quat.npy", "scale.npy"]
        optional_files = ["normal.npy", "segment.npy", "instance.npy"]
        
        data_dict = {}
        
        # Load required files
        for file_name in required_files:
            file_path = scene_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file {file_path} not found")
            data_dict[file_name[:-4]] = np.load(file_path)
            print(f"  Loaded {file_name}: {data_dict[file_name[:-4]].shape}")
         
        # Load optional files
        for file_name in optional_files:
            file_path = scene_dir / file_name
            if file_path.exists():
                data_dict[file_name[:-4]] = np.load(file_path)
                print(f"  Loaded {file_name}: {data_dict[file_name[:-4]].shape}")

        return data_dict
    
    
    def process_scene(self, scene_path: str = None) -> Dict[str, torch.Tensor]:
        """Complete processing pipeline for a scene"""
        # Load raw data based on data type
        if self.data_type == "ply":
            raw_data = self._load_ply_data()
        elif self.data_type == "npz":
            if scene_path is None:
                raise ValueError("scene_path required for NPZ format")
            raw_data = self._load_npz_data(scene_path)
        else:  # npy
            if scene_path is None:
                raise ValueError("scene_path required for NPY format")
            raw_data = self._load_npy_data(scene_path)
        
        # Apply transforms directly to raw data (preprocessing is handled by transforms)
        data = self.transform(raw_data)
        return data


class SceneSplat:
    """
    Wrapper for PointTransformerV3 model with GS-specific processing
    """
    
    def __init__(self, model_folder: str, device: str = "cuda", use_normal: bool = True, save_sparse: bool = False):
        self.device = device
        self.model_folder = Path(model_folder)
        self.use_normal = use_normal
        self.model = self._load_model(save_sparse=save_sparse)
        
    def _get_model_config(self):
        """Load model configuration from config_inference.py"""
        config_path = self.model_folder / "config_inference.py"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config from file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        config = config_module.config.copy()
        # Adjust input channels based on normal usage
        if not self.use_normal:
            config["in_channels"] = 11  # 3+1+4+3 (color+opacity+quat+scale, no normal)
        return config
    
    def _load_model(self, save_sparse: bool = False):
        """Load the PointTransformerV3 model"""
        # Find checkpoint file in model folder
        checkpoint_files = list(self.model_folder.glob("*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No .pth checkpoint found in {self.model_folder}")
        checkpoint_path = checkpoint_files[0]  # Use first .pth file found
        
        print(f"Loading model from: {checkpoint_path}")
        print(f"Using config from: {self.model_folder / 'config_inference.py'}")
        
        # Create model with config
        config = self._get_model_config()
        model = PointTransformerV3(**config, save_sparse=save_sparse)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        
        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        
        # Remap keys to match local model (strip distributed/backbone prefixes)
        def strip_prefix(k: str) -> str:
            for p in ("module.backbone.", "backbone.", "module."):
                if k.startswith(p):
                    return k[len(p):]
            return k
        remapped = {strip_prefix(k): v for k, v in state_dict.items()}
        
        # Load weights with remapped keys
        missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        return model
    
    def prepare_input(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare input for the model"""
        # Extract required tensors (keep 2D format for Point structure)
        coord = data_dict["coord"].float()
        grid_coord = data_dict["grid_coord"].int()  # int32 for spconv - Disabled with GridSample
        feat = data_dict["feat"].float()
        
        # Create batch tensor (assuming batch_size=1)
        batch = torch.zeros(coord.shape[0], dtype=torch.long)
        # grid_coord = coord.int()
        model_input = {
            "coord": coord.to(self.device),
            "grid_coord": grid_coord.to(self.device),
            "feat": feat.to(self.device),
            "batch": batch.to(self.device),
        }
        
        print(f"Model input shapes:")
        for key, value in model_input.items():
            print(f"  {key}: {value.shape} ({value.dtype})")
        
        return model_input
    
    @torch.no_grad()
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference and return deserialized results"""
        model_input = self.prepare_input(data_dict)
        print("Running model inference...")
        point_features = self.model(model_input)
        
        # print(f"Output features shape: {point_features.feat.shape}")
        
        # Reverse serialization to get original point ordering
        # Based on SceneSplat's pointcept/engines/test.py approach
        return point_features


def main():
    parser = argparse.ArgumentParser(description="Run Gaussian Splatting Pipeline")
    parser.add_argument("--npy_folder", type=str, 
                        help="Root directory containing NPY GS data")
    parser.add_argument("--ply", type=str,
                        help="Path to PLY file containing GS data")
    parser.add_argument("--npz_folder", type=str,
                        help="Root directory containing NPZ GS data (processed ScanNet)")
    parser.add_argument("--scannet_folder", type=str,
                        help="Root directory containing ScanNet Gaussian PLY files")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Specific scene to process (e.g., 'train/scene_00001' for NPY, 'scene0000_00' for NPZ)")
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Path to model folder containing checkpoint and config_inference.py")
    parser.add_argument("--normal", action="store_true", 
                        help="Include normal vectors in features (default: False)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--list_scenes", action="store_true",
                        help="List available scenes and exit (for NPY and NPZ formats)")
    parser.add_argument("--save_features", action="store_true",
                        help="Save extracted features to file")
    parser.add_argument("--save_output", action="store_true",
                        help="Save coord/attributes and features to files")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output features")
    parser.add_argument("--sample_num", type=int, default=100_000_000,
                        help="Sample number of points")
    parser.add_argument("--save_sparse", action="store_true",
                        help="Save sparse features")
    parser.add_argument("--save_per_gaussian", action="store_true",
                        help="Save processed Gaussians as NPZ files (one per scene)")
    parser.add_argument("--process_all_scenes", action="store_true",
                        help="Process all scenes in ScanNet folder")
    args = parser.parse_args()
    
    # Validate input arguments
    input_args = [args.ply, args.npy_folder, args.npz_folder, args.scannet_folder]
    if sum(x is not None for x in input_args) != 1:
        parser.error("Exactly one of --ply, --npy_folder, --npz_folder, or --scannet_folder must be specified")
    
    # Determine data type and initialize appropriate loader
    if args.ply:
        print("Using PLY input mode")
        data_loader = GaussianSplatDataLoader(args.ply, use_normal=args.normal, data_type="ply", sample_num=args.sample_num)
        
        if args.list_scenes:
            print("--list_scenes not supported for PLY input")
            return
        
        processed_data = data_loader.process_scene()
        scene_name = Path(args.ply).stem  # Use PLY filename as scene name
        
    elif args.npz_folder:
        print("Using NPZ folder input mode")
        data_loader = GaussianSplatDataLoader(args.npz_folder, use_normal=args.normal, data_type="npz", sample_num=args.sample_num)
        
        # List scenes if requested
        if args.list_scenes:
            scenes = data_loader.list_scenes()
            print(f"Available scenes ({len(scenes)}):")
            for scene in scenes[:20]:  # Show first 20
                print(f"  {scene}")
            if len(scenes) > 20:
                print(f"  ... and {len(scenes) - 20} more")
            return
        
        # Determine scene to process
        if args.scene_name is None:
            scenes = data_loader.list_scenes()
            if not scenes:
                print("No scenes found in data directory")
                return
            scene_name = scenes[0]  # Use first available scene
            print(f"No scene specified, using: {scene_name}")
        else:
            scene_name = args.scene_name
        
        processed_data = data_loader.process_scene(scene_name)
    
    elif args.npy_folder:
        print("Using NPY folder input mode")
        data_loader = GaussianSplatDataLoader(args.npy_folder, use_normal=args.normal, data_type="npy", sample_num=args.sample_num)
        
        # List scenes if requested
        if args.list_scenes:
            scenes = data_loader.list_scenes()
            print(f"Available scenes ({len(scenes)}):")
            for scene in scenes[:20]:  # Show first 20
                print(f"  {scene}")
            if len(scenes) > 20:
                print(f"  ... and {len(scenes) - 20} more")
            return
        
        # Determine scene to process
        if args.scene_name is None:
            scenes = data_loader.list_scenes()
            if not scenes:
                print("No scenes found in data directory")
                return
            scene_name = scenes[0]  # Use first available scene
            print(f"No scene specified, using: {scene_name}")
        else:
            scene_name = args.scene_name
        
        processed_data = data_loader.process_scene(scene_name)
    
    elif args.scannet_folder:
        print("Using ScanNet folder input mode")
        scannet_loader = ScanNetGaussianLoader(args.scannet_folder)
        
        # List scenes if requested
        if args.list_scenes:
            scenes = scannet_loader.get_scene_list()
            print(f"Available scenes ({len(scenes)}):")
            for scene in scenes[:20]:  # Show first 20
                print(f"  {scene}")
            if len(scenes) > 20:
                print(f"  ... and {len(scenes) - 20} more")
            return
        
        # Process all scenes if requested
        if args.process_all_scenes:
            scenes = scannet_loader.get_scene_list()
            print(f"Processing all {len(scenes)} scenes...")
            
            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize model once
            model = SceneSplat(args.model_folder, args.device, use_normal=args.normal, save_sparse=args.save_sparse)
            
            # Process each scene
            for i, scene_name in enumerate(tqdm(scenes, desc="Processing scenes")):
                # Load Gaussians
                try:
                    data_time_start = time.time()
                    raw_data = scannet_loader.load_scene_gaussians(scene_name)
                    # Create a data loader for transforms
                    data_loader = GaussianSplatDataLoader(str(scannet_loader.scannet_folder), use_normal=args.normal, data_type="ply", sample_num=args.sample_num)
                    # Apply transforms directly to raw data
                    data = data_loader.transform(raw_data)
                    data_time_end = time.time()
                    # Run model inference
                    model_time_start = time.time()
                    result = model.forward(data)
                    model_time_end = time.time()

                    # Save results
                    save_time_start = time.time()
                    scene_output_dir = output_dir / "scannet" / scene_name
                    scene_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    if args.save_sparse:
                        feat = result["feat"]
                        feat = torch.nn.functional.normalize(feat, dim=1, p=2).to(torch.float16)
                        coord = result["coord"]
                        code = result["code"]
                        
                        save_npz = {
                            "feat": feat.detach().cpu().numpy(),
                            "coord": coord.detach().cpu().numpy(),
                            "code": code.detach().cpu().numpy(),
                        }
                        features_file = scene_output_dir / "features.npz"
                        np.savez_compressed(features_file, **save_npz)
                        print(f" Saved features to: {features_file}, feat shape: {feat.shape}, coord shape: {coord.shape}, code shape: {code.shape}")
                    save_time_end = time.time()
                    print(f"Data time: {data_time_end - data_time_start}, Model time: {model_time_end - model_time_start}, Save time: {save_time_end - save_time_start}")
                except Exception as e:
                    print(f"Error processing {scene_name}: {e}")
                    continue
                    

            
            print(f"\nCompleted processing all scenes. Results saved to: {output_dir}")
            return
        

if __name__ == "__main__":
    main() 