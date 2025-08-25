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

def SH2RGB(sh):
    """Convert SH DC coefficients to RGB [0,1] using viewer's method"""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

class GaussianSplatDataLoader:
    """
    Loads and processes Gaussian Splatting data from either PLY or NPY format
    """
    
    def __init__(self, data_path: str, use_normal: bool = True, data_type: str = "npy", sample_num: int = 100_000_000):
        self.data_path = Path(data_path)
        self.use_normal = use_normal
        self.data_type = data_type  # "npy" or "ply"
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
                grid_size=0.01,
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
        """List available scenes in the data directory (NPY format only)"""
        if self.data_type == "ply":
            return [self.data_path.stem]  # Single PLY file
        
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
         
        if self.sample_num is not None:
            random_idx = np.random.choice(data_dict["coord"].shape[0], size=self.sample_num, replace=False)
            for key in data_dict:
                data_dict[key] = data_dict[key][random_idx]
        
        return data_dict
    
    def preprocess_data(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess raw GS data (handles both PLY and NPY)"""
        processed = data_dict.copy()
        
        # Ensure correct data types and shapes
        processed["coord"] = processed["coord"].astype(np.float32)
        processed["color"] = processed["color"].astype(np.float32)
        
        # Ensure color is in [0, 255] range
        if processed["color"].max() <= 1.0:
            print("  Scaling colors from [0,1] to [0,255]")
            processed["color"] = processed["color"] * 255.0
        
        # Process opacity
        processed["opacity"] = processed["opacity"].astype(np.float32)
        if processed["opacity"].ndim == 1:
            processed["opacity"] = processed["opacity"].reshape(-1, 1)
        
        # Apply sigmoid for PLY data (raw opacity), skip for NPY (already processed)
        if self.data_type == "ply":
            processed["opacity"] = torch.sigmoid(torch.from_numpy(processed["opacity"])).numpy()
        
        # Normalize quaternions
        processed["quat"] = processed["quat"].astype(np.float32)
        quat_norms = np.linalg.norm(processed["quat"], axis=1, keepdims=True)
        processed["quat"] = processed["quat"] / (quat_norms + 1e-8)
        
        # Clip scales
        processed["scale"] = processed["scale"].astype(np.float32)
        # Apply exp for PLY data (raw scales), clip for NPY (already processed)
        if self.data_type == "ply":
            processed["scale"] = np.exp(processed["scale"])
        processed["scale"] = np.clip(processed["scale"], 0, 1.5)
        
        # Process normal if available
        if "normal" in processed and self.use_normal:
            processed["normal"] = processed["normal"].astype(np.float32)
        elif self.use_normal:
            # Create dummy normals if not available
            num_points = processed["coord"].shape[0]
            processed["normal"] = np.zeros((num_points, 3), dtype=np.float32)
            print("  Created dummy normals")
        # If use_normal is False, don't include normal at all
        
        # Process segmentation if available
        if "segment" in processed:
            processed["segment"] = processed["segment"].reshape(-1).astype(np.int32)
        else:
            # Create dummy segmentation
            num_points = processed["coord"].shape[0]
            processed["segment"] = np.ones(num_points, dtype=np.int32) * -1
            print("  Created dummy segmentation")
        
        return processed
    
    def process_scene(self, scene_path: str = None) -> Dict[str, torch.Tensor]:
        """Complete processing pipeline for a scene"""
        # Load raw data based on data type
        if self.data_type == "ply":
            raw_data = self._load_ply_data()
        else:  # npy
            if scene_path is None:
                raise ValueError("scene_path required for NPY format")
            raw_data = self._load_npy_data(scene_path)
        
        # Preprocess
        processed_data = self.preprocess_data(raw_data)
        
        # Apply transforms to get grid-sampled data and grid_coord
        data = self.transform(processed_data)
        return data


class SceneSplat:
    """
    Wrapper for PointTransformerV3 model with GS-specific processing
    """
    
    def __init__(self, model_folder: str, device: str = "cuda", use_normal: bool = True):
        self.device = device
        self.model_folder = Path(model_folder)
        self.use_normal = use_normal
        self.model = self._load_model()
        
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
    
    def _load_model(self):
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
        model = PointTransformerV3(**config)
        
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
        
        print(f"Output features shape: {point_features.feat.shape}")
        
        # Reverse serialization to get original point ordering
        # Based on SceneSplat's pointcept/engines/test.py approach
        return point_features


def main():
    parser = argparse.ArgumentParser(description="Run Gaussian Splatting Pipeline")
    parser.add_argument("--npy_folder", type=str, 
                        help="Root directory containing NPY GS data")
    parser.add_argument("--ply", type=str,
                        help="Path to PLY file containing GS data")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Specific scene to process (e.g., 'train/scene_00001') - only for NPY format")
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Path to model folder containing checkpoint and config_inference.py")
    parser.add_argument("--normal", action="store_true", 
                        help="Include normal vectors in features (default: False)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--list_scenes", action="store_true",
                        help="List available scenes and exit (only for NPY format)")
    parser.add_argument("--save_features", action="store_true",
                        help="Save extracted features to file")
    parser.add_argument("--save_output", action="store_true",
                        help="Save coord/attributes and features to files")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output features")
    parser.add_argument("--sample_num", type=int, default=100_000_000,
                        help="Sample number of points")
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.ply and not args.npy_folder:
        parser.error("Either --ply or --npy_folder must be specified")
    
    if args.ply and args.npy_folder:
        parser.error("Cannot specify both --ply and --npy_folder")
    
    # Determine data type and initialize appropriate loader
    if args.ply:
        print("Using PLY input mode")
        data_loader = GaussianSplatDataLoader(args.ply, use_normal=args.normal, data_type="ply", sample_num=args.sample_num)
        
        if args.list_scenes:
            print("--list_scenes not supported for PLY input")
            return
        
        processed_data = data_loader.process_scene()
        scene_name = Path(args.ply).stem  # Use PLY filename as scene name
        
    else:
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
    
    # Run model inference
    model = SceneSplat(args.model_folder, args.device, use_normal=args.normal)
    data_dict = model.forward(processed_data)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Successfully processed scene: {scene_name}")
    print(f"Input points: {processed_data['coord'].shape[0]}")
    
    # Process and save features
    feat = data_dict["feat"]
    feat = torch.nn.functional.normalize(feat, dim=1, p=2).to(torch.float16)
    
    # Save features and/or outputs if requested
    if args.save_features:
        inverse = processed_data['inverse']
        feat = feat[inverse]
        feat_to_save = feat.cpu().numpy()
        save_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(save_dir, exist_ok=True)

        feat_path = os.path.join(save_dir, f"pred_langfeat.npy")
        np.save(feat_path, feat_to_save)
        print(f"Saved features to: {feat_path} (shape: {feat_to_save.shape})")

if __name__ == "__main__":
    main() 