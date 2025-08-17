#!/usr/bin/env python3
"""
Complete Gaussian Splatting Pipeline Runner
Ready-to-run script for ScanNet GS data processing and inference

Usage:
    # List available scenes
    python run_gs_pipeline.py --data_root /path/to/scannet_gs --list_scenes
    
    # Run inference on a specific scene
    python run_gs_pipeline.py --data_root /path/to/scannet_gs --scene_name scene_00001
    
    # Run inference and save features with deserialization
    python run_gs_pipeline.py --data_root /path/to/scannet_gs --scene_name scene_00001 --save_features --output_dir ./features
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional


# Add the project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import required modules
from scenesplat.scenesplat import PointTransformerV3
from scenesplat.serialization import decode
from scenesplat.utils.transform import Compose

class GaussianSplatDataLoader:
    """
    Loads and processes Gaussian Splatting data for a single scene
    """
    
    def __init__(self, data_root: str, scene_name: str = None):
        self.data_root = Path(data_root)
        self.scene_name = scene_name
        
        # Initialize transforms based on the config
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        """Get the transform pipeline from config"""
        return Compose([
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="test",  # Use test mode for inference
                keys=("coord", "color", "opacity", "quat", "scale", "segment", "normal"),
                return_inverse=True,
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "inverse"),
                feat_keys=("color", "opacity", "quat", "scale", "normal"),
            ),
        ])
    
    def list_scenes(self) -> List[str]:
        """List available scenes in the data directory"""
        scenes = []
        for split in ["train", "val", "test"]:
            split_path = self.data_root / split
            if split_path.exists():
                for scene_dir in split_path.iterdir():
                    if scene_dir.is_dir():
                        scenes.append(f"{split}/{scene_dir.name}")
        return sorted(scenes)
    
    def load_scene_data(self, scene_path: str) -> Dict[str, np.ndarray]:
        """Load all .npy files for a scene"""
        scene_dir = self.data_root / scene_path
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
    
    def preprocess_data(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess raw GS data"""
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
        
        # Normalize quaternions
        processed["quat"] = processed["quat"].astype(np.float32)
        quat_norms = np.linalg.norm(processed["quat"], axis=1, keepdims=True)
        processed["quat"] = processed["quat"] / (quat_norms + 1e-8)
        
        # Clip scales
        processed["scale"] = processed["scale"].astype(np.float32)
        processed["scale"] = np.clip(processed["scale"], 0, 1.5)
        
        # Process normal if available
        if "normal" in processed:
            processed["normal"] = processed["normal"].astype(np.float32)
        else:
            # Create dummy normals if not available
            num_points = processed["coord"].shape[0]
            processed["normal"] = np.zeros((num_points, 3), dtype=np.float32)
            print("  Created dummy normals")
        
        # Process segmentation if available
        if "segment" in processed:
            processed["segment"] = processed["segment"].reshape(-1).astype(np.int32)
        else:
            # Create dummy segmentation
            num_points = processed["coord"].shape[0]
            processed["segment"] = np.ones(num_points, dtype=np.int32) * -1
            print("  Created dummy segmentation")
        
        return processed
    
    def process_scene(self, scene_path: str) -> Dict[str, torch.Tensor]:
        """Complete processing pipeline for a scene"""
        # Load raw data
        raw_data = self.load_scene_data(scene_path)
        
        # Preprocess
        processed_data = self.preprocess_data(raw_data)
        
        # Apply transforms to get grid-sampled data and grid_coord
        data = self.transform(processed_data)
        # Ensure grid_coord exists (fallback if not provided)
        if "grid_coord" not in data:
            coord_np = data["coord"].numpy() if hasattr(data["coord"], "numpy") else data["coord"]
            grid_size = 0.01
            grid_np = np.floor(coord_np / grid_size).astype(np.int32)
            data["grid_coord"] = torch.from_numpy(grid_np)
        
        print(f"  Final data shapes:")
        for k in ["coord", "grid_coord", "feat"]:
            if k in data:
                v = data[k]
                print(f"    {k}: {v.shape} ({v.dtype})")
        return data


class SceneSplat:
    """
    Wrapper for PointTransformerV3 model with GS-specific processing
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.model = self._load_model(checkpoint_path)
        
    def _get_model_config(self):
        """Get model configuration matching the config file"""
        config = dict(
            in_channels=14,
            order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
            stride=(2, 2, 2),
            enc_depths=(2, 2, 2, 6),
            enc_channels=(32, 64, 128, 256),
            enc_num_head=(2, 4, 8, 16),
            enc_patch_size=(1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2),
            dec_channels=(768, 512, 256),
            dec_num_head=(16, 16, 16),
            dec_patch_size=(1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False)
        return config
    
    def _load_model(self, checkpoint_path: str):
        """Load the PointTransformerV3 model"""
        print(f"Loading model from: {checkpoint_path}")
        
        # Create model with config
        config = self._get_model_config()
        model = PointTransformerV3(**config)
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
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
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing GS data")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Specific scene to process (e.g., 'train/scene_00001')")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/runyi_yang/Gen3D/SpatialScripts/SpatialLM/checkpoints/scenesplat/model_best_lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--list_scenes", action="store_true",
                        help="List available scenes and exit")
    parser.add_argument("--save_features", action="store_true",
                        help="Save extracted features to file")
    parser.add_argument("--save_output", action="store_true",
                        help="Save coord/attributes and features to files")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output features")
    
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = GaussianSplatDataLoader(args.data_root)
    
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
    model = SceneSplat(args.checkpoint, args.device)
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