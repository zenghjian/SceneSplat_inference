#!/usr/bin/env python3
"""
ScanNet Gaussian Splatting Dataloader

This script processes ScanNet Gaussian Splatting PLY files and converts them to NPZ format
for efficient loading in training pipelines.

Based on the run_gs_pipeline.py implementation for proper PLY parsing.
"""

import os
import sys
import numpy as np
import torch
import plyfile
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def SH2RGB(sh):
    """Convert SH DC coefficients to RGB [0,1] using viewer's method"""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


class ScanNetGaussianProcessor:
    """
    Processes ScanNet Gaussian Splatting data from PLY files to NPZ format
    """
    
    def __init__(self, data_root: str, output_root: str, max_gaussians: Optional[int] = None):
        """
        Initialize the processor
        
        Args:
            data_root: Path to ScanNet Gaussian data (contains scene folders)
            output_root: Path where NPZ files will be saved
            max_gaussians: Maximum number of Gaussians to keep per scene (None = keep all)
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.max_gaussians = max_gaussians
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_scenes': 0,
            'processed_scenes': 0,
            'failed_scenes': 0,
            'total_gaussians': 0,
            'avg_gaussians_per_scene': 0
        }
    
    def get_scene_list(self) -> List[Path]:
        """Get list of all scene directories"""
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")
        
        scene_dirs = []
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name.startswith('scene'):
                ply_path = item / 'ckpts' / 'point_cloud_30000.ply'
                if ply_path.exists():
                    scene_dirs.append(item)
                else:
                    logger.warning(f"No PLY file found in {item}")
        
        scene_dirs.sort()
        self.stats['total_scenes'] = len(scene_dirs)
        logger.info(f"Found {len(scene_dirs)} scenes with PLY files")
        return scene_dirs
    
    def extract_gaussian_parameters(self, ply_path: Path) -> Dict[str, np.ndarray]:
        """
        Extract Gaussian parameters from PLY file
        Based on the run_gs_pipeline.py implementation
        
        Args:
            ply_path: Path to PLY file
            
        Returns:
            Dictionary containing Gaussian parameters
        """
        try:
            # Read PLY file
            logger.debug(f"Loading PLY file: {ply_path}")
            ply_data = plyfile.PlyData.read(str(ply_path))
            vertex = ply_data["vertex"]
            
            num_gaussians = len(vertex)
            logger.debug(f"Processing {num_gaussians} Gaussians from {ply_path}")
            
            # Extract coordinates (positions)
            coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
            
            # Extract DC features (RGB color) and convert properly
            features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
            # Convert SH DC to RGB [0,1] then to [0,255]
            color = SH2RGB(features_dc) * 255.0
            color = np.clip(color, 0, 255).astype(np.float32)
            
            # Extract opacity
            opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
            # Apply sigmoid activation for PLY data
            opacity = 1 / (1 + np.exp(-opacity))
            
            # Extract quaternion (rot_0, rot_1, rot_2, rot_3 -> wxyz)
            quat = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)
            # Normalize quaternions
            quat_norms = np.linalg.norm(quat, axis=1, keepdims=True)
            quat = quat / (quat_norms + 1e-8)
            
            # Extract scales
            scale = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
            # Apply exponential activation for PLY data and clip
            scale = np.exp(scale)
            scale = np.clip(scale, 0, 1.5)
            
            # Extract normals if available
            normal = None
            # Check if normal properties exist by trying to access them
            try:
                normal = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
                # Normalize normals
                normal_norms = np.linalg.norm(normal, axis=1, keepdims=True)
                normal = normal / (normal_norms + 1e-8)
                logger.debug(f"Loaded normals: {normal.shape}")
            except (ValueError, KeyError):
                logger.debug("No normals found in PLY file")
                normal = None
            
            # Apply subsampling if requested
            if self.max_gaussians is not None and num_gaussians > self.max_gaussians:
                indices = np.random.choice(num_gaussians, self.max_gaussians, replace=False)
                coord = coord[indices]
                color = color[indices]
                opacity = opacity[indices]
                quat = quat[indices]
                scale = scale[indices]
                if normal is not None:
                    normal = normal[indices]
                logger.info(f"Subsampled from {num_gaussians} to {self.max_gaussians} Gaussians")
                num_gaussians = self.max_gaussians
            
            # Create result dictionary
            result = {
                'coord': coord,
                'color': color,
                'opacity': opacity,
                'quat': quat,
                'scale': scale,
            }
            
            if normal is not None:
                result['normal'] = normal
            
            logger.debug(f"Extracted Gaussian parameters:")
            logger.debug(f"  coord: {coord.shape}")
            logger.debug(f"  color: {color.shape}, range: [{color.min():.2f}, {color.max():.2f}]")
            logger.debug(f"  opacity: {opacity.shape}, range: [{opacity.min():.4f}, {opacity.max():.4f}]")
            logger.debug(f"  quat: {quat.shape}")
            logger.debug(f"  scale: {scale.shape}, range: [{scale.min():.4f}, {scale.max():.4f}]")
            if normal is not None:
                logger.debug(f"  normal: {normal.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {ply_path}: {e}")
            raise
    
    def process_scene(self, scene_path: Path) -> bool:
        """
        Process a single scene
        
        Args:
            scene_path: Path to scene directory
            
        Returns:
            True if successful, False otherwise
        """
        scene_name = scene_path.name
        ply_path = scene_path / 'ckpts' / 'point_cloud_30000.ply'
        output_path = self.output_root / f"{scene_name}.npz"
        
        # Skip if already processed
        if output_path.exists():
            logger.info(f"Skipping {scene_name} (already processed)")
            return True
        
        try:
            logger.info(f"Processing scene: {scene_name}")
            
            # Extract Gaussian parameters
            gaussian_data = self.extract_gaussian_parameters(ply_path)
            
            # Add metadata
            gaussian_data['scene_name'] = scene_name
            gaussian_data['num_gaussians'] = gaussian_data['coord'].shape[0]
            
            # Save as NPZ
            np.savez_compressed(output_path, **gaussian_data)
            
            # Update statistics
            self.stats['processed_scenes'] += 1
            self.stats['total_gaussians'] += gaussian_data['num_gaussians']
            
            logger.info(f"Saved {scene_name}: {gaussian_data['num_gaussians']} Gaussians -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {scene_name}: {e}")
            self.stats['failed_scenes'] += 1
            return False
    
    def process_all_scenes(self):
        """Process all scenes in the dataset"""
        scene_list = self.get_scene_list()
        
        logger.info(f"Starting to process {len(scene_list)} scenes")
        logger.info(f"Output directory: {self.output_root}")
        if self.max_gaussians is not None:
            logger.info(f"Max Gaussians per scene: {self.max_gaussians}")
        
        # Process scenes with progress bar
        for scene_path in tqdm(scene_list, desc="Processing scenes"):
            self.process_scene(scene_path)
        
        # Calculate final statistics
        if self.stats['processed_scenes'] > 0:
            self.stats['avg_gaussians_per_scene'] = self.stats['total_gaussians'] / self.stats['processed_scenes']
        
        # Save statistics
        stats_path = self.output_root / 'processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Print summary
        logger.info("="*60)
        logger.info("Processing complete!")
        logger.info(f"Total scenes: {self.stats['total_scenes']}")
        logger.info(f"Processed successfully: {self.stats['processed_scenes']}")
        logger.info(f"Failed: {self.stats['failed_scenes']}")
        logger.info(f"Total Gaussians: {self.stats['total_gaussians']}")
        logger.info(f"Average Gaussians per scene: {self.stats['avg_gaussians_per_scene']:.1f}")
        logger.info(f"Statistics saved to: {stats_path}")
        logger.info(f"NPZ files saved to: {self.output_root}")


class ScanNetGaussianDataset:
    """
    Dataset class for loading processed ScanNet Gaussian data from NPZ files
    Compatible with PyTorch DataLoader
    """
    
    def __init__(self, data_root: str, split: str = "train", cache: bool = False):
        """
        Initialize dataset
        
        Args:
            data_root: Path to processed NPZ files
            split: Dataset split (currently not used, all data is treated as one split)
            cache: Whether to cache loaded data in memory
        """
        self.data_root = Path(data_root)
        self.split = split
        self.cache = cache
        self.cached_data = {}
        
        # Get list of NPZ files
        self.npz_files = list(self.data_root.glob("scene*.npz"))
        self.npz_files.sort()
        
        if not self.npz_files:
            raise FileNotFoundError(f"No NPZ files found in {data_root}")
        
        logger.info(f"Found {len(self.npz_files)} scenes in dataset")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        """Load data for a single scene"""
        npz_path = self.npz_files[idx]
        scene_name = npz_path.stem
        
        # Check cache
        if self.cache and scene_name in self.cached_data:
            return self.cached_data[scene_name]
        
        # Load NPZ file
        data = np.load(npz_path)
        
        # Convert to dictionary
        result = {}
        for key in data.files:
            if key == 'scene_name':
                result[key] = str(data[key])  # Convert numpy string to regular string
            else:
                result[key] = data[key]
        
        # Cache if requested
        if self.cache:
            self.cached_data[scene_name] = result
        
        return result
    
    def get_scene_name(self, idx):
        """Get scene name from index"""
        return self.npz_files[idx].stem


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Process ScanNet Gaussian Splatting data")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to ScanNet Gaussian data directory")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to save processed NPZ files")
    parser.add_argument("--max_gaussians", type=int, default=None,
                        help="Maximum number of Gaussians per scene (None = keep all)")
    parser.add_argument("--test_loading", action="store_true",
                        help="Test loading processed data after conversion")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process all scenes
    processor = ScanNetGaussianProcessor(
        data_root=args.data_root,
        output_root=args.output_root,
        max_gaussians=args.max_gaussians
    )
    
    processor.process_all_scenes()
    
    # Test loading if requested
    if args.test_loading:
        logger.info("="*60)
        logger.info("Testing data loading...")
        dataset = ScanNetGaussianDataset(args.output_root)
        
        # Load first few scenes
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Scene: {sample['scene_name']}")
            logger.info(f"  Number of Gaussians: {sample['num_gaussians']}")
            logger.info(f"  Available keys: {list(sample.keys())}")
            logger.info(f"  Coordinate shape: {sample['coord'].shape}")
            logger.info(f"  Color range: [{sample['color'].min():.2f}, {sample['color'].max():.2f}]")
            logger.info(f"  Opacity range: [{sample['opacity'].min():.4f}, {sample['opacity'].max():.4f}]")
            if 'normal' in sample:
                logger.info(f"  Normal shape: {sample['normal'].shape}")
        
        logger.info("Test loading successful!")


if __name__ == "__main__":
    main()