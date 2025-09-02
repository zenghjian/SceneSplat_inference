#!/usr/bin/env python3
"""
Convert coord.npy and feat.npy to PLY point cloud file
Uses PCA to reduce feature dimensions to RGB colors

Usage:
    python convert_npy_to_ply.py --coord coord.npy --feat feat.npy --output scene.ply
"""

import argparse
import numpy as np
import plyfile
from sklearn.decomposition import PCA
from pathlib import Path


def load_data(coord_path: str, feat_path: str):
    """Load coordinate and feature data"""
    print(f"Loading coordinates from: {coord_path}")
    coord = np.load(coord_path).astype(np.float32)
    print(f"  Coordinates shape: {coord.shape}")
    
    print(f"Loading features from: {feat_path}")
    feat = np.load(feat_path).astype(np.float32)
    print(f"  Features shape: {feat.shape}")
    
    # Ensure same number of points
    assert coord.shape[0] == feat.shape[0], f"Coordinate and feature point counts don't match: {coord.shape[0]} vs {feat.shape[0]}"
    
    return coord, feat


def pca_to_colors(feat: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Use PCA to reduce features to 3D and convert to RGB colors [0,255] with enhanced contrast"""
    print(f"Applying PCA to reduce {feat.shape[1]}D features to {n_components}D...")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    feat_reduced = pca.fit_transform(feat)
    
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Enhanced contrast normalization
    feat_normalized = np.zeros_like(feat_reduced)
    
    for i in range(n_components):
        channel = feat_reduced[:, i]
        
        # Use percentile-based normalization for better contrast
        p_low, p_high = np.percentile(channel, [2, 98])  # Remove outliers
        channel_clipped = np.clip(channel, p_low, p_high)
        
        # Normalize to [0, 1]
        if p_high > p_low:
            channel_norm = (channel_clipped - p_low) / (p_high - p_low)
        else:
            channel_norm = np.zeros_like(channel_clipped)
        
        # Apply histogram equalization for better contrast
        channel_eq = histogram_equalization(channel_norm)
        
        # Apply gamma correction to enhance mid-tones
        gamma = 0.7  # Values < 1 brighten mid-tones
        channel_gamma = np.power(channel_eq, gamma)
        
        feat_normalized[:, i] = channel_gamma
     
    # Convert to RGB [0, 255]
    colors = (feat_normalized * 255).astype(np.uint8)
    
    print(f"  Enhanced color range per channel:")
    for i in range(n_components):
        channel_name = ['Red', 'Green', 'Blue'][i] if i < 3 else f'Channel_{i}'
        print(f"    {channel_name}: [{colors[:, i].min()}, {colors[:, i].max()}]")
    
    return colors


def histogram_equalization(data: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Apply histogram equalization to enhance contrast"""
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=n_bins, range=(0, 1))
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize CDF to [0, 1]
    cdf_normalized = cdf / cdf[-1]
    
    # Interpolate to get equalized values
    data_eq = np.interp(data, bin_edges[:-1], cdf_normalized)
    
    return data_eq


def save_ply(coord: np.ndarray, colors: np.ndarray, output_path: str):
    """Save coordinates and colors to PLY file"""
    print(f"Saving PLY file to: {output_path}")
    
    num_points = coord.shape[0]
    
    # Create vertex array with required dtype
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    
    vertices = np.empty(num_points, dtype=vertex_dtype)
    vertices['x'] = coord[:, 0]
    vertices['y'] = coord[:, 1] 
    vertices['z'] = coord[:, 2]
    vertices['red'] = colors[:, 0]
    vertices['green'] = colors[:, 1]
    vertices['blue'] = colors[:, 2]
    
    # Create PLY element
    vertex_element = plyfile.PlyElement.describe(vertices, 'vertex')
    
    # Write PLY file
    plyfile.PlyData([vertex_element]).write(output_path)
    
    print(f"Successfully saved {num_points} points to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert coord.npy and feat.npy to PLY point cloud")
    parser.add_argument("--coord", type=str, required=True,
                        help="Path to coord.npy file [N, 3]")
    parser.add_argument("--feat", type=str, required=True,
                        help="Path to feat.npy file [N, D]")
    parser.add_argument("--output", type=str, required=True,
                        help="Output PLY file path")
    parser.add_argument("--pca_components", type=int, default=3,
                        help="Number of PCA components (default: 3 for RGB)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.coord).exists():
        raise FileNotFoundError(f"Coordinate file not found: {args.coord}")
    if not Path(args.feat).exists():
        raise FileNotFoundError(f"Feature file not found: {args.feat}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    coord, feat = load_data(args.coord, args.feat)
    
    # Apply PCA to features
    colors = pca_to_colors(feat, args.pca_components)
    
    # Save PLY file
    save_ply(coord, colors, args.output)
    
    print("\nConversion completed successfully!")
    print(f"Input: {coord.shape[0]} points, {feat.shape[1]}D features")
    print(f"Output: {args.output} with PCA-reduced RGB colors")


if __name__ == "__main__":
    main() 