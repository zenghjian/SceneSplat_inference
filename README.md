# SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining - A Minimal Implementation of Inference

<h2 align="left">ICCV 2025 Oral</h2>

[![arXiv](https://img.shields.io/badge/arXiv-2503.18052-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2503.18052)
[![GitHub](https://img.shields.io/badge/GitHub-Webpage-blue?logo=github)](https://github.com/unique1i/SceneSplat_webpage)
[![🤗 HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k)

<p align="left">
    <img src="imgs/vis.png" alt="SceneSplat Demo" style="width:100%; max-width:1200px;">
</p>

The minimal inference implementation of our work: <strong>SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining</strong>. 
Input Gaussian Splats
***

#### $^\star$[Yue Li](https://unique1i.github.io/)<sup>1</sup>, $^\star$[Qi Ma](https://qimaqi.github.io/)<sup>2,3</sup>, [Runyi Yang](https://runyiyang.github.io/)<sup>3</sup>, [Huapeng Li](https://scholar.google.com/citations?user=LkF7__QAAAAJ)<sup>2</sup>, [Mengjiao Ma](https://insait.ai/mengjiao-ma/)<sup>3,4</sup>, $^\dagger$[Bin Ren](https://amazingren.github.io/)<sup>3,5,6</sup>, [Nikola Popovic](https://insait.ai/dr-nikola-popovic/)<sup>3</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>6</sup>, [Ender Konukoglu](https://people.ee.ethz.ch/~kender/)<sup>2</sup>, [Theo Gevers](https://staff.science.uva.nl/th.gevers/)<sup>1</sup>, [Luc Van Gool](https://insait.ai/prof-luc-van-gool/)<sup>2,3</sup>, [Martin R. Oswald](https://oswaldm.github.io/)<sup>1</sup>, and [Danda Pani Paudel](https://insait.ai/dr-danda-paudel/)<sup>3</sup>

$^\star$: Equal Contribution, $^\dagger$: Corresponding Author <br>

<sup>1</sup> University of Amsterdam <br>
<sup>2</sup> ETH Zürich <br>
<sup>3</sup> INSAIT <br>
<sup>4</sup> Nanjing University of Aeronautics and Astronautics <br>
<sup>5</sup> University of Pisa <br>
<sup>6</sup> University of Trento <br>

## Installation

Please set up the provided conda environment with Python 3.10, PyTorch 2.5.1, and CUDA 12.4. 

```bash
conda env create -f env.yaml
conda activate scene_splat
```

## TL;DR
More Details and how to prepare npy data should be refered to <a href=https://github.com/unique1i/SceneSplat>SceneSplat</a>

## Run

### Basic Usage

Run SceneSplat inference on NPY data:

```bash
python run_gs_pipeline.py \
    --npy_folder example_data \
    --scene_name scene0000_00 \
    --model_folder checkpoints/model_normal/ \
    --device cuda \
    --save_features
```

Run SceneSplat inference on PLY data:

```bash
python run_gs_pipeline.py \
    --ply /path/to/scene.ply \
    --model_folder checkpoints/model_normal/ \
    --device cuda \
    --save_features
```

### Command Options

- `--npy_folder`: Root directory containing NPY scene data (with structure: `train/`, `val/`, `test/` subdirs)
- `--ply`: Path to PLY file containing Gaussian Splatting data
- `--model_folder`: Path to folder containing model checkpoint (.pth) and config_inference.py
- `--normal`: Include normal vectors in features (adds 3 channels, default: False)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--save_features`: Save extracted language features to `pred_langfeat.npy`
- `--save_output`: Save input attributes (coord, color, opacity, quat, scale, normal)
- `--output_dir`: Output directory for saved files (default: `./output`)
- `--list_scenes`: List all available scenes in npy_folder and exit (NPY format only)

### Input Data Format

**NPY Format (Preprocessed):**
Each scene should be a directory containing these `.npy` files:
```
scene0000_00/
├── coord.npy      # [N, 3] 3D coordinates
├── color.npy      # [N, 3] RGB colors (0-255 or 0-1)
├── opacity.npy    # [N, 1] or [N] opacity values
├── quat.npy       # [N, 4] quaternions (wxyz)
├── scale.npy      # [N, 3] scaling factors
├── normal.npy     # [N, 3] surface normals (optional)
└── segment.npy    # [N] semantic labels (optional)
```

**PLY Format (Raw Gaussian Splatting):**
Standard 3D Gaussian Splatting PLY files with these attributes:
```
scene.ply
├── x, y, z           # 3D coordinates
├── f_dc_0/1/2        # Spherical harmonic DC coefficients (RGB)
├── opacity           # Raw opacity values
├── rot_0/1/2/3       # Quaternion components (wxyz)
├── scale_0/1/2       # Log-space scaling factors
├── nx, ny, nz        # Normal vectors (optional)
└── f_rest_*          # Higher-order SH coefficients (ignored)
```


### Output

When `--save_features` is used, the script saves:
- `pred_langfeat.npy`: [N, D] L2-normalized language features (float16)

Features are automatically mapped back to original point order using inverse sampling if available.

### Examples

List available NPY scenes:
```bash
python run_gs_pipeline.py --npy_folder example_data --list_scenes
```

Process NPY data with custom output:
```bash
python run_gs_pipeline.py \
    --npy_folder /path/to/data \
    --scene_name scene0000_00 \
    --model_folder checkpoints/model_normal/ \
    --save_features \
    --output_dir ./results
```

Process PLY data with normals:
```bash
python run_gs_pipeline.py \
    --ply /path/to/gaussians.ply \
    --model_folder checkpoints/model_normal/ \
    --normal \
    --save_features \
    --output_dir ./results
```

### Feature Dimensions

The model input channels depend on the `--normal` flag:
- **Without `--normal`**: 11 channels (3 color + 1 opacity + 4 quat + 3 scale)
- **With `--normal`**: 14 channels (3 color + 1 opacity + 4 quat + 3 scale + 3 normal)

Make sure your model checkpoint matches the expected input dimensions.

## Viewer
Please refer to <a href="viewer/README.md">Viewer</a> to visualize language feature.
## Acknowledgement
We sincerely thank all the author teams of the original datasets for their contributions. Our work builds on the following repositories:

- [Pointcept](https://github.com/Pointcept/Pointcept) repository, on which we develop our codebase,
- [gsplat](https://github.com/nerfstudio-project/gsplat) repository, which we adapted to optimize the 3DGS scenes,
- [Occam's LGS](https://github.com/insait-institute/OccamLGS) repository, which we adapted for 3DGS pseudo label collection.

We are grateful to the authors for their open-source contributions!