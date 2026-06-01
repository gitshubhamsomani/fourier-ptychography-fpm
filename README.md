# FPM Torch Meshreki

A PyTorch-based reconstruction pipeline for **Fourier Ptychographic Microscopy (FPM)** with config-driven experiments, flexible crop handling, EXR image loading, and crop-aware illumination geometry.

This repository is designed for running and comparing FPM reconstructions across different datasets and acquisition setups while keeping experiments reproducible and easy to configure.

---

## Features

- PyTorch-based FPM reconstruction
- GPU acceleration with CUDA when available
- Config-driven experiments using JSON
- Support for:
  - pre-cropped image directories
  - full-frame image loading with on-the-fly cropping
- Flexible crop selection:
  - single
  - list
  - range
  - generated crops
- LED geometry driven by coordinate files and setup calibration
- Timestamped outputs with logs and metadata
- Partial backward compatibility with older config formats

---

## Repository Structure

```text
fpm_torch_codebase/
├── configs/
│   ├── experiment_leech.json
│   └── led_coords_leech.json
├── fpm/
│   ├── config_models.py
│   ├── geometry.py
│   ├── io.py
│   ├── logger.py
│   ├── pipeline.py
│   ├── reconstruction.py
│   ├── tiling.py
│   └── utils.py
├── scripts/
│   └── run_fpm.py
└── README.md

## Install

```bash
conda create -n fpm_torch python=3.11

conda activate fpm_torch

pip install torch numpy matplotlib pydantic

conda install -c conda-forge openexr imath

## Run
```bash
python scripts/run_fpm.py --config configs/experiment_leech.json

## Config Example
Each experiment config has four sections:

- dataset — input paths, filename pattern, LED coords file
- setup — microscope and LED geometry
- reconstruction — optimization settings
- crops — crop loading or crop generation settings

## Minimal Example
{
  "dataset": {
    "sample_name": "Leech",
    "input_mode": "load_cropped_dirs",
    "input_root": "/path/to/data",
    "file_pattern": "image_x_{x}_y_{y}.exr",
    "coords_file": "configs/led_coords_leech.json",
    "nused": 293,
    "save_root": "/path/to/output"
  },
  "setup": {
    "wavelength_um": 0.519,
    "objective_na": 0.3,
    "magnification": 10.2,
    "camera_pixel_um": 6.4,
    "patch_size": [512, 512],
    "image_center": [256, 256],
    "led_distance_um": 66500.0,
    "led_coord_unit_um": 63.0,
    "led_coord_center": [960.0, 540.0],
    "led_coords_convention": "xy",
    "led_axis_signs": [-1, -1],
    "z0": 0.0
  },
  "reconstruction": {
    "tol": 0.01,
    "max_iter": 10,
    "min_iter": 1,
    "monotone": true,
    "step_size": 0.01,
    "op_alpha": 1.0,
    "op_beta": 1000.0,
    "display": "full",
    "mode": "fourier",
    "use_default_p0": true,
    "optimize_p_flags": [true],
    "pupil_scaling_factors": [1.0]
  },
  "crops": {
    "mode": "load",
    "selection": "single",
    "crop_dir": "crop_82",
    "full_image_shape": [3753, 5634],
    "crop_size": [512, 512],
    "overlap": 0.30,
    "max_crops": 150,
    "indexing": "row_major"
  }
}

## Contact
John Meshreki
john.meshreki@gmail.com