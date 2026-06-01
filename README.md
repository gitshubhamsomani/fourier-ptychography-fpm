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
```

## Install

```bash
conda create -n fpm_torch python=3.11

conda activate fpm_torch

pip install torch numpy matplotlib pydantic

conda install -c conda-forge openexr imath
```

## Config Example
Each experiment config (e.g. experiment_leech.json) has four sections:

- dataset — input paths, filename pattern, LED coords file
- crops — crop loading or crop generation settings
- setup — microscope and LED geometry
- reconstruction — optimization settings

Note: The main experiment configuration includes the path to the LED coordinates file. This keeps all experiment-level information accessible from a single JSON file.

## Run
```bash
python scripts/run_fpm.py --config configs/experiment_leech.json
```

## Contact
John Meshreki
john.meshreki@gmail.com