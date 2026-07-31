# Low-Cost Fourier Ptychographic Microscope (FPM)

A low-cost, portable computational microscope built around a **Raspberry Pi 3 Model B**, a reversed-lens **Raspberry Pi Camera Module V2.1 NoIR** (Sony IMX219), and a **Pimoroni Unicorn HAT Mini 17×7 (119-LED) RGB array**, together with a config-driven, GPU-accelerated PyTorch reconstruction pipeline based on **Embedded Pupil Function Recovery (EPRY)**.

The instrument sequentially illuminates a sample from up to 119 discrete LED angles, captures one low-resolution image per angle, and computationally reconstructs a higher-resolution complex (amplitude + phase) image — achieving resolution beyond the native objective NA.

> M.Sc. Mechatronics Semester Project — University of Siegen, 2026

**[Live project page](https://gitshubhamsomani.github.io/fourier-ptychography-fpm/)** · **[Full report (PDF)](docs/report_21_07_updated.pdf)**

---

## Results

- Optical alignment validated on a **USAF-1951** resolution target, resolving spatial frequencies up to **~180 line pairs/mm (~2.8 µm features)**.
- A full reconstruction run produced a **25-megapixel** image with a reported sub-micron resolution of **~780 nm**.
- Evaluated on biological tissue samples (mushroom and intestine sections).

---

## How It Works

Fourier Ptychographic Microscopy addresses the field-of-view vs. resolution trade-off **computationally** rather than optically. Each oblique illumination angle shifts a different patch of the sample's spatial-frequency spectrum into the pass-band of a low-NA objective. Combining many such measurements synthesizes a much larger effective aperture, so the synthetic NA approaches `NA_obj + NA_illum` — decoupling resolution from the physical objective's NA while retaining its wide native field of view.

The reconstruction uses the **EPRY** alternating-minimization algorithm [Ou, Zheng & Yang, 2014], which jointly recovers both the high-resolution sample spectrum and the objective's pupil function directly from the captured image set — removing the need for a separate aberration-characterization step.

---

## Key Parameters

| Parameter | Value |
|---|---|
| Wavelength | 530 nm (green channel) |
| Objective NA | 0.15 |
| Magnification | 1.5× |
| Camera pixel size | 1.12 µm |
| LED array grid | 17 × 7 (119 LEDs) |
| LED pitch | 3.3 mm |
| LED array–to–sample distance | 62 mm |
| Working distance | ≈ 10 mm |
| Reconstruction crop size | 256 × 256 px |
| Dark-field LED self-calibration | On |

---

## Pipeline Overview

- **live_script** — live camera preview to verify sample alignment and focus before acquisition.
- **capture** — captures a dark frame, then sequentially illuminates each LED, saving one raw (DNG) and one JPEG per position plus the LED coordinate file.
- **config_models** — defines and validates the `ExperimentConfig` schema from each experiment's JSON file.
- **geometry** — computes derived setup tensors (pupil radius, per-LED illumination NA, per-LED k-vectors).
- **io** — loads raw LED image stacks, LED coordinate files, and dark-frame calibration images.
- **tiling** — splits the full frame into crops and selects active crop(s), keeping GPU memory tractable.
- **calibration** — dark-field LED position self-calibration [Eckert, Phillips & Waller, 2018].
- **reconstruction** — EPRY alternating-minimization (Fourier magnitude projection + object/pupil gradient updates) on PyTorch tensors.
- **logger** — structured per-run logging of parameters, convergence metrics, and output paths.
- **pipeline** — top-level `run_experiment` orchestrating dark-frame subtraction, optional self-calibration, tiling, and reconstruction.
- **run_fpm** — command-line entry point that loads a JSON config and runs the pipeline.

---

## Install

```bash
conda create -n fpm_torch python=3.11
conda activate fpm_torch

pip install torch numpy matplotlib pydantic
conda install -c conda-forge openexr imath
```

## Run

```bash
python scripts/run_fpm.py --config configs/experiment_intestine.json
```

Each experiment config has four sections:

- **dataset** — input paths, filename pattern, LED coordinate file
- **crops** — crop loading or crop-generation settings
- **setup** — microscope and LED geometry
- **reconstruction** — optimization settings (tolerance, min/max iterations, step sizes, pupil and DF-calibration flags)

Adding a new sample means writing a new JSON config — no source-code changes required.

---

## Hardware Assembly Notes

The mechanical assembly was adapted from a different LED-array footprint to fit the 17×7 Unicorn HAT Mini:

- **Camera mounting offset:** the camera mount and aperture were shifted ~8 mm from the housing's original centre so the optical axis aligns with the array's actual centre LED.
- **Lens inversion:** the stock Camera Module V2.1 lens (designed for infinite-conjugate imaging) was removed and re-mounted in a reversed orientation, shifting the focal plane to the object side for stable focus at ~10 mm working distance.

---

## Attribution

This project is based on the **FPM Torch** reconstruction codebase by **John Meshreki** (MIT License — see [LICENSE](LICENSE), retained in full). The reconstruction implements Embedded Pupil Function Recovery (EPRY) [Ou, Zheng & Yang, 2014], and the LED position self-calibration follows the method of Eckert, Phillips & Waller (2018).

**Contributions in this work:**
- Assembly of the physical instrument (camera-mount offset and reversed-lens re-mounting).
- Adaptation of the dark-field LED self-calibration to the coarser 17×7 Unicorn HAT Mini array.
- Experiment configurations for the USAF-1951, intestine, and mushroom samples.
- A tiled acquisition/reconstruction scheme processing crops of the full camera frame.
- Standard Operating Procedure (SOP) documentation for reproduction by future students.

---

## References

1. G. Zheng, R. Horstmeyer, C. Yang, "Wide-field, high-resolution Fourier ptychographic microscopy," *Nature Photonics*, 7, 739–745, 2013.
2. X. Ou, G. Zheng, C. Yang, "Embedded pupil function recovery for Fourier ptychographic microscopy," *Optics Express*, 22, 4960–4972, 2014.
3. R. Eckert, Z. F. Phillips, L. Waller, "Efficient illumination angle self-calibration in Fourier ptychography," *Applied Optics*, 57, 5434, 2018.
4. T. Aidukas, R. Eckert, A. R. Harvey, L. Waller, P. C. Konda, "Low-cost, sub-micron resolution, wide-field computational microscopy using open-source hardware," *Scientific Reports*, 9, 7457, 2019.

---

## License

MIT License — see [LICENSE](LICENSE). Original copyright © 2026 johnmeshreki.
