"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

CLI entry point for config-driven FPM reconstruction.

Usage
-----
python scripts/run_fpm.py --config configs/experiment_leech.json

This script is intentionally thin:
- parse the config path
- load and validate the JSON config
- select a device
- run the experiment pipeline

The heavy lifting lives in the fpm package so that the same experiment
runner can later be reused from notebooks, tests, or batch jobs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from fpm.config_models import load_experiment_config
from fpm.pipeline import run_experiment
from fpm.tiling import resolve_crop_descriptors
# Global logging utility used for experiment logging and output management
from fpm.logger import FPMLoggerObject


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for this script.
    """
    parser = argparse.ArgumentParser(description="Run an FPM reconstruction experiment from a JSON config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment JSON file.",
    )
    return parser


def main() -> None:
    """
    Main CLI entry point.
    """
    # Build the argument parser and parse the command-line arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load the experiment configuration from the specified JSON file
    cfg = load_experiment_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve all crop directories to process
    crop_descs = resolve_crop_descriptors(cfg)

    # Create the output directory and set up logging
    FPMLoggerObject.set_debug_level(logging.INFO)
    run_dir = Path(FPMLoggerObject.create_timestamped_output_dir(cfg.dataset.save_root))
    FPMLoggerObject.set_log_file_path_from_output_dir()
    FPMLoggerObject.save_json_config(cfg.model_dump())
    FPMLoggerObject.save_json_config(
        {"crop_descs": crop_descs},
        filename="resolved_crop_descs.json",
    )

    FPMLoggerObject.log.info("Starting FPM run")
    FPMLoggerObject.log.info("Config file: %s", args.config)
    FPMLoggerObject.log.info("Device: %s", device)
    FPMLoggerObject.log.info("Run directory: %s", run_dir)

    FPMLoggerObject.log.info("Dataset input mode: %s", cfg.dataset.input_mode)
    FPMLoggerObject.log.info("Crops mode: %s", cfg.crops.mode)
    FPMLoggerObject.log.info("Crops selection: %s", cfg.crops.selection)
    FPMLoggerObject.log.info("Number of crops to process: %d", len(crop_descs))

    if cfg.crops.crop_size is not None:
        FPMLoggerObject.log.info(
            "Crop size: %s x %s",
            cfg.crops.crop_size[0],
            cfg.crops.crop_size[1],
        )

    if cfg.crops.overlap is not None:
        FPMLoggerObject.log.info("Crop overlap: %.2f", cfg.crops.overlap)

    for crop_idx_run, crop_desc in enumerate(crop_descs, start=1):

        FPMLoggerObject.log.info("--------------------------------------------------")
        FPMLoggerObject.log.info(
            "Processing crop %d/%d",
            crop_idx_run,
            len(crop_descs),
        )
        FPMLoggerObject.log.info("Crop label: %s", crop_desc["crop_dir"])
        FPMLoggerObject.log.info("Active crop index: %s", crop_desc["crop_index"])
        FPMLoggerObject.log.info(
            "Active crop origin: row=%s, col=%s",
            crop_desc["origin_rc"][0],
            crop_desc["origin_rc"][1],
        )

        FPMLoggerObject.save_json_config(
            crop_desc,
            filename=f"active_crop_{crop_desc['crop_dir']}.json",
        )

        # Run the experiment pipeline with the loaded config and resolved crop descriptor
        run_experiment(cfg, device=device, run_dir=run_dir, crop_desc=crop_desc)

        FPMLoggerObject.log.info("Finished crop: %s", crop_desc["crop_dir"])

    FPMLoggerObject.log.info("FPM run completed successfully")


if __name__ == "__main__":
    main()