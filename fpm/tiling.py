"""     
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

-----------------------------------------------------------------------------
Tiling and cropping utilities for FPM.          
This module provides functions for:
- computing crop strides from crop size and overlap
- parsing crop directory names to extract indices
- generating crop origins to cover the full image with specified overlap
- resolving the active single-crop descriptor from the experiment config        
The crop grid is generated based on the full image shape, crop size, and 
overlap, ensuring that the last crops in each direction align with the image boundaries. 
The single-crop descriptor includes the crop directory name, index, top-left origin, and crop size, 
which are used by the reconstruction pipeline to load and process the correct image region.
"""

from __future__ import annotations

import re


def parse_crop_index(crop_dir: str) -> int:
    """
    Parse a crop directory name of the form ``crop_<index>``.

    Parameters
    ----------
    crop_dir:
        Crop directory name, for example ``"crop_82"``.

    Returns
    -------
    int
        Parsed crop index.

    Raises
    ------
    ValueError
        If the name does not follow the expected ``crop_<index>`` pattern.
    """
    match = re.fullmatch(r"crop_(\d+)", crop_dir)
    if match is None:
        raise ValueError(f"Invalid crop directory name: {crop_dir}")
    return int(match.group(1))


def generate_crop_origins(
    full_image_shape: list[int],
    crop_size: list[int],
    overlap: float,
) -> list[tuple[int, int]]:
    """
    Generate zero-based crop origins that match the original cropping layout.

    Parameters
    ----------
    full_image_shape:
        Full image shape as ``[rows, cols]``.
    crop_size:
        Crop size as ``[rows, cols]``.
    overlap:
        Fractional overlap between neighboring crops.

    Returns
    -------
    list[tuple[int, int]]
        List of zero-based crop origins as ``[(row0, col0), ...]``.

    Raises
    ------
    ValueError
        If the supplied image or crop shape is invalid.

    Notes
    -----
    This implementation matches the original crop-generation layout:
    - origins are generated on a regular row-major grid
    - stride is ``int(crop_size * (1 - overlap))``
    - only crops that fully fit are included
    - remaining border margins are left unused
    """
    print("DEBUG full_image_shape:", full_image_shape)
    if len(full_image_shape) != 2:
        raise ValueError(f"full_image_shape must have length 2, got {full_image_shape}")
    if len(crop_size) != 2:
        raise ValueError(f"crop_size must have length 2, got {crop_size}")

    full_rows, full_cols = full_image_shape
    crop_rows, crop_cols = crop_size

    print("DEBUG full_rows:", full_rows)
    print("DEBUG full_cols:", full_cols)
    print("DEBUG crop_rows:", crop_rows)
    print("DEBUG crop_cols:", crop_cols)    

    if crop_rows > full_rows or crop_cols > full_cols:
        raise ValueError(
            f"Crop size {crop_size} is larger than full image shape {full_image_shape}"
        )

    stride_rows = int(crop_rows * (1.0 - overlap))
    stride_cols = int(crop_cols * (1.0 - overlap))

    row_positions = list(range(0, full_rows - crop_rows + 1, stride_rows))
    col_positions = list(range(0, full_cols - crop_cols + 1, stride_cols))

    origins: list[tuple[int, int]] = []
    for row0 in row_positions:
        for col0 in col_positions:
            origins.append((row0, col0))

    return origins

def resolve_crop_descriptors(cfg) -> list[dict]:
    """
    Resolve all crop descriptors for the current run.

    Returns
    -------
    list[dict]
        List of crop descriptor dictionaries. Each descriptor contains:
        - ``crop_dir``
        - ``crop_index``
        - ``origin_rc``
        - ``crop_size``
    """
    crops = cfg.crops

    if crops.mode == "load":
        if crops.selection == "single":
            return [resolve_loaded_crop_descriptor_from_dir(cfg, crops.crop_dir)]

        if crops.selection == "list":
            return [resolve_loaded_crop_descriptor_from_dir(cfg, crop_dir) for crop_dir in crops.crop_dirs]

        if crops.selection == "range":
            start_idx, end_idx = crops.crop_range
            return [
                resolve_loaded_crop_descriptor_from_dir(cfg, f"crop_{idx}")
                for idx in range(start_idx, end_idx + 1)
            ]

    if crops.mode == "generate":
        if crops.selection == "single":
            origin_rc = tuple(crops.crop_origin_rc)
            return [{
                "crop_dir": f"generated_r{origin_rc[0]}_c{origin_rc[1]}",
                "crop_index": None,
                "origin_rc": origin_rc,
                "crop_size": crops.crop_size,
            }]

        if crops.selection == "list":
            if crops.crop_origins_rc:
                return [
                    {
                        "crop_dir": f"generated_r{origin[0]}_c{origin[1]}",
                        "crop_index": None,
                        "origin_rc": tuple(origin),
                        "crop_size": crops.crop_size,
                    }
                    for origin in crops.crop_origins_rc
                ]

            if crops.crop_dirs:
                origins = generate_crop_origins(
                    full_image_shape=crops.full_image_shape,
                    crop_size=crops.crop_size,
                    overlap=crops.overlap,
                )

                crop_descs = []
                for crop_dir in crops.crop_dirs:
                    crop_index = parse_crop_index(crop_dir)
                    crop_descs.append(
                        {
                            "crop_dir": crop_dir,
                            "crop_index": crop_index,
                            "origin_rc": origins[crop_index],
                            "crop_size": crops.crop_size,
                        }
                    )
                return crop_descs

        if crops.selection == "range":
            origins = generate_crop_origins(
                full_image_shape=crops.full_image_shape,
                crop_size=crops.crop_size,
                overlap=crops.overlap,
            )
            start_idx, end_idx = crops.crop_range
            return [
                {
                    "crop_dir": f"crop_{idx}",
                    "crop_index": idx,
                    "origin_rc": origins[idx],
                    "crop_size": crops.crop_size,
                }
                for idx in range(start_idx, end_idx + 1)
            ]

    raise ValueError(f"Unsupported crops configuration: mode={crops.mode}, selection={crops.selection}")



def resolve_loaded_crop_descriptor_from_dir(cfg, crop_dir: str) -> dict:
    """
    Resolve a crop descriptor for a specific crop directory name.

    Parameters
    ----------
    cfg:
        Validated experiment configuration object.
    crop_dir:
        Crop directory name such as ``"crop_82"``.

    Returns
    -------
    dict
        Dictionary with:
        - ``crop_dir``
        - ``crop_index``
        - ``origin_rc``
        - ``crop_size``
    """
    crop_index = parse_crop_index(crop_dir)

    origins = generate_crop_origins(
        full_image_shape=cfg.crops.full_image_shape,
        crop_size=cfg.crops.crop_size,
        overlap=cfg.crops.overlap,
    )

    origin_rc = origins[crop_index]

    return {
        "crop_dir": crop_dir,
        "crop_index": crop_index,
        "origin_rc": origin_rc,
        "crop_size": cfg.crops.crop_size,
    }