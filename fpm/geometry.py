"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

-----------------------------------------------------------------------------
Optical and LED-array geometry helpers for FPM.

This module converts the experiment setup configuration into the tensors
needed by the reconstruction pipeline:
- pupil support mask
- LED geometry and illumination NA
- LED frequency shifts
- object-grid size
- propagation kernel H0
"""

from __future__ import annotations

import torch

from .config_models import SetupConfig
from fpm.utils import round_matlab


def compute_setup_tensors(
    setup_cfg: SetupConfig,
    crop_desc: dict,
    full_image_shape: list[int],
    coords: torch.Tensor,
) -> dict:
    """
    Compute all setup-dependent tensors and derived parameters for one crop.

    Parameters
    ----------
    setup_cfg:
        Physical microscope and illumination setup configuration.
    crop_desc:
        Resolved crop descriptor for the active crop. This dictionary must
        contain:
        - ``crop_dir``: crop directory name
        - ``crop_index``: integer crop index
        - ``origin_rc``: top-left crop origin as ``(row0, col0)``
        - ``crop_size``: crop size as ``[rows, cols]``
    full_image_shape:
        Full sensor image shape as ``[rows, cols]``.
    coords:
        LED coordinates tensor of shape ``[nled, 2]`` containing the LED
        positions in the specified convention (e.g. "xy" or "rc"). The
        coordinate convention and units are determined by the setup config
        fields:
        - ``led_coords_convention``: either "xy" or "rc"
        - ``led_coord_unit_um``: physical size of one coordinate unit in microns
        - ``led_coord_center``: coordinate of the LED array center in the same units
        - ``led_axis_signs``: signs to apply to the coordinates for correct orientation,
          as a list of two elements corresponding to the coordinate order (e.g. [sx, sy] where sx is the sign for the first coordinate and sy for the second)       

    Returns
    -------
    dict
        Dictionary containing derived tensors corresponding to the actual used LEDs from coords.

    Notes
    -----
    This function is crop-aware: the LED geometry is computed using the
    crop center relative to the full-frame sensor center. This is required
    for correct off-axis geometry when reconstructing non-central crops.
    """
    lambda_ = setup_cfg.wavelength_um
    na = setup_cfg.objective_na
    mag = setup_cfg.magnification
    dpix_c = setup_cfg.camera_pixel_um
    dpix_m = dpix_c / mag

    npatch = crop_desc["crop_size"]
    nrows, ncols = npatch

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"coords must have shape (N, 2), got {tuple(coords.shape)}"
        )

    um_m = na / lambda_

    fov_idx = nrows * dpix_m
    fov_idy = ncols * dpix_m

    if nrows % 2 == 1:
        du = 1 / dpix_m / (nrows - 1)
        dv = 1 / dpix_m / (ncols - 1)
    else:
        du = 1 / fov_idx
        dv = 1 / fov_idy

    m_x = torch.arange(1, nrows + 1)
    m_y = torch.arange(1, ncols + 1)
    mm, nn = torch.meshgrid(
        m_x - round_matlab(torch.tensor((nrows + 1) / 2)),
        m_y - round_matlab(torch.tensor((ncols + 1) / 2)),
        indexing="xy",
    )

    ridx = torch.sqrt(mm**2 + nn**2)
    um_idx = um_m / du

    w_na = (ridx < um_idx).to(torch.float32)
    phc = torch.ones((nrows, ncols))
    aberration = torch.ones((nrows, ncols))
    pupil = w_na * phc * aberration

    crop_rows, crop_cols = crop_desc["crop_size"]
    crop_row0, crop_col0 = crop_desc["origin_rc"]

    full_rows, full_cols = full_image_shape

    full_center_rc = [full_rows / 2, full_cols / 2]
    crop_center_rc = [crop_row0 + crop_rows / 2, crop_col0 + crop_cols / 2]

    img_center = [
        (crop_center_rc[0] - full_center_rc[0]) * dpix_m,
        (crop_center_rc[1] - full_center_rc[1]) * dpix_m,
    ]

    z_led = setup_cfg.led_distance_um

    coord_a = coords[:, 0].to(torch.float32)
    coord_b = coords[:, 1].to(torch.float32)

    if setup_cfg.led_coords_convention == "xy":
        coord_x = coord_a
        coord_y = coord_b
    elif setup_cfg.led_coords_convention == "rc":
        coord_x = coord_b
        coord_y = coord_a
    else:
        raise ValueError(f"Unsupported led_coords_convention: {setup_cfg.led_coords_convention}")

    x_center = float(setup_cfg.led_coord_center[0])
    y_center = float(setup_cfg.led_coord_center[1])

    sx = float(setup_cfg.led_axis_signs[0])
    sy = float(setup_cfg.led_axis_signs[1])

    if len(setup_cfg.led_axis_signs) != 2:
        raise ValueError(
            f"led_axis_signs must have length 2, got {setup_cfg.led_axis_signs}"
        )

    led_x_um = sx * (coord_x - x_center) * setup_cfg.led_coord_unit_um
    led_y_um = sy * (coord_y - y_center) * setup_cfg.led_coord_unit_um

    dd_used = torch.sqrt(
        (led_x_um - img_center[0]) ** 2
        + (led_y_um - img_center[1]) ** 2
        + z_led**2
    )

    # Keep this axis mapping aligned with the original reconstruction convention:
    # vertical-angle term uses led_x_um, and horizontal-angle term uses led_y_um.
    # If results look mirrored or rotated, this mapping/sign convention is the
    # first place to verify.
    sin_thetav_used = (led_x_um - img_center[0]) / dd_used
    sin_thetah_used = (led_y_um - img_center[1]) / dd_used
    illumination_na_used = torch.sqrt(sin_thetav_used**2 + sin_thetah_used**2)

    vled_spatial_used = sin_thetav_used / lambda_
    uled_spatial_used = sin_thetah_used / lambda_

    idx_u_used = round_matlab(uled_spatial_used / du).to(torch.int64)
    idx_v_used = round_matlab(vled_spatial_used / dv).to(torch.int64)

    nled_used = int(coords.shape[0])
    nbf = torch.sum(illumination_na_used < na) 

    um_p = torch.max(illumination_na_used) / lambda_ + um_m
    dx0_p = 1 / um_p / 2
    n_obj = [
        int(torch.ceil(round_matlab(2 * um_p / du) * 2 / nrows) * nrows),
        int(torch.ceil(round_matlab(2 * um_p / dv) * 2 / ncols) * ncols),
    ]

    um_obj = [du * n_obj[0] / 2, dv * n_obj[1] / 2]
    dx_obj = [1 / um_obj[0] / 2, 1 / um_obj[1] / 2]

    u = torch.arange(-um_obj[0], um_obj[0], du)
    v = torch.arange(-um_obj[1], um_obj[1], dv)
    u, v = torch.meshgrid(u, v, indexing="xy")

    z0 = setup_cfg.z0
    h0 = torch.exp(torch.tensor(1j * 2 * torch.pi * z0 / lambda_, dtype=torch.complex64)) * \
         torch.exp(-1j * torch.pi * lambda_ * z0 * (u**2 + v**2))
    

    return {
    "lambda_": lambda_,
    "na": na,
    "mag": mag,
    "dpix_m": dpix_m,
    "du": du,
    "dv": dv,
    "w_NA": w_na,
    "pupil": pupil,
    "idx_u_used": idx_u_used,
    "idx_v_used": idx_v_used,
    "illumination_na_used": illumination_na_used,
    "NBF": nbf,
    "um_p": um_p,
    "dx0_p": dx0_p,
    "N_obj": n_obj,
    "um_obj": um_obj,
    "dx_obj": dx_obj,
    "H0": h0,
    "img_center": img_center,
    "nled_used": nled_used,
    "led_x_um": led_x_um,
    "led_y_um": led_y_um,
    }