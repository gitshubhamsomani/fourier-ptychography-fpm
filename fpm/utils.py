"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

---------------------------------------
Utility functions for FPM reconstruction and image I/O.
This module includes:
- EXR loading and saving helpers    
- Fourier transform helpers
- array padding and cropping helpers
- conversion to uint16 with specified dynamic range
- Saving floating-point grayscale imaeges as EXR
"""
import torch
import OpenEXR
import Imath
import array
import os
import re
import imageio.v2 as imageio
from pathlib import Path
import numpy as np

def load_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (_, G, _) = [array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
    G = torch.tensor(G, dtype=torch.float32).reshape(size[1], size[0])
    return G

def natsortfiles(file_list):
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\\d+)', text)]
    return sorted(file_list, key=natural_keys)

def F(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

def Ft(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))

def padarray(x, pad_shape):
    pad_y = (pad_shape[0] - x.shape[0]) // 2
    pad_x = (pad_shape[1] - x.shape[1]) // 2
    return torch.nn.functional.pad(
        x,
        (pad_x, pad_shape[1] - x.shape[1] - pad_x,
         pad_y, pad_shape[0] - x.shape[0] - pad_y)
    )

def crop_center(x, shape):
    y, x_ = x.shape
    sy, sx = shape
    cy, cx = y // 2, x_ // 2
    return x[cy-sy//2:cy+sy//2, cx-sx//2:cx+sx//2]

def round_matlab(x):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

UINT16_MAX = 65535

def to_uint16_minmax(x, vmin=None, vmax=None):
    x = x.detach().cpu().numpy()
    import numpy as np
    if vmin is None:
        vmin = np.nanmin(x)
    if vmax is None:
        vmax = np.nanmax(x)
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint16)
    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0.0, 1.0)
    return (y * UINT16_MAX + 0.5).astype(np.uint16)

def angle_to_uint16(angle_rad):
    angle_rad = angle_rad.detach().cpu().numpy()
    import numpy as np
    y = (angle_rad + np.pi) / (2 * np.pi)
    y = np.clip(y, 0.0, 1.0)
    return (y * UINT16_MAX + 0.5).astype(np.uint16)

def save_u16_png(path, u16_img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, u16_img)

def save_png_matlab_imwrite_like(path: str | Path, image: np.ndarray) -> None:
    """
    Save a floating-point grayscale image in a way that matches MATLAB
    ``imwrite(image, filename)`` behavior for PNG as closely as possible.

    Parameters
    ----------
    path:
        Output PNG path.
    image:
        2D NumPy array. Expected to be ``float32`` or ``float64``.

    Notes
    -----
    MATLAB ``imwrite`` treats floating-point grayscale images as having an
    expected dynamic range of ``[0, 1]`` for standard image formats such as
    PNG, and writes them as 8-bit values after scaling by 255. Values below
    0 are clipped to 0 and values above 1 are clipped to 1 before the 8-bit
    conversion. This helper reproduces that behavior explicitly so the Python
    output can be compared more directly against MATLAB output.  [oai_citation:2‡MathWorks](https://www.mathworks.com/help/matlab/ref/imwrite.html?utm_source=chatgpt.com)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(image, dtype=np.float64)
    img = np.clip(img, 0.0, 1.0)
    img_u8 = np.round(img * 255.0).astype(np.uint8)

    imageio.imwrite(path, img_u8)


def save_exr_gray_as_rgb(path: str | Path, image: np.ndarray) -> None:
    """
    Save a 2D floating-point grayscale image as an EXR file by replicating the
    grayscale values into the R, G, and B channels.

    Parameters
    ----------
    path:
        Output EXR path.
    image:
        2D NumPy array containing floating-point image data.

    Notes
    -----
    This mirrors the MATLAB workflow where a grayscale image is expanded to
    three identical channels before writing EXR.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {img.shape}")

    height, width = img.shape

    header = OpenEXR.Header(width, height)
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    data = img.astype(np.float32).tobytes()

    exr = OpenEXR.OutputFile(str(path), header)
    exr.writePixels({"R": data, "G": data, "B": data})
    exr.close()    