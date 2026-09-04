# SPDX-License-Identifier: MIT
"""Render synthetic Schlieren observations from CFD flowfields."""

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

def _masked_gaussian_filter(values, valid, sigma):
    """Apply a normalized Gaussian blur over valid image pixels.

    Parameters
    ----------
    values : numpy.ndarray
        Image values with shape ``(height, width)``.
    valid : numpy.ndarray
        Boolean mask with the same shape as ``values``.
    sigma : float
        Gaussian standard deviation in pixels.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape as ``values``.
    """

    if sigma <= 0.0:
        return values.copy()

    numerator = gaussian_filter(
        np.where(valid, values, 0.0),
        sigma=sigma,
        mode="nearest",
    )

    denominator = gaussian_filter(
        valid.astype(float),
        sigma=sigma,
        mode="nearest",
    )

    filtered = np.zeros_like(values, dtype=float)

    np.divide(
        numerator,
        denominator,
        out=filtered,
        where=denominator > np.finfo(float).eps,
    )

    return filtered

def _downsample_valid_pixels(
    values,
    valid,
    height,
    width,
    supersampling,
):
    """Average valid subpixels into the final camera resolution.

    Parameters
    ----------
    values : numpy.ndarray
        Supersampled image with shape ``(height * supersampling, width *
        supersampling)``.
    valid : numpy.ndarray
        Boolean validity mask with the same shape as ``values``.
    height : int
        Final image height in pixels.
    width : int
        Final image width in pixels.
    supersampling : int
        Number of subpixels per final-pixel axis.

    Returns
    -------
    final_values : numpy.ndarray
        Downsampled values with shape ``(height, width)``.
    final_valid : numpy.ndarray
        Boolean mask identifying final pixels whose subpixels are all valid.
    """

    if supersampling == 1:
        return values.copy(), valid.copy()

    block_values = np.where(valid, values, 0.0).reshape(
        height,
        supersampling,
        width,
        supersampling,
    )

    block_valid = valid.astype(float).reshape(
        height,
        supersampling,
        width,
        supersampling,
    )

    value_sum = block_values.sum(axis=(1, 3))
    valid_count = block_valid.sum(axis=(1, 3))

    number_of_subpixels = supersampling**2

    # Accept a final pixel only when every subpixel is inside
    # the CFD domain.
    final_valid = valid_count == number_of_subpixels

    final_values = np.zeros(
        (height, width),
        dtype=float,
    )

    np.divide(
        value_sum,
        valid_count,
        out=final_values,
        where=valid_count > 0.0,
    )

    return final_values, final_valid

def generate_synthetic_schlieren(
    file_path,
    image,
    rendering,
    density_name="rho",
    grad_type="magnitude",
    dims=2,
):
    """Generate synthetic Schlieren observations on a camera grid.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CFD VTK file.
    image : dict
        Image settings containing ``resolution`` and optionally
        ``supersampling``.
    rendering : dict
        Rendering settings containing ``blur_sigma_pixels`` and
        ``normalization_percentile``.
    density_name : str, optional
        Name of the CFD density field.
    grad_type : {"grad_x", "grad_y", "magnitude"}
        Density-gradient quantity used as the Schlieren signal.
    dims : int, optional
        Spatial dimension. Synthetic camera rendering currently supports only
        two-dimensional flowfields.

    Returns
    -------
    dict
        Valid final-pixel coordinates under ``"x"`` and ``"y"``, together
        with the selected density-gradient signal under ``grad_type``.
    """

    if dims != 2:
        raise NotImplementedError(
            "Synthetic camera rendering currently supports 2D CFD"
        )

    valid_gradient_types = {"grad_x", "grad_y", "magnitude"}

    if grad_type not in valid_gradient_types:
        raise ValueError(
            f"Invalid Schlieren gradient type: {grad_type}. "
        )

    # Load CFD mesh.
    mesh = pv.read(Path(file_path))
    # Bounds
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, mesh.bounds)

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            "The CFD mesh has invalid x or y bounds"
        )

    # Determine where density is stored.
    if density_name in mesh.point_data:
        association = "point"
    elif density_name in mesh.cell_data:
        association = "cell"
    else:
        raise KeyError(
            f"Density variable {density_name} was not found. "
            f"Available arrays: {mesh.array_names}"
        )

    # Compute the density gradient.
    mesh_with_gradient = mesh.compute_derivative(
        scalars=density_name,
        gradient="density_gradient",
        preference=association,
    )

    # Convert cell gradients to point data for interpolation.
    if "density_gradient" in mesh_with_gradient.cell_data:
        mesh_with_gradient = (
            mesh_with_gradient.cell_data_to_point_data()
        )

    # Check if it was possible to convert the cell gradients
    # to points data.
    if "density_gradient" not in mesh_with_gradient.point_data:
        raise RuntimeError(
            "Could not create a point-associated density gradient"
        )

    # Read image configuration.
    # width, height
    resolution = image.get("resolution") 

    if resolution is None or len(resolution) != 2:
        raise ValueError(
            "image['resolution'] must be [width, height]"
        )

    width = int(resolution[0])
    height = int(resolution[1])

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image width and height must be positive"
        )

    supersampling = int(
        image.get("supersampling", 1)
    )

    if supersampling <= 0:
        raise ValueError(
            "image['supersampling'] must be positive"
        )

    # Read rendering configuration.
    blur_sigma_pixels = float(
        rendering.get("blur_sigma_pixels", 0.8)
    )

    normalization_percentile = float(
        rendering.get(
            "normalization_percentile",
            99.0,
        )
    )

    if blur_sigma_pixels < 0.0:
        raise ValueError(
            "blur_sigma_pixels cannot be negative"
        )

    if not 0.0 < normalization_percentile <= 100.0:
        raise ValueError(
            "normalization_percentile must be in (0, 100]"
        )

    # Internal supersampled resolution.
    render_width = width * supersampling
    render_height = height * supersampling

    dx = (xmax - xmin) / render_width
    dy = (ymax - ymin) / render_height

    # Internal subpixel-center coordinates.
    x_render = xmin + (np.arange(render_width) + 0.5) * dx
    y_render = ymin + (np.arange(render_height) + 0.5) * dy

    xx_render, yy_render = np.meshgrid(x_render, y_render)

    # VTK requires three-coordinate points.
    z_plane = 0.5 * (zmin + zmax)

    # Create camera points
    # NumPy array containing only numerical coordinates
    camera_points = np.column_stack([
        xx_render.ravel(), # Flatten
        yy_render.ravel(), # Flatten
        np.full(xx_render.size, z_plane),
    ])

    # Interpolate the mesh gradient at camera locations.
    # PyVista/VTK point-cloud object that can interact with 
    # CFD meshes
    camera_cloud = pv.PolyData(camera_points)
    sampled = camera_cloud.sample(mesh_with_gradient)

    if "vtkValidPointMask" not in sampled.point_data:
        raise RuntimeError(
            "PyVista did not return vtkValidPointMask"
        )

    if "density_gradient" not in sampled.point_data:
        raise RuntimeError(
            "Density gradient was not interpolated "
            "onto the camera points"
        )

    # Get valid points, which are inside the geometry
    valid = np.asarray(sampled["vtkValidPointMask"], dtype=bool)

    # Make sampled["density_gradient"] a numpy array
    gradient = np.asarray(sampled["density_gradient"],dtype=float)

    if gradient.shape != (camera_points.shape[0], 3):
        raise ValueError(
            "Unexpected density-gradient shape: "
            f"{gradient.shape}"
        )

    # Select the requested Schlieren quantity.
    if grad_type == "grad_x":
        signal = gradient[:, 0]
    elif grad_type == "grad_y":
        signal = gradient[:, 1]
    else:
        signal = np.linalg.norm(gradient[:, :2], axis=1)

    # Restore the internal image shape.
    signal = signal.reshape(render_height, render_width)

    valid = valid.reshape(render_height, render_width)

    if not np.any(valid):
        raise ValueError(
            "No camera pixels lie inside the CFD geometry"
        )

    # Apply optical blur.
    signal = _masked_gaussian_filter(
        values=signal,
        valid=valid,
        sigma=(blur_sigma_pixels * supersampling),
    )

    # Average internal subpixels into final pixels.
    signal, valid = _downsample_valid_pixels(
        values=signal,
        valid=valid,
        height=height,
        width=width,
        supersampling=supersampling,
    )

    if not np.any(valid):
        raise ValueError(
            "No valid final camera pixels remain"
        )

    # Normalize the clean signal.
    signal_scale = np.percentile(
        np.abs(signal[valid]),
        normalization_percentile,
    )

    if (
        not np.isfinite(signal_scale)
        or signal_scale <= 0.0
    ):
        raise ValueError(
            "The Schlieren signal has zero or invalid scale"
        )

    signal = signal / signal_scale

    if grad_type == "magnitude":
        signal = np.clip(signal, 0.0, 1.0)
    else:
        signal = np.clip(signal, -1.0, 1.0)

    # Final camera-pixel coordinates.
    final_dx = (xmax - xmin) / width
    final_dy = (ymax - ymin) / height

    x = xmin + (np.arange(width) + 0.5) * final_dx
    y = ymin + (np.arange(height) + 0.5) * final_dy
    xx, yy = np.meshgrid(x, y)

    # Return only what _prepare_schlieren() consumes.
    return {
        "x": xx[valid],
        "y": yy[valid],
        grad_type: signal[valid],
    }
