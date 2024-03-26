import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import scipy.ndimage as scn
from scipy.interpolate import interpn
import validation
import filters
from skimage import io
from pyprocess3D_edit import extended_search_area_piv, get_rect_coordinates, \
    get_field_shape
import matplotlib.pyplot as plt
import os
from natsort import natsorted

@dataclass
class PIVSettings:
    correlation_method: str="circular"
    normalized_correlation: bool=False
    windowsizes: Tuple[int, ...]=((16,64,64),(8,32,32),(4,16,16))
    overlap: Tuple[int, ...] =  ((8,32,32),(4,16,16),(2,8,8))
    num_iterations: int = len(windowsizes)
    subpixel_method: str = "gaussian"
    use_vectorized: bool = True
    deformation_method: str = 'symmetric'  # 'symmetric' or 'second image'
    interpolation_order: int=3
    sig2noise_method: Optional[str]="peak2mean"
    sig2noise_mask: int=2 # Only used if peak2peak
    sig2noise_threshold: float=1.0
    sig2noise_validate: bool=True
    validation_first_pass: bool=True
    min_max_u_disp: Tuple=(-50, 50) # currently testing the effect of increasing this threshold (inactivating it)
    min_max_v_disp: Tuple=(-50, 50)
    min_max_w_disp: Tuple=(-50, 50)
    std_threshold: int=10 # TODO: Adjust?changed from 10 to 3 # threshold of the std (global) validation
    median_threshold: int=3  # TODO: This does seem to have an effect
    median_size: int=1  # TODO: This does seem to have an effect
    replace_vectors: bool=True
    filter_method: str="localmean"
    max_filter_iteration: int=9 # TODO: adjust?
    filter_kernel_size: int=2

def transform_coordinates(x, y, z, u, v, w):
    """
    Converts coordinate systems between image based and physical based
    """
    y = y[:, ::-1, :]
    v *= -1
    return x, y, z, u, v, w

# Create a function to slice out zstacks from timeseries based on index
def prepare_images(folder, files,
                   idx: int
                   )-> Tuple[np.ndarray, np.ndarray]:
    frame_a = io.imread(folder + files[idx-1])
    frame_b = io.imread(folder + files[idx])
    return (frame_a, frame_b)

def piv(settings):

    # Read in the timeseries
    folder = "/Users/jojo/Downloads/JuliaPIV/tests/"
    files = natsorted([f for f in os.listdir(folder) if "noplank" in f])

    # Start loop over consecutive images
    for k in range(1, len(files)):

        frame_a, frame_b = prepare_images(
                folder,
                files,
                k
        )

        image_mask = None #frame_a <= 1

        # "first pass"
        x, y, z, u, v, w, s2n = first_pass(
            frame_a,
            frame_b,
            settings
        )
        if image_mask is None:
            grid_mask = np.zeros_like(u, dtype=bool)
        else:
            grid_mask = scn.map_coordinates(image_mask, [z,y,x]).astype(bool)

        # mask the velocity
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)
        w = np.ma.masked_array(w, mask=grid_mask)

        if settings.validation_first_pass:
            flags = validation.typical_validation(u, v, w, s2n, settings)
        else:
            flags = np.zeros_like(u, dtype=bool)

        # "filter to replace the values that where marked by the validation"
        if (settings.num_iterations == 1 and settings.replace_vectors) \
            or (settings.num_iterations > 1):
            u, v, w = validation.replace_outliers(
                u,
                v,
                w,
                flags,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )

            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)
            w = np.ma.masked_array(w, mask=grid_mask)
        # Multi pass
        for i in range(1, settings.num_iterations):

            x, y, z, u, v, w, grid_mask, flags = multipass_img_deform(
                frame_a,
                frame_b,
                i,
                x,
                y,
                z,
                u,
                v,
                w,
                image_mask,
                settings,
            )
            if not isinstance(u, np.ma.MaskedArray):
                raise ValueError('not a masked array anymore')

            if image_mask is not None:
                grid_mask = scn.map_coordinates(image_mask, [z, y, x]).astype(bool)
                u = np.ma.masked_array(u, mask=grid_mask)
                v = np.ma.masked_array(v, mask=grid_mask)
                w = np.ma.masked_array(w, mask=grid_mask)
            else:
                u = np.ma.masked_array(u, np.ma.nomask)
                v = np.ma.masked_array(v, np.ma.nomask)
                w = np.ma.masked_array(w, np.ma.nomask)

        u = u.filled(0.)
        v = v.filled(0.)
        w = w.filled(0.)

        if image_mask is not None:
            grid_mask = scn.map_coordinates(image_mask, [z, y, x]).astype(bool)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)
            w = np.ma.masked_array(w, mask=grid_mask)
        #else:
        #    u = np.ma.masked_array(u, np.ma.nomask)
        #    v = np.ma.masked_array(v, np.ma.nomask)
        #    w = np.ma.masked_array(w, np.ma.nomask)

        # before saving we convert to the right_hand coordinate system
        x, y, z, u, v, w = transform_coordinates(x, y, z, u, v, w)
        timepoint = np.full_like(u, k)

        # Saving
        #out = np.vstack([m.flatten() for m in [x, y, z, u, v, w, grid_mask, flags, timepoint]])
        #out = pd.DataFrame(out.T, columns = ['x', 'y', 'z', 'u', 'v', 'w', 'mask','flag', 'timepoint'])
        #out.to_csv(f'H:/Dispersal/WT_replicate1_processed/{k}.csv')
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        unique_z = np.unique(z)
        start_x, end_x, num_x = unique_x.min(), unique_x.max(), len(unique_x)
        start_y, end_y, num_y = unique_y.min(), unique_y.max(), len(unique_y)
        start_z, end_z, num_z = unique_z.min(), unique_z.max(), len(unique_z)
        linspace_x = np.linspace(start_x, end_x, num_x)
        linspace_y = np.linspace(start_y, end_y, num_y)
        linspace_z = np.linspace(start_z, end_z, num_z)

        np.save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/x.npy", linspace_x)
        np.save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/y.npy", linspace_y)
        np.save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/z.npy", linspace_z)
        np.save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/u.npy", u)
        np.save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/v.npy", v)
        np.save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/w.npy", w)
        np.save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/flags.npy", flags.data)

def create_deformation_field(frame, x, y, z, u, v, w, interpolation_order = 3):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.

    Returns
    -------
        x,y : new grid (after meshgrid)
        u,v : deformation field
    """
    z1 = z[:, 0, 0]  # extract first coloumn from meshgrid
    y1 = y[0, :, 0]  # extract first coloumn from meshgrid
    x1 = x[0, 0, :]  # extract first row from meshgrid
    side_x = np.arange(frame.shape[2])  # extract the image grid
    side_y = np.arange(frame.shape[1])
    side_z = np.arange(frame.shape[0])

    z, y, x = np.meshgrid(side_z, side_y, side_x, indexing='ij')
    # interpolating displacements onto a new meshgrid
    u = np.ascontiguousarray(u)
    v = np.ascontiguousarray(v)
    w = np.ascontiguousarray(w)
    z1 = np.ascontiguousarray(z1)
    y1 = np.ascontiguousarray(y1)
    x1 = np.ascontiguousarray(x1)
    z = np.ascontiguousarray(z)
    y = np.ascontiguousarray(y)
    x = np.ascontiguousarray(x)
    ut = interpn((z1, y1, x1), u, (z, y, x), bounds_error=False, fill_value=None)
    vt = interpn((z1, y1, x1), v, (z, y, x), bounds_error=False, fill_value=None)
    wt = interpn((z1, y1, x1), w, (z, y, x), bounds_error=False, fill_value=None)

    return x, y, z, ut, vt, wt


def deform_windows(frame, x, y, z, u, v, w, interpolation_order=1, interpolation_order2=3,
                   debugging=False):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.

    Returns
    -------
    frame_def:
        a deformed image based on the meshgrid and displacements of the
        previous pass
    """

    frame = frame.astype(np.float32)
    x, y, z, ut, vt, wt, = \
        create_deformation_field(frame,
                                 x, y, z, u, v, w,
                                 interpolation_order=interpolation_order2)
    frame_def = scn.map_coordinates(
        frame, ((z + wt, y - vt, x + ut,)), order=interpolation_order, mode='nearest')

    return frame_def


def first_pass(frame_a, frame_b, settings):
    """
    First pass of the PIV evaluation.
    """

    u, v, w, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=settings.windowsizes[0],
        overlap=settings.overlap[0],
        search_area_size=settings.windowsizes[0],
        width=settings.sig2noise_mask,
        subpixel_method=settings.subpixel_method,
        sig2noise_method=settings.sig2noise_method,
        correlation_method=settings.correlation_method,
        normalized_correlation=settings.normalized_correlation,
        use_vectorized = settings.use_vectorized,
    )

    x, y, z = get_rect_coordinates(frame_a.shape,
                           settings.windowsizes[0],
                           settings.overlap[0])
    return x, y, z, u, v, w, s2n


def multipass_img_deform(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    current_iteration: int,
    x_old: np.ndarray,
    y_old: np.ndarray,
    z_old: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    img_mask: np.ndarray,
    settings: "PIVSettings",
):

    if not isinstance(u_old, np.ma.MaskedArray):
        raise ValueError('Expected masked array')

    window_size = settings.windowsizes[current_iteration] # integer only
    overlap = settings.overlap[current_iteration] # integer only, won't work for rectangular windows

    x, y, z = get_rect_coordinates(frame_a.shape,
                           window_size,
                           overlap)

    # 1D arrays for the interpolation

    z_old = z_old[:, 0, 0]
    y_old = y_old[0, :, 0]
    x_old = x_old[0, 0, :]

    z_int = z[:, 0, 0]
    y_int = y[0, :, 0]
    x_int = x[0, 0, :]
    Z, Y, X = np.meshgrid(z_int, y_int, x_int, indexing='ij')
    u_old_contiguous = np.ascontiguousarray(np.ma.filled(u_old, 0.))
    v_old_contiguous = np.ascontiguousarray(np.ma.filled(v_old, 0.))
    w_old_contiguous = np.ascontiguousarray(np.ma.filled(w_old, 0.))
    z_old = np.ascontiguousarray(z_old)
    y_old = np.ascontiguousarray(y_old)
    x_old = np.ascontiguousarray(x_old)
    Z = np.ascontiguousarray(Z)
    Y = np.ascontiguousarray(Y)
    X = np.ascontiguousarray(X)
    u_pre = interpn((z_old, y_old, x_old), u_old_contiguous, (Z, Y, X), bounds_error=False, fill_value=None)
    v_pre = interpn((z_old, y_old, x_old), v_old_contiguous, (Z, Y, X), bounds_error=False, fill_value=None)
    w_pre = interpn((z_old, y_old, x_old), w_old_contiguous, (Z, Y, X), bounds_error=False, fill_value=None)
    old_frame_a = frame_a.copy()
    old_frame_b = frame_b.copy()

    # Image deformation has to occur in image coordinates
    # therefore we need to convert the results of the
    # previous pass which are stored in the physical units
    # and so y from the get_coordinates

    if settings.deformation_method == "symmetric":
        # this one is doing the image deformation (see above)
        x_new, y_new, z_new, ut, vt, wt = create_deformation_field(
            frame_a, x, y, z, u_pre, v_pre, w_pre)
        frame_a = scn.map_coordinates(
            frame_a, ((z_new - wt / 2, y_new - vt / 2, x_new - ut / 2)),
            order=settings.interpolation_order, mode='nearest')
        frame_b = scn.map_coordinates(
            frame_b, ((z_new + wt / 2, y_new + vt / 2, x_new + ut / 2)),
            order=settings.interpolation_order, mode='nearest')
    elif settings.deformation_method == "second image":
        frame_b = deform_windows(
            frame_b, x, y, z, u_pre, -v_pre, w_pre,
            interpolation_order=settings.interpolation_order)
    else:
        raise Exception("Deformation method is not valid.")

    if settings.sig2noise_validate is False:
        settings.sig2noise_method = None

    u, v, w, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        width=settings.sig2noise_mask,
        subpixel_method=settings.subpixel_method,
        sig2noise_method=settings.sig2noise_method,
        correlation_method=settings.correlation_method,
        normalized_correlation=settings.normalized_correlation,
        use_vectorized = settings.use_vectorized,
    )

    # get_field_shape expects tuples for rectangular windows
    shapes = np.array(get_field_shape(frame_a.shape,
                                      window_size,
                                      overlap)
                                      )
    u = u.reshape(shapes)
    v = v.reshape(shapes)
    w = w.reshape(shapes)
    s2n = s2n.reshape(shapes)

    u += u_pre
    v += v_pre
    w += w_pre

    # reapply the image mask to the new grid
    if img_mask is not None:
        grid_mask = scn.map_coordinates(img_mask, [z, y, x]).astype(bool)
    else:
        grid_mask = np.zeros_like(u, dtype=bool)

    u = np.ma.masked_array(u, mask=grid_mask)
    v = np.ma.masked_array(v, mask=grid_mask)
    w = np.ma.masked_array(w, mask=grid_mask)

    # validate in the multi-pass by default
    flags = validation.typical_validation(u, v, w, s2n, settings)

    if np.all(flags):
        raise ValueError("Something happened in the validation")

    # we have to replace outliers
    u, v, w = validation.replace_outliers(
        u,
        v,
        w,
        flags,
        method=settings.filter_method,
        max_iter=settings.max_filter_iteration,
        kernel_size=settings.filter_kernel_size,
    )

    return x, y, z, u, v, w, grid_mask, flags

if __name__ == "__main__":
    settings = PIVSettings()
    piv(settings)
