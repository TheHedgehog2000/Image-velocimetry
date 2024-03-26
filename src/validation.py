import warnings
from typing import Tuple
import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
from filters import replace_nans

def global_val(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    u_thresholds: Tuple[int, int],
    v_thresholds: Tuple[int, int],
    w_thresholds: Tuple[int, int],
    )-> np.ndarray:
    """Eliminate spurious vectors with a global threshold.

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with Nan (Not a Number) if at
    least one of the two velocity components is out of a specified global
    range.

    Returns
    -------
    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """

    warnings.filterwarnings("ignore")

    ind = np.logical_or(
        np.logical_or(u < u_thresholds[0], u > u_thresholds[1]),
        np.logical_or(v < v_thresholds[0], v > v_thresholds[1]),
        np.logical_or(w < w_thresholds[0], w > w_thresholds[1]),
    )

    return ind


def global_std(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    std_threshold: int=5,
    )->np.ndarray:
    """Eliminate spurious vectors with a global threshold defined by the
    standard deviation

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with NaN (Not a Number) if at least
    one of the two velocity components is out of a specified global range.

    Parameters
    ----------
    u : 2d masked np.ndarray
        a two dimensional array containing the u velocity component.

    v : 2d masked np.ndarray
        a two dimensional array containing the v velocity component.

    std_threshold: float
        If the length of the vector (actually the sum of squared components) is
        larger than std_threshold times standard deviation of the flow field,
        then the vector is treated as an outlier. [default = 3]

    Returns
    -------
    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """
    tmpu = np.ma.copy(u).filled(np.nan)
    tmpv = np.ma.copy(v).filled(np.nan)
    tmpw = np.ma.copy(w).filled(np.nan)

    ind = np.logical_or(np.abs(tmpu - np.nanmean(tmpu)) > std_threshold * np.nanstd(tmpu),
                        np.abs(tmpv - np.nanmean(tmpv)) > std_threshold * np.nanstd(tmpv),
                        np.abs(tmpw - np.nanmean(tmpw)) > std_threshold * np.nanstd(tmpw))

    if np.all(ind): # if all is True, something is really wrong
        print('Warning! probably a uniform shift data, do not use this filter')
        ind = ~ind

    return ind


def sig2noise_val(
    s2n: np.ndarray,
    threshold: float=1.0,
    )->np.ndarray:
    """ Marks spurious vectors if signal to noise ratio is below a specified threshold.

    Parameters
    ----------
    u : 2d or 3d np.ndarray
        a two or three dimensional array containing the u velocity component.

    v : 2d or 3d np.ndarray
        a two or three dimensional array containing the v velocity component.

    s2n : 2d np.ndarray
        a two or three dimensional array containing the value  of the signal to
        noise ratio from cross-correlation function.
    w : 2d or 3d np.ndarray
        a two or three dimensional array containing the w (in z-direction)
        velocity component.

    threshold: float
        the signal to noise ratio threshold value.

    Returns
    -------

    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    References
    ----------
    R. D. Keane and R. J. Adrian, Measurement Science & Technology, 1990,
        1, 1202-1215.

    """
    ind = s2n < threshold

    return ind


def local_median_val(u, v, w, u_threshold, v_threshold, w_threshold, size=1):
    """Eliminate spurious vectors with a local median threshold.

    This validation method tests for the spatial consistency of the data.
    Vectors are classified as outliers and replaced with Nan (Not a Number) if
    the absolute difference with the local median is greater than a user
    specified threshold. The median is computed for both velocity components.

    The image masked areas (obstacles, reflections) are marked as masked array:
       u = np.ma.masked(u, flag = image_mask)
    and it should not be replaced by the local median, but remain masked.

    Returns
    -------

    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """

    # kernel footprint
    f = np.ones((2*size+1, 2*size+1, 2*size+1))
    f[size,size,size] = 0

    masked_u = np.where(~u.mask, u.data, np.nan)
    masked_v = np.where(~v.mask, v.data, np.nan)
    masked_w = np.where(~w.mask, w.data, np.nan)

    um = generic_filter(masked_u, np.nanmedian, mode='constant',
                        cval=np.nan, footprint=f)
    vm = generic_filter(masked_v, np.nanmedian, mode='constant',
                        cval=np.nan, footprint=f)
    wm = generic_filter(masked_w, np.nanmedian, mode='constant',
                        cval=np.nan, footprint=f)

    ind = (np.abs((u - um)) > u_threshold) | (np.abs((v - vm)) > v_threshold) | (np.abs((w - wm)) > w_threshold)

    return ind


def typical_validation(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    s2n: np.ndarray,
    settings: "PIVSettings"
    )->np.ndarray:
    """
    validation using gloabl limits and std and local median,
    """

    # Global validation
    flag_g = global_val(u, v, w, settings.min_max_u_disp, settings.min_max_v_disp, settings.min_max_w_disp)

    flag_s = global_std(
        u, v, w, std_threshold=settings.std_threshold
    )
    flag_m = local_median_val(
        u,
        v,
        w,
        u_threshold=settings.median_threshold,
        v_threshold=settings.median_threshold,
        w_threshold=settings.median_threshold,
        size=settings.median_size,
    )

    flag = flag_g | flag_m | flag_s
    flag = flag_m
    if settings.sig2noise_validate:
        flag_s2n = sig2noise_val(s2n, settings.sig2noise_threshold)
        flag += flag_s2n
    return flag

def replace_outliers(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    flags: np.ndarray,
    method: str="localmean",
    max_iter: int=5,
    tol: float=1e-3,
    kernel_size: int=1,
    )-> Tuple[np.ndarray, ...]:

    if not isinstance(u, np.ma.MaskedArray):
        u = np.ma.masked_array(u, mask=np.ma.nomask)

    # store grid_mask for reinforcement
    grid_mask = u.mask.copy()
    u[flags] = np.nan
    v[flags] = np.nan
    w[flags] = np.nan

    uf = replace_nans(
        u, method=method, max_iter=max_iter, tol=tol,
        kernel_size=kernel_size
    )
    vf = replace_nans(
        v, method=method, max_iter=max_iter, tol=tol,
        kernel_size=kernel_size
    )
    wf = replace_nans(
        w, method=method, max_iter=max_iter, tol=tol,
        kernel_size=kernel_size
    )

    uf = np.ma.masked_array(uf, mask=grid_mask)
    vf = np.ma.masked_array(vf, mask=grid_mask)
    wf = np.ma.masked_array(wf, mask=grid_mask)

    return uf, vf, wf
