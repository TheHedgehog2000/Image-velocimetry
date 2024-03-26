from typing import Optional, Tuple, Callable, Union
import numpy as np
from numpy import log
from numpy import ma
from numpy.fft import rfftn as rfftn_, irfftn as irfftn_, fftshift as fftshift_

# CHECKED
def get_field_shape(
    image_size: Tuple[int,int,int],
    search_area_size: Tuple[int,int,int],
    overlap: Tuple[int,int,int],
    )->Tuple[int,int,int]:
    field_shape = (np.array(image_size) - np.array(search_area_size)) // (np.array(search_area_size) - np.array(overlap)) + 1
    return field_shape

# (z,y,x) -> (X,Y,Z)
def get_coordinates(
    image_size: Tuple[int,int,int],
    search_area_size: int,
    overlap: int,
    center_on_field: bool=True
    )->Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # get shape of the resulting flow field as a 3 component array
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)
    # field shape is good (Z,Y,X)
    # compute grid coordinates of the search area window centers
    if len(image_size) == 3:
        x = (
            np.arange(field_shape[2]) * (search_area_size - overlap)
            + (search_area_size) / 2.0
        )
        y = (
            np.arange(field_shape[1]) * (search_area_size - overlap)
            + (search_area_size) / 2.0
        )
        z = (
            np.arange(field_shape[0]) * (search_area_size - overlap)
            + (search_area_size) / 2.0
        )
        if center_on_field is True:
            x += (
                image_size[2]
                - 1
                - ((field_shape[2] - 1) * (search_area_size - overlap) +
                    (search_area_size - 1))
            ) // 2
            y += (
                image_size[1]
                - 1
                - ((field_shape[1] - 1) * (search_area_size - overlap) +
                    (search_area_size - 1))
            ) // 2
            z += (
                image_size[0]
                - 1
                - ((field_shape[0] - 1) * (search_area_size - overlap) +
                    (search_area_size - 1))
            ) // 2
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        return (X, Y, Z)
    else:
        x = (
            np.arange(field_shape[1]) * (search_area_size - overlap)
            + (search_area_size) / 2.0
        )
        y = (
            np.arange(field_shape[0]) * (search_area_size - overlap)
            + (search_area_size) / 2.0
        )
        if center_on_field is True:
            x += (
                image_size[1]
                - 1
                - ((field_shape[1] - 1) * (search_area_size - overlap) +
                    (search_area_size - 1))
            ) // 2
            y += (
                image_size[0]
                - 1
                - ((field_shape[0] - 1) * (search_area_size - overlap) +
                    (search_area_size - 1))
            ) // 2
        Y, X = np.meshgrid(y, x, indexing='ij')
        return (X, Y)

# (z,y,x) -> (X,Y,Z)
def get_rect_coordinates(
    image_size: Tuple[int,int,int],
    window_size: Union[int, Tuple[int,int,int]],
    overlap: Union[int, Tuple[int,int,int]],
    center_on_field: bool=False,
    ):
    if len(image_size) == 3:
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        if isinstance(overlap, int):
            overlap = (overlap, overlap, overlap)

        _, y, _ = get_coordinates(image_size, window_size[1], overlap[1], center_on_field=center_on_field)
        x, _, _ = get_coordinates(image_size, window_size[2], overlap[2], center_on_field=center_on_field)
        _, _, z = get_coordinates(image_size, window_size[0], overlap[0], center_on_field=center_on_field)
        Z, Y, X = np.meshgrid(z[:,0,0], y[0,:,0], x[0,0,:], indexing='ij')
        return (X, Y, Z)
    else:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if isinstance(overlap, int):
            overlap = (overlap, overlap)
        _, y = get_coordinates(image_size, window_size[0], overlap[0], center_on_field=center_on_field)
        x, _ = get_coordinates(image_size, window_size[1], overlap[1], center_on_field=center_on_field)
        Y, X = np.meshgrid(y[:,0], x[0,:], indexing='ij')
        return (X, Y)


# CHECKED
def sliding_window_array(
    image: np.ndarray,
    window_size: Tuple[int,int,int],
    overlap: Tuple[int,int,int],
    )-> np.ndarray:
    if len(image.shape) == 3:
        x, y, z = get_rect_coordinates(image.shape, window_size, overlap, center_on_field = False)
        x = (x - window_size[2]//2).astype(int)
        y = (y - window_size[1]//2).astype(int)
        z = (z - window_size[0]//2).astype(int)
        x, y, z = np.reshape(x, (-1,1,1,1)), np.reshape(y, (-1,1,1,1)), np.reshape(z, (-1,1,1,1))

        win_z, win_y, win_x = np.meshgrid(np.arange(0, window_size[0]), np.arange(0, window_size[1]), np.arange(0, window_size[2]), indexing='ij')
        win_x = win_x[np.newaxis,:,:,:] + x
        win_y = win_y[np.newaxis,:,:,:] + y
        win_z = win_z[np.newaxis,:,:,:] + z
        windows = image[win_z, win_y, win_x]
    else:
        x, y = get_rect_coordinates(image.shape, window_size, overlap, center_on_field = False)
        x = (x - window_size[1]//2).astype(int)
        y = (y - window_size[0]//2).astype(int)
        x, y = np.reshape(x, (-1,1,1)), np.reshape(y, (-1,1,1))

        win_y, win_x = np.meshgrid(np.arange(0, window_size[1]), np.arange(0, window_size[0]), indexing='ij')
        win_x = win_x[np.newaxis,:,:] + x
        win_y = win_y[np.newaxis,:,:] + y
        windows = image[win_y, win_x]
    return windows

# CHECKED
def find_all_first_peaks(corr):
    if len(corr.shape) == 4:
        ind = corr.reshape(corr.shape[0], -1).argmax(-1)
        peaks = np.array(np.unravel_index(ind, corr.shape[-3:]))
        peaks = np.vstack((peaks[0], peaks[1], peaks[2])).T
        index_list = [(i, v[0], v[1], v[2]) for i, v in enumerate(peaks)]
        peaks_max = np.nanmax(corr, axis = (-3, -2, -1))
    else:
        ind = corr.reshape(corr.shape[0], -1).argmax(-1)
        peaks = np.array(np.unravel_index(ind, corr.shape[-2:]))
        peaks = np.vstack((peaks[0], peaks[1])).T
        index_list = [(i, v[0], v[1]) for i, v in enumerate(peaks)]
        peaks_max = np.nanmax(corr, axis = (-2, -1))
    return np.array(index_list), np.array(peaks_max)


def find_all_second_peaks(corr, width = 2):
    '''
    Returns
    -------
        index_list : integers, index of the peak position in (N,z,i,j)
        peaks_max  : amplitude of the peak
    '''
    if len(corr.shape) == 4:
        indexes = find_all_first_peaks(corr)[0].astype(int)
        ind = indexes[:, 0]
        z = indexes[:, 1]
        x = indexes[:, 2]
        y = indexes[:, 3]
        iini = x - width
        ifin = x + width + 1
        jini = y - width
        jfin = y + width + 1
        zini = z - width
        zfin = z + width + 1
        zini[zini < 0] = 0 # border checking
        zfin[zfin > corr.shape[1]] = corr.shape[1]
        iini[iini < 0] = 0 # border checking
        ifin[ifin > corr.shape[2]] = corr.shape[2]
        jini[jini < 0] = 0
        jfin[jfin > corr.shape[3]] = corr.shape[3]
        tmp = corr.view(np.ma.MaskedArray)
        for i in ind:
            tmp[i, zini[i]:zfin[i], iini[i]:ifin[i], jini[i]:jfin[i]] = np.ma.masked
        indexes, peaks = find_all_first_peaks(tmp)
    else:
        indexes = find_all_first_peaks(corr)[0].astype(int)
        ind = indexes[:, 0]
        y = indexes[:, 1]
        x = indexes[:, 2]
        iini = x - width
        ifin = x + width + 1
        jini = y - width
        jfin = y + width + 1
        iini[iini < 0] = 0 # border checking
        ifin[ifin > corr.shape[1]] = corr.shape[1]
        jini[jini < 0] = 0
        jfin[jfin > corr.shape[2]] = corr.shape[2]
        tmp = corr.view(np.ma.MaskedArray)
        for i in ind:
            tmp[i, iini[i]:ifin[i], jini[i]:jfin[i]] = np.ma.masked
        indexes, peaks = find_all_first_peaks(tmp)
    return indexes, peaks

def vectorized_sig2noise_ratio(correlation,
                               sig2noise_method = 'peak2peak',
                               width = 2):
    if len(correlation.shape) == 4:
        if sig2noise_method == "peak2peak":
            ind1, peaks1 = find_all_first_peaks(correlation)
            ind2, peaks2 = find_all_second_peaks(correlation, width = width)
            peaks1_z, peaks1_i, peaks1_j = ind1[:, 1], ind1[:, 2], ind1[:, 3]
            peaks2_z, peaks2_i, peaks2_j = ind2[:, 1], ind2[:, 2], ind2[:, 3]
            # peak checking
            flag = np.zeros(peaks1.shape).astype(bool)
            flag[peaks1 < 1e-3] = True
            flag[peaks1_z == 0] = True
            flag[peaks1_z == correlation.shape[1]-1] = True
            flag[peaks1_i == 0] = True
            flag[peaks1_i == correlation.shape[2]-1] = True
            flag[peaks1_j == 0] = True
            flag[peaks1_j == correlation.shape[3]-1] = True
            flag[peaks2 < 1e-3] = True
            flag[peaks2_z == 0] = True
            flag[peaks2_z == correlation.shape[1]-1] = True
            flag[peaks2_i == 0] = True
            flag[peaks2_i == correlation.shape[2]-1] = True
            flag[peaks2_j == 0] = True
            flag[peaks2_j == correlation.shape[3]-1] = True
            # peak-to-peak calculation
            peak2peak = np.divide(
                peaks1, peaks2,
                out=np.zeros_like(peaks1),
                where=(peaks2 > 0.0)
            )
            peak2peak[flag is True] = 0 # replace invalid values
            return peak2peak

        elif sig2noise_method == "peak2mean":
            peaks, peaks1max = find_all_first_peaks(correlation)
            peaks = np.array(peaks)
            peaks1_z, peaks1_i, peaks1_j = peaks[:,1], peaks[:, 2], peaks[:, 3]
            peaks2mean = np.abs(np.nanmean(correlation, axis = (-3, -2, -1)))
            # peak checking
            flag = np.zeros(peaks1max.shape).astype(bool)
            flag[peaks1max < 1e-3] = True
            flag[peaks1_z == 0] = True
            flag[peaks1_z == correlation.shape[1]-1] = True
            flag[peaks1_i == 0] = True
            flag[peaks1_i == correlation.shape[2]-1] = True
            flag[peaks1_j == 0] = True
            flag[peaks1_j == correlation.shape[3]-1] = True
            # peak-to-mean calculation
            peak2mean = np.divide(
                peaks1max, peaks2mean,
                out=np.zeros_like(peaks1max),
                where=(peaks2mean > 0.0)
            )
            peak2mean[flag is True] = 0 # replace invalid values
            return peak2mean
        else:
            raise ValueError(f"sig2noise_method not supported: {sig2noise_method}")
    else:
        if sig2noise_method == "peak2peak":
            ind1, peaks1 = find_all_first_peaks(correlation)
            ind2, peaks2 = find_all_second_peaks(correlation, width = width)
            peaks1_i, peaks1_j = ind1[:, 1], ind1[:, 2]
            peaks2_i, peaks2_j = ind2[:, 1], ind2[:, 2]
            # peak checking
            flag = np.zeros(peaks1.shape).astype(bool)
            flag[peaks1 < 1e-3] = True
            flag[peaks1_i == 0] = True
            flag[peaks1_i == correlation.shape[1]-1] = True
            flag[peaks1_j == 0] = True
            flag[peaks1_j == correlation.shape[2]-1] = True
            flag[peaks2 < 1e-3] = True
            flag[peaks2_i == 0] = True
            flag[peaks2_i == correlation.shape[1]-1] = True
            flag[peaks2_j == 0] = True
            flag[peaks2_j == correlation.shape[2]-1] = True
            # peak-to-peak calculation
            peak2peak = np.divide(
                peaks1, peaks2,
                out=np.zeros_like(peaks1),
                where=(peaks2 > 0.0)
            )
            peak2peak[flag is True] = 0 # replace invalid values
            return peak2peak

        elif sig2noise_method == "peak2mean":
            peaks, peaks1max = find_all_first_peaks(correlation)
            peaks = np.array(peaks)
            peaks1_i, peaks1_j = peaks[:,1], peaks[:, 2]
            peaks2mean = np.abs(np.nanmean(correlation, axis = (-2, -1)))
            # peak checking
            flag = np.zeros(peaks1max.shape).astype(bool)
            flag[peaks1max < 1e-3] = True
            flag[peaks1_i == 0] = True
            flag[peaks1_i == correlation.shape[1]-1] = True
            flag[peaks1_j == 0] = True
            flag[peaks1_j == correlation.shape[2]-1] = True
            # peak-to-mean calculation
            peak2mean = np.divide(
                peaks1max, peaks2mean,
                out=np.zeros_like(peaks1max),
                where=(peaks2mean > 0.0)
            )
            peak2mean[flag is True] = 0 # replace invalid values
            return peak2mean
        else:
            raise ValueError(f"sig2noise_method not supported: {sig2noise_method}")



def fft_correlate_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    correlation_method: str="circular",
    normalized_correlation: bool=True,
    conj: Callable=np.conj,
    rfftn = rfftn_,
    irfftn = irfftn_,
    fftshift = fftshift_,
    )->np.ndarray:

    if len(image_a.shape) == 4:
        if normalized_correlation:
            image_a = normalize_intensity(image_a[-3:])
            image_b = normalize_intensity(image_b[-3:])

        s1 = np.array(image_a.shape)
        s2 = np.array(image_b.shape)

        if correlation_method == "circular":
            f2a = conj(rfftn(image_a, axes=(-3,-2,-1)))
            f2b = rfftn(image_b, axes=(-3,-2,-1))
            corr = fftshift(irfftn(f2a * f2b, axes=(-3,-2,-1)).real, axes=(-3, -2, -1))
        else:
            print(f"correlation method {correlation_method } is not implemented")

        if normalized_correlation:
            corr = corr/(s2[0]*s2[1]*s2[2])
            corr = np.clip(corr, 0, 1)
    else:
        if normalized_correlation:
            image_a = normalize_intensity(image_a[-2:])
            image_b = normalize_intensity(image_b[-2:])

        s1 = np.array(image_a.shape)
        s2 = np.array(image_b.shape)

        if correlation_method == "circular":
            f2a = conj(rfftn(image_a, axes=(-2,-1)))
            f2b = rfftn(image_b, axes=(-2,-1))
            corr = fftshift(irfftn(f2a * f2b, axes=(-2,-1)).real, axes=(-2, -1))
        else:
            print(f"correlation method {correlation_method } is not implemented")

        if normalized_correlation:
            corr = corr/(s2[0]*s2[1])
            corr = np.clip(corr, 0, 1)
    return corr


def normalize_intensity(window):
    window = window.astype(np.float32)
    window -= window.mean(axis=(-3, -2, -1),
                          keepdims=True, dtype=np.float32)
    tmp = window.std(axis=(-3, -2, -1), keepdims=True)
    window = np.divide(window, tmp, out=np.zeros_like(window),
                       where=(tmp != 0))
    return np.clip(window, 0, window.max())


def extended_search_area_piv(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    window_size: Union[int, Tuple[int,int,int]],
    overlap: Union[int, Tuple[int,int,int]],
    dt: float=1.0,
    search_area_size: Optional[Union[int, Tuple[int,int,int]]]=None,
    correlation_method: str="circular",
    subpixel_method: str="gaussian",
    sig2noise_method: Union[str, None]='peak2mean',
    width: int=2,
    normalized_correlation: bool=False,
    use_vectorized: bool=False,
):
    if len(frame_a.shape) == 3:
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        if isinstance(overlap, int):
            overlap = (overlap, overlap, overlap)

        # if no search_size, copy window_size
        if search_area_size is None:
            search_area_size = window_size
        elif isinstance(search_area_size, int):
            search_area_size = (search_area_size, search_area_size, search_area_size)

        # verify that things are logically possible:
        if overlap[0] >= window_size[0] or overlap[1] >= window_size[1]:
            raise ValueError("Overlap has to be smaller than the window_size")

        if search_area_size[0] < window_size[0] or search_area_size[1] < window_size[1]:
            raise ValueError("Search size cannot be smaller than the window_size")

        if (window_size[1] > frame_a.shape[0]) or (window_size[0] > frame_a.shape[1]):
            raise ValueError("window size cannot be larger than the image")

        # get field shape
        n_layers, n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap)

        # We implement the new vectorized code
        aa = sliding_window_array(frame_a, search_area_size, overlap)
        bb = sliding_window_array(frame_b, search_area_size, overlap)
        corr = fft_correlate_images(aa, bb,
                                    correlation_method=correlation_method,
                                    normalized_correlation=normalized_correlation)
        u, v, w = vectorized_correlation_to_displacements(corr, n_layers, n_rows, n_cols,
                                           subpixel_method=subpixel_method)

        # return output depending if user wanted sig2noise information
        if sig2noise_method is not None:
            if use_vectorized is True:
                sig2noise = vectorized_sig2noise_ratio(
                    corr, sig2noise_method=sig2noise_method, width=width
                )
            else:
                sig2noise = sig2noise_ratio(
                    corr, sig2noise_method=sig2noise_method, width=width
                )
        else:
            sig2noise = np.zeros_like(u)*np.nan

        sig2noise = sig2noise.reshape(n_layers, n_rows, n_cols)

        return u/dt, v/dt, w/dt, sig2noise
    else:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if isinstance(overlap, int):
            overlap = (overlap, overlap)

        # if no search_size, copy window_size
        if search_area_size is None:
            search_area_size = window_size
        elif isinstance(search_area_size, int):
            search_area_size = (search_area_size, search_area_size)

        # verify that things are logically possible:
        if overlap[0] >= window_size[0] or overlap[1] >= window_size[1]:
            raise ValueError("Overlap has to be smaller than the window_size")

        if search_area_size[0] < window_size[0] or search_area_size[1] < window_size[1]:
            raise ValueError("Search size cannot be smaller than the window_size")

        if (window_size[1] > frame_a.shape[0]) or (window_size[0] > frame_a.shape[1]):
            raise ValueError("window size cannot be larger than the image")

        # get field shape
        n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap)

        # We implement the new vectorized code
        aa = sliding_window_array(frame_a, search_area_size, overlap)
        bb = sliding_window_array(frame_b, search_area_size, overlap)
        corr = fft_correlate_images(aa, bb,
                                    correlation_method=correlation_method,
                                    normalized_correlation=normalized_correlation)
        u, v = vectorized_correlation_to_displacements(corr, n_rows, n_cols,
                                           subpixel_method=subpixel_method)

        # return output depending if user wanted sig2noise information
        if sig2noise_method is not None:
            if use_vectorized is True:
                sig2noise = vectorized_sig2noise_ratio(
                    corr, sig2noise_method=sig2noise_method, width=width
                )
            else:
                sig2noise = sig2noise_ratio(
                    corr, sig2noise_method=sig2noise_method, width=width
                )
        else:
            sig2noise = np.zeros_like(u)*np.nan

        sig2noise = sig2noise.reshape(n_rows, n_cols)

        return u/dt, v/dt, sig2noise


# Key function where coordinates change
def vectorized_correlation_to_displacements(corr: np.ndarray,
                                            n_layers: Optional[int]=None,
                                            n_rows: Optional[int]=None,
                                            n_cols: Optional[int]=None,
                                            subpixel_method: str='gaussian',
                                            eps: float=1e-7
):
    if subpixel_method not in ("gaussian", "centroid", "parabolic"):
        raise ValueError(f"Method not implemented {subpixel_method}")
    if len(corr.shape) == 4:
        corr = corr.astype(np.float32) + eps # avoids division by zero
        peaks = find_all_first_peaks(corr)[0]
        ind, peaks_z, peaks_y, peaks_x = peaks[:,0], peaks[:,1], peaks[:,2], peaks[:,3]
        peaks1_i, peaks1_j, peaks1_z = peaks_y, peaks_x, peaks_z

        # peak checking
        if subpixel_method in ("gaussian", "centroid", "parabolic"):
            mask_width = 1
        invalid = list(np.where(peaks1_z < mask_width)[0])
        invalid += list(np.where(peaks1_z > corr.shape[1] - mask_width - 1)[0])
        invalid += list(np.where(peaks1_i < mask_width)[0])
        invalid += list(np.where(peaks1_i > corr.shape[2] - mask_width - 1)[0])
        invalid += list(np.where(peaks1_j < mask_width)[0])
        invalid += list(np.where(peaks1_j > corr.shape[3] - mask_width - 1)[0])
        peaks1_z[invalid] = corr.shape[1] // 2 # temp. so no errors would be produced
        peaks1_i[invalid] = corr.shape[2] // 2
        peaks1_j[invalid] = corr.shape[3] // 2

        print(f"Found {len(invalid)} bad peak(s)")
        if len(invalid) == corr.shape[0]: # in case something goes horribly wrong
            return np.zeros((np.size(corr, 0), 3))*np.nan

        #points
        c = corr[ind, peaks1_z, peaks1_i, peaks1_j]
        ci = corr[ind, peaks1_z - 1, peaks1_i, peaks1_j]
        co = corr[ind, peaks1_z + 1, peaks1_i, peaks1_j]
        cl = corr[ind, peaks1_z, peaks1_i - 1, peaks1_j]
        cr = corr[ind, peaks1_z, peaks1_i + 1, peaks1_j]
        cd = corr[ind, peaks1_z, peaks1_i, peaks1_j - 1]
        cu = corr[ind, peaks1_z, peaks1_i, peaks1_j + 1]

        if subpixel_method == "centroid":
            shift_i = ((peaks1_i - 1) * cl + peaks1_i * c + (peaks1_i + 1) * cr) / (cl + c + cr)
            shift_j = ((peaks1_j - 1) * cd + peaks1_j * c + (peaks1_j + 1) * cu) / (cd + c + cu)
            shift_z = ((peaks1_z - 1) * ci + peaks1_z * c + (peaks1_z + 1) * co) / (ci + c + co)

        elif subpixel_method == "gaussian":
            inv = list(np.where(c <= 0)[0]) # get rid of any pesky NaNs
            inv += list(np.where(ci <= 0)[0])
            inv += list(np.where(co <= 0)[0])
            inv += list(np.where(cl <= 0)[0])
            inv += list(np.where(cr <= 0)[0])
            inv += list(np.where(cd <= 0)[0])
            inv += list(np.where(cu <= 0)[0])

            nom1 = log(cl) - log(cr)
            den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
            nom2 = log(cd) - log(cu)
            den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)
            nom3 = log(ci) - log(co)
            den3 = 2 * log(ci) - 4 * log(c) + 2 * log(co)
            shift_i = np.divide(
                nom1, den1,
                out=np.zeros_like(nom1),
                where=(den1 != 0.0)
            )
            shift_j = np.divide(
                nom2, den2,
                out=np.zeros_like(nom2),
                where=(den2 != 0.0)
            )
            shift_z = np.divide(
                nom3, den3,
                out=np.zeros_like(nom3),
                where=(den3 != 0.0)
            )

            if len(inv) >= 1:
                print(f'Found {len(inv)} negative correlation indices resulting in NaNs\n'+
                       'Fallback for negative indices is a 3 point parabolic curve method')
                shift_i[inv] = (cl[inv] - cr[inv]) / (2 * cl[inv] - 4 * c[inv] + 2 * cr[inv])
                shift_j[inv] = (cd[inv] - cu[inv]) / (2 * cd[inv] - 4 * c[inv] + 2 * cu[inv])
                shift_z[inv] = (ci[inv] - co[inv]) / (2 * ci[inv] - 4 * c[inv] + 2 * co[inv])

        elif subpixel_method == "parabolic":
            shift_i = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
            shift_j = (cd - cu) / (2 * cd - 4 * c + 2 * cu)
            shift_z = (ci - co) / (2 * ci - 4 * c + 2 * co)

        if subpixel_method != "centroid":
            disp_vz = (peaks1_z.astype(np.float64) + shift_z) - np.floor(np.array(corr.shape[1])/2)
            disp_vy = (peaks1_i.astype(np.float64) + shift_i) - np.floor(np.array(corr.shape[2])/2)
            disp_vx = (peaks1_j.astype(np.float64) + shift_j) - np.floor(np.array(corr.shape[3])/2)
        else:
            disp_vz = shift_z - np.floor(np.array(corr.shape[1])/2)
            disp_vy = shift_i - np.floor(np.array(corr.shape[2])/2)
            disp_vx = shift_j - np.floor(np.array(corr.shape[3])/2)

        disp_vx[invalid] = peaks_x[invalid]*np.nan
        disp_vy[invalid] = peaks_y[invalid]*np.nan
        disp_vz[invalid] = peaks_z[invalid]*np.nan
        if n_rows == None or n_cols == None or n_layers == None:
            return disp_vx, disp_vy, disp_vz
        else:
            return disp_vx.reshape((n_layers, n_rows, n_cols)),disp_vy.reshape((n_layers, n_rows, n_cols)),disp_vz.reshape((n_layers, n_rows, n_cols))
    else:
        corr = corr.astype(np.float32) + eps # avoids division by zero
        peaks = find_all_first_peaks(corr)[0]
        ind, peaks_y, peaks_x = peaks[:,0], peaks[:,1], peaks[:,2]
        peaks1_i, peaks1_j = peaks_y, peaks_x

        # peak checking
        if subpixel_method in ("gaussian", "centroid", "parabolic"):
            mask_width = 1
        invalid = list(np.where(peaks1_i < mask_width)[0])
        invalid += list(np.where(peaks1_i > corr.shape[1] - mask_width - 1)[0])
        invalid += list(np.where(peaks1_j < mask_width)[0])
        invalid += list(np.where(peaks1_j > corr.shape[2] - mask_width - 1)[0])
        peaks1_i[invalid] = corr.shape[1] // 2
        peaks1_j[invalid] = corr.shape[2] // 2

        print(f"Found {len(invalid)} bad peak(s)")
        if len(invalid) == corr.shape[0]: # in case something goes horribly wrong
            return np.zeros((np.size(corr, 0), 2))*np.nan

        #points
        c = corr[ind, peaks1_i, peaks1_j]
        cl = corr[ind, peaks1_i - 1, peaks1_j]
        cr = corr[ind, peaks1_i + 1, peaks1_j]
        cd = corr[ind, peaks1_i, peaks1_j - 1]
        cu = corr[ind, peaks1_i, peaks1_j + 1]

        if subpixel_method == "centroid":
            shift_i = ((peaks1_i - 1) * cl + peaks1_i * c + (peaks1_i + 1) * cr) / (cl + c + cr)
            shift_j = ((peaks1_j - 1) * cd + peaks1_j * c + (peaks1_j + 1) * cu) / (cd + c + cu)

        elif subpixel_method == "gaussian":
            inv = list(np.where(c <= 0)[0]) # get rid of any pesky NaNs
            inv += list(np.where(cl <= 0)[0])
            inv += list(np.where(cr <= 0)[0])
            inv += list(np.where(cd <= 0)[0])
            inv += list(np.where(cu <= 0)[0])

            nom1 = log(cl) - log(cr)
            den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
            nom2 = log(cd) - log(cu)
            den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)
            shift_i = np.divide(
                nom1, den1,
                out=np.zeros_like(nom1),
                where=(den1 != 0.0)
            )
            shift_j = np.divide(
                nom2, den2,
                out=np.zeros_like(nom2),
                where=(den2 != 0.0)
            )

            if len(inv) >= 1:
                print(f'Found {len(inv)} negative correlation indices resulting in NaNs\n'+
                       'Fallback for negative indices is a 3 point parabolic curve method')
                shift_i[inv] = (cl[inv] - cr[inv]) / (2 * cl[inv] - 4 * c[inv] + 2 * cr[inv])
                shift_j[inv] = (cd[inv] - cu[inv]) / (2 * cd[inv] - 4 * c[inv] + 2 * cu[inv])

        elif subpixel_method == "parabolic":
            shift_i = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
            shift_j = (cd - cu) / (2 * cd - 4 * c + 2 * cu)

        if subpixel_method != "centroid":
            disp_vy = (peaks1_i.astype(np.float64) + shift_i) - np.floor(np.array(corr.shape[1])/2)
            disp_vx = (peaks1_j.astype(np.float64) + shift_j) - np.floor(np.array(corr.shape[2])/2)
        else:
            disp_vy = shift_i - np.floor(np.array(corr.shape[1])/2)
            disp_vx = shift_j - np.floor(np.array(corr.shape[2])/2)

        disp_vx[invalid] = peaks_x[invalid]*np.nan
        disp_vy[invalid] = peaks_y[invalid]*np.nan
        if n_rows == None or n_cols == None:
            return disp_vx, disp_vy
        else:
            return disp_vx.reshape((n_rows, n_cols)),disp_vy.reshape((n_rows, n_cols))

