from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize
from nnunetv2.configuration import ANISO_THRESHOLD

_RESAMPLE_MAX_WORKERS = int(os.environ.get('RESAMPLE_WORKERS', max((os.cpu_count() or 2) // 2, 2)))


def _calc_workers(new_shape, num_channels):
    cpu_limit = min(_RESAMPLE_MAX_WORKERS, num_channels)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1]) * 1024
                    vol_bytes = int(np.prod(new_shape)) * 8
                    # Base: output array + coord_map + OS headroom
                    base = vol_bytes * (num_channels + 3) + 4 * 1024**3
                    per_worker = vol_bytes * 2
                    usable = total - base
                    if usable <= 0:
                        return 1
                    mem_limit = max(int(usable / per_worker), 1)
                    return min(cpu_limit, mem_limit)
    except Exception:
        pass
    return min(cpu_limit, 4)


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def determine_do_sep_z_and_axis(
        force_separate_z: bool,
        current_spacing,
        new_spacing,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD) -> Tuple[bool, Union[int, None]]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_data_or_seg_to_spacing(data: np.ndarray,
                                    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    is_seg: bool = False,
                                    order: int = 3, order_z: int = 0,
                                    force_separate_z: Union[bool, None] = False,
                                    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data.shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    """
    needed for segmentation export. Stupid, I know
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg(data: np.ndarray, new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, axis: Union[None, int] = None, order: int = 3,
                         do_separate_z: bool = False, order_z: int = 0, dtype_out = None):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        if do_separate_z:
            assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            # Pre-compute coord_map once (identical for all channels)
            _need_z_resample = bool(shape[axis] != new_shape[axis])
            _coord_map = None
            if _need_z_resample:
                rows, cols, dim = int(new_shape[0]), int(new_shape[1]), int(new_shape[2])
                _tmp_shape = deepcopy(new_shape)
                _tmp_shape[axis] = shape[axis]
                orig_rows, orig_cols, orig_dim = int(_tmp_shape[0]), int(_tmp_shape[1]), int(_tmp_shape[2])
                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim
                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5
                _coord_map = np.array([map_rows, map_cols, map_dims])
                del map_rows, map_cols, map_dims

            def _resample_channel_sep_z(c):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if _need_z_resample:
                    if not is_seg or order_z == 0:
                        return map_coordinates(reshaped_here, _coord_map, order=order_z, mode='nearest')[None]
                    else:
                        result = np.zeros(new_shape, dtype=dtype_out)
                        unique_labels = np.sort(pd.unique(reshaped_here.ravel()))
                        for i, cl in enumerate(unique_labels):
                            result[np.round(
                                map_coordinates((reshaped_here == cl).astype(float), _coord_map, order=order_z,
                                                mode='nearest')) > 0.5] = cl
                        return result
                else:
                    return reshaped_here

            _n_workers = _calc_workers(new_shape, data.shape[0])
            with ThreadPoolExecutor(max_workers=_n_workers) as executor:
                for c, result in zip(range(data.shape[0]), executor.map(_resample_channel_sep_z, range(data.shape[0]))):
                    reshaped_final[c] = result
        else:
            def _resample_channel(c):
                return resize_fn(data[c], new_shape, order, **kwargs)
            _n_workers = _calc_workers(new_shape, data.shape[0])
            with ThreadPoolExecutor(max_workers=_n_workers) as executor:
                for c, result in zip(range(data.shape[0]), executor.map(_resample_channel, range(data.shape[0]))):
                    reshaped_final[c] = result
        return reshaped_final
    else:
        # print("no resampling necessary")
        return data
    

if __name__ == '__main__':
    input_array = np.random.random((1, 42, 231, 142))
    output_shape = (52, 256, 256)
    out = resample_data_or_seg(input_array, output_shape, is_seg=False, axis=3, order=1, order_z=0, do_separate_z=True)
    print(out.shape, input_array.shape)
