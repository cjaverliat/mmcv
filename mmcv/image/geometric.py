# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from typing import List, Optional, Tuple, Union, no_type_check, Literal

import torch
import torchvision.transforms.v2.functional as F_tv
import cv2
import numpy as np
from mmengine.utils import to_2tuple
import warnings

from ..utils.math import (
    get_rotation_matrix_2d,
    warp_affine,
    get_shear_matrix,
    get_translate_matrix,
)
from .io import imread_backend

try:
    from PIL import Image
except ImportError:
    Image = None


def get_image_chw(img: np.ndarray | torch.Tensor) -> Tuple[int, int, int]:
    """Get the channel, height, and width of an image.

    Args:
        img (ndarray | torch.Tensor): The input image.

    """
    if img.ndim == 2:
        return 1, img.shape[0], img.shape[1]
    else:
        return img.shape[2], img.shape[0], img.shape[1]


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

cv2_border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "transparent": cv2.BORDER_TRANSPARENT,
    "isolated": cv2.BORDER_ISOLATED,
}

# Pillow >=v9.1.0 use a slightly different naming scheme for filters.
# Set pillow_interp_codes according to the naming scheme used.
if Image is not None:
    if hasattr(Image, "Resampling"):
        pillow_interp_codes = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
        }
    else:
        pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }


def imresize(
    img: np.ndarray | torch.Tensor,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray | torch.Tensor] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray | torch.Tensor): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray | torch.Tensor): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    _, img_h, img_w = get_image_chw(img)

    if isinstance(img, torch.Tensor):
        # At this point, img is stored in HWC format
        # So we need to change it to work with torchvision layout (CHW)

        if interpolation == "lanczos" or interpolation == "area":
            warnings.warn(
                f"{interpolation} interpolation is not supported for torch.Tensor, using nearest instead"
            )
            interpolation = "nearest"

        resized_img = F_tv.resize_image(
            img.permute(2, 0, 1),  # HWC -> CHW
            size[::-1],  # (w, h) -> (h, w)
            interpolation=F_tv.InterpolationMode(interpolation),
            antialias=True,
        ).permute(1, 2, 0)  # CHW -> HWC

        if out is not None:
            if isinstance(out, torch.Tensor):
                out.copy_(resized_img)
            elif isinstance(out, np.ndarray):
                out[:] = resized_img.numpy()
            resized_img = out

        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / img_w
            h_scale = size[1] / img_h
            return resized_img, w_scale, h_scale

    if backend is None:
        backend = imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation]
        )
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / img_w
        h_scale = size[1] / img_h
        return resized_img, w_scale, h_scale


@no_type_check
def imresize_to_multiple(
    img: np.ndarray | torch.Tensor,
    divisor: Union[int, Tuple[int, int]],
    size: Union[int, Tuple[int, int], None] = None,
    scale_factor: Union[float, int, Tuple[float, float], Tuple[int, int], None] = None,
    keep_ratio: bool = False,
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray | torch.Tensor] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image according to a given size or scale factor and then rounds
    up the the resized or rescaled image size to the nearest value that can be
    divided by the divisor.

    Args:
        img (ndarray | torch.Tensor): The input image.
        divisor (int | tuple): Resized image size will be a multiple of
            divisor. If divisor is a tuple, divisor should be
            (w_divisor, h_divisor).
        size (None | int | tuple[int]): Target size (w, h). Default: None.
        scale_factor (None | float | int | tuple[float] | tuple[int]):
            Multiplier for spatial size. Should match input size if it is a
            tuple and the 2D style is (w_scale_factor, h_scale_factor).
            Default: None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: False.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray | torch.Tensor): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    _, img_h, img_w = get_image_chw(img)

    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is None and scale_factor is None:
        raise ValueError("one of size or scale_factor should be defined")
    elif size is not None:
        size = to_2tuple(size)
        if keep_ratio:
            size = rescale_size((img_w, img_h), size, return_scale=False)
    else:
        size = _scale_size((img_w, img_h), scale_factor)

    divisor = to_2tuple(divisor)
    size = tuple(int(np.ceil(s / d)) * d for s, d in zip(size, divisor))
    resized_img, w_scale, h_scale = imresize(
        img,
        size,
        return_scale=True,
        interpolation=interpolation,
        out=out,
        backend=backend,
    )
    if return_scale:
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def imresize_like(
    img: np.ndarray | torch.Tensor,
    dst_img: np.ndarray | torch.Tensor,
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray | torch.Tensor, float, float], np.ndarray | torch.Tensor]:
    """Resize image to the same size of a given image.

    Args:
        img (ndarray | torch.Tensor): The input image.
        dst_img (ndarray | torch.Tensor): The target image.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    _, dst_img_h, dst_img_w = get_image_chw(dst_img)
    return imresize(
        img, (dst_img_w, dst_img_h), return_scale, interpolation, backend=backend
    )


def rescale_size(
    old_size: tuple,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(
            f"Scale must be a number or tuple of int, but got {type(scale)}"
        )

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: np.ndarray | torch.Tensor,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[np.ndarray | torch.Tensor, Tuple[np.ndarray | torch.Tensor, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray | torch.Tensor): The input image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray | torch.Tensor: The rescaled image.
    """
    _, img_h, img_w = get_image_chw(img)
    new_size, scale_factor = rescale_size((img_w, img_h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imflip(
    img: np.ndarray | torch.Tensor, direction: str = "horizontal"
) -> np.ndarray | torch.Tensor:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray | torch.Tensor): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray | torch.Tensor: The flipped image.
    """

    assert direction in ["horizontal", "vertical", "diagonal"]

    if direction == "horizontal":
        if isinstance(img, torch.Tensor):
            return torch.flip(img, dims=(1,))
        else:
            return np.flip(img, axis=1)
    elif direction == "vertical":
        if isinstance(img, torch.Tensor):
            return torch.flip(img, dims=(0,))
        else:
            return np.flip(img, axis=0)
    else:
        if isinstance(img, torch.Tensor):
            return torch.flip(img, dims=(0, 1))
        else:
            return np.flip(img, axis=(0, 1))


def imflip_(
    img: np.ndarray | torch.Tensor, direction: str = "horizontal"
) -> np.ndarray | torch.Tensor:
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray | torch.Tensor): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray | torch.Tensor: The flipped image (inplace).
    """
    assert direction in ["horizontal", "vertical", "diagonal"]
    if direction == "horizontal":
        if isinstance(img, torch.Tensor):
            return torch.flip(img, dims=(2,), inplace=True)
        else:
            return cv2.flip(img, 1, img)
    elif direction == "vertical":
        if isinstance(img, torch.Tensor):
            return torch.flip(img, dims=(1,), inplace=True)
        else:
            return cv2.flip(img, 0, img)
    else:
        if isinstance(img, torch.Tensor):
            return torch.flip(img, dims=(0, 1), inplace=True)
        else:
            return cv2.flip(img, -1, img)

def _border_value_to_tensor(
    n_channels: int,
    border_value: Optional[
        Union[int, float, tuple[int, int, int], np.ndarray, torch.Tensor]
    ],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if border_value is None:
        return None
    
    if isinstance(border_value, (int, float)):
        return torch.full((n_channels,), fill_value=border_value, dtype=dtype, device=device)
    elif isinstance(border_value, (list, tuple)):
        if len(border_value) == 1:
            return torch.full((n_channels,), fill_value=border_value[0], dtype=dtype, device=device)
        elif len(border_value) == n_channels:
            return torch.tensor(border_value, dtype=dtype, device=device)
        else:
            raise ValueError(f"Expected 1 or {n_channels} elements in `border_value`, but got {len(border_value)}")
    elif isinstance(border_value, np.ndarray):
        if border_value.shape == (n_channels,):
            return torch.tensor(border_value, dtype=dtype, device=device)
        elif border_value.shape == (1,):
            return torch.full((n_channels,), fill_value=border_value[0], dtype=dtype, device=device)
        else:
            raise ValueError(f"Expected 1 or {n_channels} elements in `border_value`, but got {border_value.shape}")
    elif isinstance(border_value, torch.Tensor):
        if border_value.shape == (n_channels,):
            return border_value
        elif border_value.shape == (1,):
            return torch.full((n_channels,), fill_value=border_value[0], dtype=dtype, device=device)
        else:
            raise ValueError(f"Expected 1 or {n_channels} elements in `border_value`, but got {border_value.shape}")
    else:
        raise ValueError(f"Invalid type {type(border_value)} for `border_value`")

def imrotate(
    img: np.ndarray | torch.Tensor,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    border_value: Optional[Union[int, float, tuple[int, int, int], np.ndarray, torch.Tensor]] = None,
    interpolation: str = "bilinear",
    auto_bound: bool = False,
    border_mode: Literal["constant", "replicate", "reflect"] = "constant",
) -> np.ndarray | torch.Tensor:
    """Rotate an image.

    Args:
        img (np.ndarray | torch.Tensor): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int | float | tuple[int, int, int] | np.ndarray | torch.Tensor): Border value used in case of a constant border.
            Defaults to 0.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
        border_mode (str): Pixel extrapolation method. Defaults to 'constant'.

    Returns:
        np.ndarray | torch.Tensor: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError("`auto_bound` conflicts with `center`")
    img_c, img_h, img_w = get_image_chw(img)
    if center is None:
        center = ((img_w - 1) * 0.5, (img_h - 1) * 0.5)
    assert isinstance(center, tuple)

    if isinstance(img, torch.Tensor):

        border_value = _border_value_to_tensor(img_c, border_value, img.dtype, img.device)

        if border_mode == "replicate":
            border_mode = "border"
        elif border_mode == "constant":
            if border_value is None:
                border_mode = "zeros"
            else:
                border_mode = "fill"
                border_value = border_value.to(img.device, img.dtype)

        matrix = get_rotation_matrix_2d(center, -angle, scale, device=img.device)
        if auto_bound:
            cos = torch.abs(matrix[0, 0])
            sin = torch.abs(matrix[0, 1])
            new_w = img_h * sin + img_w * cos
            new_h = img_h * cos + img_w * sin
            matrix[0, 2] += (new_w - img_w) * 0.5
            matrix[1, 2] += (new_h - img_h) * 0.5
            img_w = int(torch.round(new_w))
            img_h = int(torch.round(new_h))

        has_channel_dim = img.ndim == 3
        if not has_channel_dim:
            img = img.unsqueeze(-1)

        rotated = warp_affine(
            img.permute(2, 0, 1),  # HWC -> CHW
            matrix,
            (img_h, img_w),
            interpolation=interpolation,
            padding_mode=border_mode,
            fill_value=border_value,
        ).permute(1, 2, 0)  # CHW -> HWC

        if not has_channel_dim:
            rotated = rotated.squeeze(-1)
    else:
        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = img_h * sin + img_w * cos
            new_h = img_h * cos + img_w * sin
            matrix[0, 2] += (new_w - img_w) * 0.5
            matrix[1, 2] += (new_h - img_h) * 0.5
            img_w = int(np.round(new_w))
            img_h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix,
            (img_w, img_h),
            flags=cv2_interp_codes[interpolation],
            borderMode=cv2_border_modes[border_mode],
            borderValue=border_value,
        )
    return rotated


def bbox_clip(bboxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple[int]): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
    cmin[0::2] = img_shape[1] - 1
    cmin[1::2] = img_shape[0] - 1
    clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
    return clipped_bboxes


def bbox_scaling(
    bboxes: np.ndarray,
    scale: float,
    clip_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple[int], optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def imcrop(
    img: np.ndarray | torch.Tensor,
    bboxes: np.ndarray,
    scale: float = 1.0,
    pad_fill: Union[float, list, None] = None,
) -> Union[np.ndarray | torch.Tensor, List[np.ndarray | torch.Tensor]]:
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray | torch.Tensor): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no scaling.
        pad_fill (Number | list[Number]): Value to be filled for padding.
            Default: None, which means no padding.

    Returns:
        list[ndarray | torch.Tensor] | ndarray | torch.Tensor: The cropped image patches.
    """
    img_c, _, _ = get_image_chw(img)

    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(img_c)]
        assert len(pad_fill) == img_c

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1 : y2 + 1, x1 : x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            patch_h = _y2 - _y1 + 1
            patch_w = _x2 - _x1 + 1
            if img_c == 1:
                patch_shape = (patch_h, patch_w)
            else:
                patch_shape = (patch_h, patch_w, img_c)  # type: ignore

            if isinstance(img, torch.Tensor):
                patch = torch.tensor(
                    pad_fill, dtype=img.dtype, device=img.device
                ) * torch.ones(patch_shape, dtype=img.dtype, device=img.device)
            else:
                patch = np.array(pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype
                )

            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start : y_start + h, x_start : x_start + w, ...] = img[
                y1 : y1 + h, x1 : x1 + w, ...
            ]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def impad(
    img: np.ndarray | torch.Tensor,
    *,
    shape: Optional[Tuple[int, int]] = None,
    padding: Union[int, tuple, None] = None,
    pad_val: Union[float, List] = 0,
    padding_mode: str = "constant",
) -> np.ndarray | torch.Tensor:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray | torch.Tensor): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray | torch.Tensor: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)

    img_c, img_h, img_w = get_image_chw(img)

    if shape is not None:
        width = max(shape[1] - img_w, 0)
        height = max(shape[0] - img_h, 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img_c
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError(
            f"pad_val must be a int or a tuple. But received {type(pad_val)}"
        )

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(
            f"Padding must be a int or a 2, or 4 element tuple.But received {padding}"
        )

    # check padding mode
    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

    padding_top = padding[1]
    padding_bottom = padding[3]
    padding_left = padding[0]
    padding_right = padding[2]

    if isinstance(img, torch.Tensor):
        has_channel_dim = img.ndim == 3

        if not has_channel_dim:
            img = img.unsqueeze(-1)

        if padding_mode != "constant" and isinstance(pad_val, (list, tuple)):
            # Padding mode edge/reflect/symmetric is not supported if fill is not scalar, using first element of fill as padding value
            pad_val = pad_val[0]

        padded_img = F_tv.pad_image(
            img.permute(2, 0, 1),
            padding=(padding_left, padding_top, padding_right, padding_bottom),
            fill=pad_val,
            padding_mode=padding_mode,
        ).permute(1, 2, 0)

        if not has_channel_dim:
            padded_img = padded_img.squeeze(-1)

        return padded_img

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    img = cv2.copyMakeBorder(
        img,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        border_type[padding_mode],
        value=pad_val,
    )

    return img


def impad_to_multiple(
    img: np.ndarray | torch.Tensor, divisor: int, pad_val: Union[float, List] = 0
) -> np.ndarray | torch.Tensor:
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray | torch.Tensor): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray | torch.Tensor: The padded image.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


def cutout(
    img: np.ndarray | torch.Tensor,
    shape: Union[int, Tuple[int, int]],
    pad_val: Union[int, float, tuple] = 0,
) -> np.ndarray | torch.Tensor:
    """Randomly cut out a rectangle from the original img.

    Args:
        img (ndarray | torch.Tensor): Image to be cutout.
        shape (int | tuple[int]): Expected cutout shape (h, w). If given as a
            int, the value will be used for both h and w.
        pad_val (int | float | tuple[int | float]): Values to be filled in the
            cut area. Defaults to 0.

    Returns:
        ndarray | torch.Tensor: The cutout image.
    """

    img_c, _, _ = get_image_chw(img)

    if isinstance(shape, int):
        cut_h, cut_w = shape, shape
    else:
        assert isinstance(shape, tuple) and len(shape) == 2, (
            f"shape must be a int or a tuple with length 2, but got type "
            f"{type(shape)} instead."
        )
        cut_h, cut_w = shape
    if isinstance(pad_val, (int, float)):
        pad_val = tuple([pad_val] * img_c)
    elif isinstance(pad_val, tuple):
        assert len(pad_val) == img_c, (
            "Expected the num of elements in tuple equals the channels"
            "of input image. Found {} vs {}".format(len(pad_val), img_c)
        )
    else:
        raise TypeError(f"Invalid type {type(pad_val)} for `pad_val`")

    img_h, img_w = img.shape[:2]
    y0 = np.random.uniform(img_h)
    x0 = np.random.uniform(img_w)

    y1 = int(max(0, y0 - cut_h / 2.0))
    x1 = int(max(0, x0 - cut_w / 2.0))
    y2 = min(img_h, y1 + cut_h)
    x2 = min(img_w, x1 + cut_w)

    if img.ndim == 2:
        patch_shape = (y2 - y1, x2 - x1)
    else:
        patch_shape = (y2 - y1, x2 - x1, img_c)  # type: ignore

    if isinstance(img, torch.Tensor):
        img_cutout = img.clone()
        patch = torch.tensor(pad_val, dtype=img.dtype, device=img.device) * torch.ones(
            patch_shape, dtype=img.dtype, device=img.device
        )
        img_cutout[y1:y2, x1:x2, ...] = patch
    else:
        img_cutout = img.copy()
        patch = np.array(pad_val, dtype=img.dtype) * np.ones(
            patch_shape, dtype=img.dtype
        )
        img_cutout[y1:y2, x1:x2, ...] = patch

    return img_cutout


def _get_shear_matrix(
    magnitude: Union[int, float], direction: str = "horizontal"
) -> np.ndarray:
    """Generate the shear matrix for transformation.

    Args:
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".

    Returns:
        ndarray: The shear matrix with dtype float32.
    """
    if direction == "horizontal":
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif direction == "vertical":
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix


def imshear(
    img: np.ndarray | torch.Tensor,
    magnitude: Union[int, float],
    direction: str = "horizontal",
    border_value: Optional[Union[int, float, tuple[int, int, int], np.ndarray, torch.Tensor]] = None,
    interpolation: str = "bilinear",
) -> np.ndarray | torch.Tensor:
    """Shear an image.

    Args:
        img (ndarray | torch.Tensor): Image to be sheared with format (h, w)
            or (h, w, c).
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray | torch.Tensor: The sheared image.
    """
    assert direction in ["horizontal", "vertical"], f"Invalid direction: {direction}"
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]

    if isinstance(img, torch.Tensor):

        border_value = _border_value_to_tensor(channels, border_value, img.dtype, img.device)

        if border_value is None:
            padding_mode = "zeros"
        else:
            padding_mode = "fill"
            border_value = border_value.to(img.device, img.dtype)

        shear_matrix = get_shear_matrix(magnitude, direction, device=img.device)
        has_channel_dim = img.ndim == 3
        if not has_channel_dim:
            img = img.unsqueeze(-1)

        sheared = warp_affine(
            img.permute(2, 0, 1),
            shear_matrix,
            (height, width),
            interpolation=interpolation,
            padding_mode=padding_mode,
            fill_value=border_value,
        ).permute(1, 2, 0)

        if not has_channel_dim:
            sheared = sheared.squeeze(-1)

        return sheared
    else:
        if border_value is None:
            border_value = tuple([0] * channels)
        elif isinstance(border_value, int):
            border_value = tuple([border_value] * channels)  # type: ignore
        elif isinstance(border_value, tuple):
            assert len(border_value) == channels, \
                'Expected the num of elements in tuple equals the channels' \
                'of input image. Found {} vs {}'.format(
                    len(border_value), channels)
        else:
            raise ValueError(
                f'Invalid type {type(border_value)} for `border_value`')

        shear_matrix = _get_shear_matrix(magnitude, direction)
        sheared = cv2.warpAffine(
            img,
            shear_matrix,
            (width, height),
            # Note case when the number elements in `border_value`
            # greater than 3 (e.g. shearing masks whose channels large
            # than 3) will raise TypeError in `cv2.warpAffine`.
            # Here simply slice the first 3 values in `border_value`.
            borderValue=border_value[:3],  # type: ignore
            flags=cv2_interp_codes[interpolation],
        )
    return sheared


def _get_translate_matrix(
    offset: Union[int, float], direction: str = "horizontal"
) -> np.ndarray:
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".

    Returns:
        ndarray: The translate matrix with dtype float32.
    """
    if direction == "horizontal":
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == "vertical":
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def imtranslate(
    img: np.ndarray | torch.Tensor,
    offset: Union[int, float],
    direction: str = "horizontal",
    border_value: Optional[Union[int, float, tuple[int, int, int], np.ndarray, torch.Tensor]] = None,
    interpolation: str = "bilinear",
) -> np.ndarray | torch.Tensor:
    """Translate an image.

    Args:
        img (ndarray | torch.Tensor): Image to be translated with format
            (h, w) or (h, w, c).
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray | torch.Tensor: The translated image.
    """
    assert direction in ["horizontal", "vertical"], f"Invalid direction: {direction}"
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]

    if isinstance(img, torch.Tensor):
        border_value = _border_value_to_tensor(channels, border_value, img.dtype, img.device)

        if border_value is None:
            padding_mode = "zeros"
        else:
            padding_mode = "fill"
            border_value = border_value.to(img.device, img.dtype)

        translate_matrix = get_translate_matrix(offset, direction, device=img.device)

        has_channel_dim = img.ndim == 3

        if not has_channel_dim:
            img = img.unsqueeze(-1)

        translated = warp_affine(
            img.permute(2, 0, 1),
            translate_matrix,
            (height, width),
            interpolation=interpolation,
            padding_mode=padding_mode,
            fill_value=border_value,
        ).permute(1, 2, 0)

        if not has_channel_dim:
            translated = translated.squeeze(-1)
    else:
        if border_value is None:
            border_value = tuple([0] * channels)
        elif isinstance(border_value, int):
            border_value = tuple([border_value] * channels)
        elif isinstance(border_value, tuple):
            assert len(border_value) == channels, (
                "Expected the num of elements in tuple equals the channels"
                "of input image. Found {} vs {}".format(len(border_value), channels)
            )
        else:
            raise ValueError(f"Invalid type {type(border_value)} for `border_value`.")

        translate_matrix = _get_translate_matrix(offset, direction)
        translated = cv2.warpAffine(
            img,
            translate_matrix,
            (width, height),
            # Note case when the number elements in `border_value`
            # greater than 3 (e.g. translating masks whose channels
            # large than 3) will raise TypeError in `cv2.warpAffine`.
            # Here simply slice the first 3 values in `border_value`.
            borderValue=border_value[:3],
            flags=cv2_interp_codes[interpolation],
        )
    return translated
