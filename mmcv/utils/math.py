import torch
import torch.nn.functional as F_t
from math import radians, cos, sin
from typing import Union, Optional


def _torch_inverse_cast(input: torch.Tensor) -> torch.Tensor:
    """Make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.linalg.inv(input.to(dtype)).to(input.dtype)


def get_translate_matrix(
    offset: Union[int, float], direction: str = "horizontal", device: str = "cpu"
) -> torch.Tensor:
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".
        device (str): The device to place the resulting tensor.

    Returns:
        torch.Tensor: The translate matrix with dtype float32.
    """
    if direction == "horizontal":
        translate_matrix = torch.tensor(
            [[1, 0, offset], [0, 1, 0]], dtype=torch.float32, device=device
        )
    elif direction == "vertical":
        translate_matrix = torch.tensor(
            [[1, 0, 0], [0, 1, offset]], dtype=torch.float32, device=device
        )
    return translate_matrix


def get_shear_matrix(
    magnitude: Union[int, float],
    direction: str = "horizontal",
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Generate the shear matrix for transformation.

    Args:
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".

    Returns:
        torch.Tensor: The shear matrix with dtype float32.
    """
    if direction == "horizontal":
        shear_matrix = torch.tensor(
            [[1, magnitude, 0], [0, 1, 0]], dtype=torch.float32, device=device
        )
    elif direction == "vertical":
        shear_matrix = torch.tensor(
            [[1, 0, 0], [magnitude, 1, 0]], dtype=torch.float32, device=device
        )
    return shear_matrix


def get_rotation_matrix_2d(
    center: tuple[float, float],
    angle: float,
    scale: float,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Get the rotation matrix for 2D rotation.

    Args:
        center: The center of the rotation in (x, y) format
        angle: The angle of the rotation in degrees
        scale: The scale of the rotation
    Returns:
        The rotation matrix in (2, 3) format
    """
    angle = radians(angle)
    alpha = cos(angle) * scale
    beta = sin(angle) * scale

    return torch.tensor(
        [
            [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
            [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def normal_transform_pixel(
    height: int, width: int, eps: float = 1e-14, device: torch.device = "cpu"
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height: image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors
        device: device to place the result on.
        dtype: dtype of the result.

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.

    """
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    # Prevent division by zero
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0
    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    return tr_mat.unsqueeze(0)


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: tuple[int, int],
    dsize_dst: tuple[int, int],
) -> torch.Tensor:
    """Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix (torch.Tensor): Homography matrix in shape (*, 2, 3) or (*, 3, 3)
        dsize_src (tuple[int, int]): Size of the source image (H, W)
        dsize_dst (tuple[int, int]): Size of the destination image (H, W)
    Returns:
        theta (torch.Tensor): The normalized homography matrix in shape (*, 2, 3)
    """
    has_batch_dim = dst_pix_trans_src_pix.ndim > 2

    if not has_batch_dim:
        dst_pix_trans_src_pix = dst_pix_trans_src_pix.unsqueeze(0)

    if dst_pix_trans_src_pix.shape[-2:] == (2, 3):
        dst_pix_trans_src_pix = torch.concatenate(
            [
                dst_pix_trans_src_pix,
                torch.zeros((*dst_pix_trans_src_pix.shape[:-2], 1, 3), device=dst_pix_trans_src_pix.device),
            ],
            dim=-2,
        )
        dst_pix_trans_src_pix[..., -1, -1] = 1.0

    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(
        src_h, src_w, device=dst_pix_trans_src_pix.device
    )

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(
        dst_h, dst_w, device=dst_pix_trans_src_pix.device
    )

    dst_norm_trans_src_norm = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def get_affine_matrix(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    """
    Compute the affine transformation matrix that maps src_points to dst_points.

    Args:
        src_pts: Tensor of shape (n, 2) containing source points
        dst_pts: Tensor of shape (n, 2) containing destination points
        src_width: Width of the source image
        src_height: Height of the source image

    Returns:
        transformation_matrix: Tensor of shape (2, 3) representing the affine transform
    """
    # We need at least 3 points to determine an affine transformation
    assert src_pts.shape[0] >= 3, "At least 3 points are required"
    assert src_pts.shape == dst_pts.shape, (
        "Source and destination points must have the same shape"
    )

    n = src_pts.shape[0]
    A = torch.zeros((2 * n, 6), dtype=src_pts.dtype, device=src_pts.device)
    b = torch.zeros((2 * n), dtype=src_pts.dtype, device=src_pts.device)

    for i in range(n):
        A[2 * i, 0] = src_pts[i, 0]
        A[2 * i, 1] = src_pts[i, 1]
        A[2 * i, 2] = 1.0
        b[2 * i] = dst_pts[i, 0]
        A[2 * i + 1, 3] = src_pts[i, 0]
        A[2 * i + 1, 4] = src_pts[i, 1]
        A[2 * i + 1, 5] = 1.0
        b[2 * i + 1] = dst_pts[i, 1]

    x = torch.linalg.lstsq(A, b).solution
    transformation_matrix = x.reshape(2, 3)
    return transformation_matrix


def _get_normalization_matrix(
    src_size: tuple[int, int], device: torch.device = "cpu"
) -> torch.Tensor:
    """
    Get the normalization matrix N that maps from normalized to unnormalized coordinates
    Args:
        w (int): Width of the image
        h (int): Height of the image
        device (str): Device to create the tensor on
    Returns:
        torch.Tensor: Normalization matrix N
    """
    h, w = src_size
    N = torch.zeros((3, 3), dtype=torch.float32, device=device)
    N[0, 0] = 2.0 / w
    N[0, 1] = 0
    N[1, 1] = 2.0 / h
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def _get_normalization_matrix_inv(
    src_size: tuple[int, int], device: torch.device = "cpu"
) -> torch.Tensor:
    """
    Get the inverse normalization matrix N_inv that maps from unnormalized to normalized coordinates
    Args:
        w (int): Width of the image
        h (int): Height of the image
        device (str): Device to create the tensor on
    Returns:
        torch.Tensor: Inverse normalization matrix N_inv
    """
    h, w = src_size
    N_inv = torch.zeros((3, 3), dtype=torch.float32, device=device)
    N_inv[0, 0] = w / 2.0
    N_inv[0, 1] = 0
    N_inv[1, 1] = h / 2.0
    N_inv[1, 0] = 0
    N_inv[0, -1] = w / 2.0
    N_inv[1, -1] = h / 2.0
    N_inv[-1, -1] = 1.0
    return N_inv


def affine_transform(
    pts: torch.Tensor,
    affine_mtx: torch.Tensor,
) -> torch.Tensor:
    """Apply affine transformation to the points.

    Args:
        pts (torch.Tensor): The points in shape (N, 2)
        affine_mtx (torch.Tensor): The affine transformation matrix in shape (2, 3)

    Returns:
        torch.Tensor: The transformed points in shape (N, 2)
    """
    assert pts.shape[-1] == 2, f"Expected points in shape (*, 2), but got {pts.shape}"
    assert affine_mtx.shape == (
        2,
        3,
    ), f"Expected affine matrix in shape (2, 3), but got {affine_mtx.shape}"

    batch_dims = pts.shape[:-1]
    pts_aug = torch.cat([pts, torch.ones(batch_dims + (1,), device=pts.device)], dim=-1)
    transformed_pts = torch.einsum("ij,...nj->...ni", affine_mtx, pts_aug)
    return transformed_pts[..., :2]


def warp_affine(
    src: torch.Tensor,
    affine_mtx: torch.Tensor,
    dst_size: tuple[int, int],  # (H, W)
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    fill_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply affine transformation to the source image.

    Args:
        src (torch.Tensor): The source image in shape (*, C, H, W)
        warp_mtx (torch.Tensor): The affine transformation matrix in shape (2, 3) or (3, 3)
        dst_size (tuple[int, int]): The destination size in (H, W)
        interpolation (str): interpolation mode to calculate output values ("bilinear" | "nearest"). Default: "bilinear"
        padding_mode (str): padding mode for outside grid values ("zeros" | "border" | "reflection" | "fill"). Default: "zeros"
        align_corners (bool): if True, the corner pixels are sampled at the corner locations. Default: True
        fill_value (torch.Tensor): The fill value for the outside grid values of shape (3,). Default: None
    Returns:
        torch.Tensor: The transformed image in shape (*, C, H, W)
    """
    assert len(src.shape) == 3 or len(src.shape) == 4
    assert len(affine_mtx.shape) == 2 and affine_mtx.shape in [(2, 3), (3, 3)]
    assert len(dst_size) == 2

    has_batch_dim = len(src.shape) == 4

    if not has_batch_dim:
        src = src.unsqueeze(0)

    n, c, h, w = src.shape
    dst_h, dst_w = dst_size

    dst_norm_trans_src_norm = normalize_homography(affine_mtx, (h, w), (dst_h, dst_w))
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)

    grid = F_t.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        (n, c, dst_h, dst_w),
        align_corners=align_corners,
    ).to(src.device)

    is_byte_image = src.dtype == torch.uint8

    if is_byte_image:
        src = src.float()

    if padding_mode == "fill":
        if fill_value is None:
            fill_value = torch.zeros((c,), device=src.device, dtype=src.dtype)
        dst = _fill_and_warp(
            src=src,
            grid=grid,
            mode=interpolation,
            align_corners=align_corners,
            fill_value=fill_value,
        )
    else:
        dst = F_t.grid_sample(
            input=src,
            grid=grid,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    if is_byte_image:
        dst = dst.byte()

    if not has_batch_dim:
        dst = dst.squeeze(0)

    return dst


def _fill_and_warp(
    src: torch.Tensor,
    grid: torch.Tensor,
    mode: str,
    align_corners: bool,
    fill_value: torch.Tensor,
) -> torch.Tensor:
    r"""Warp a mask of ones, then multiple with fill_value and add to default warp.

    Args:
        src: input tensor of shape :math:`(B, 3, H, W)`.
        grid: grid tensor from `transform_points`.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        align_corners: interpolation flag.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.

    Returns:
        the warped and filled tensor with shape :math:`(B, 3, H, W)`.

    """
    ones_mask = torch.ones_like(src, device=src.device, dtype=src.dtype)
    fill_value = fill_value.to(ones_mask)[
        None, :, None, None
    ]  # cast and add dimensions for broadcasting
    inv_ones_mask = 1 - F_t.grid_sample(
        ones_mask, grid, align_corners=align_corners, mode=mode, padding_mode="zeros"
    )
    inv_color_mask = inv_ones_mask * fill_value
    return (
        F_t.grid_sample(
            src, grid, align_corners=align_corners, mode=mode, padding_mode="zeros"
        )
        + inv_color_mask
    )
