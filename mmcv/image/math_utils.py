import torch
import torch.nn.functional as F_t
from math import radians, cos, sin
from typing import Union

def get_translate_matrix(
    offset: Union[int, float], 
    direction: str = "horizontal",
    device: str = "cpu"
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
        translate_matrix = torch.tensor([[1, 0, offset], [0, 1, 0]], dtype=torch.float32, device=device)
    elif direction == "vertical":
        translate_matrix = torch.tensor([[1, 0, 0], [0, 1, offset]], dtype=torch.float32, device=device)
    return translate_matrix

def get_shear_matrix(
    magnitude: Union[int, float],
    direction: str = "horizontal",
    device: str = "cpu",
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
        shear_matrix = torch.tensor([[1, magnitude, 0], [0, 1, 0]], dtype=torch.float32, device=device)
    elif direction == "vertical":
        shear_matrix = torch.tensor([[1, 0, 0], [magnitude, 1, 0]], dtype=torch.float32, device=device)
    return shear_matrix

def get_rotation_matrix_2d(
    center: tuple[float, float],
    angle: float,
    scale: float,
    device: str = "cpu",
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
    
    return torch.tensor([
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)

def convert_affine_matrix_to_theta(
    affine_mtx: torch.Tensor, src_size: tuple[int, int]
) -> torch.Tensor:
    """Convert affine matrix `M` compatible with `cv2.warpAffine` to `theta` matrix
    compatible with `torch.nn.functional.affine_grid`

    Args:
        M (torch.Tensor[2, 3]): Affine matrix
        src_size (tuple[int, int]): Size of the source image (H, W)
    Returns:
        theta (torch.Tensor[2, 3]): Theta matrix for `torch.nn.functional.affine_grid`
    """
    M_aug = torch.concatenate([affine_mtx, torch.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = _get_normalization_matrix(src_size)
    N_inv = _get_normalization_matrix_inv(src_size)
    theta = N @ M_aug @ N_inv
    theta = torch.linalg.inv(theta)
    return theta[:2, :]


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
    src_size: tuple[int, int], device: str = "cpu"
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
    src_size: tuple[int, int], device: str = "cpu"
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
    align_corners: bool = False,
) -> torch.Tensor:
    """Apply affine transformation to the source image.

    Args:
        src (torch.Tensor): The source image in shape (*, C, H, W)
        warp_mtx (torch.Tensor): The affine transformation matrix in shape (2, 3)
        dst_size (tuple[int, int]): The destination size in (H, W)

    Returns:
        torch.Tensor: The transformed image in shape (*, C, H, W)
    """
    assert len(src.shape) == 3 or len(src.shape) == 4
    assert len(affine_mtx.shape) == 2 and affine_mtx.shape == (2, 3)
    assert len(dst_size) == 2

    has_batch_dim = len(src.shape) == 4

    if not has_batch_dim:
        src = src.unsqueeze(0)

    n, c, h, w = src.shape
    dst_h, dst_w = dst_size

    theta = convert_affine_matrix_to_theta(affine_mtx, (h, w))

    grid = torch.nn.functional.affine_grid(
        theta=theta.expand(n, -1, -1),
        size=(n, c, dst_h, dst_w),
        align_corners=align_corners,
    )

    is_byte_image = src.dtype == torch.uint8

    if is_byte_image:
        src = src.float()

    dst = F_t.grid_sample(
        input=src,
        grid=grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=False,
    )

    if is_byte_image:
        dst = dst.to(torch.uint8)

    if not has_batch_dim:
        dst = dst.squeeze(0)

    return dst
