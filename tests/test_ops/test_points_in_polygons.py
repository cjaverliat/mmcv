# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import points_in_polygons
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MUSA_AVAILABLE, IS_NPU_AVAILABLE


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason="requires CUDA support"
            ),
        ),
        pytest.param(
            "npu",
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason="requires NPU support"
            ),
        ),
        pytest.param(
            "musa",
            marks=pytest.mark.skipif(
                not IS_MUSA_AVAILABLE, reason="requires MUSA support"
            ),
        ),
    ],
)
def test_points_in_polygons(device):
    points = np.array(
        [[300.0, 300.0], [400.0, 400.0], [100.0, 100], [300, 250], [100, 0]]
    )
    polygons = np.array(
        [
            [200.0, 200.0, 400.0, 400.0, 500.0, 200.0, 400.0, 100.0],
            [400.0, 400.0, 500.0, 500.0, 600.0, 300.0, 500.0, 200.0],
            [300.0, 300.0, 600.0, 700.0, 700.0, 700.0, 700.0, 100.0],
        ]
    )
    expected_output = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ).astype(np.float32)
    points = torch.tensor(points, dtype=torch.float32, device=device)
    polygons = torch.tensor(polygons, dtype=torch.float32, device=device)
    assert np.allclose(
        points_in_polygons(points, polygons).cpu().numpy(), expected_output, 1e-3
    )
