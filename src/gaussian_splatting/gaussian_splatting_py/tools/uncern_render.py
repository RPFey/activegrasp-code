import math
from typing_extensions import Literal
from typing import Optional, Tuple, Dict

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from gsplat.rendering import get_projection_matrix
import numpy as np

from diff_eig import GaussianRasterizationSettings, GaussianRasterizer

def rasterization_fisher_wrapper(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Wrapper for Inria's rasterization backend.

    .. warning::
        This function exists for comparison purpose only. Only rendered image is
        returned.

    .. warning::
        Inria's CUDA backend has its own LICENSE, so this function should be used with
        the respect to the original LICENSE at:
        https://github.com/graphdeco-inria/diff-gaussian-rasterization

    """

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device
    channels = colors.shape[-1]

    fishers = []
    depths = []
    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = get_projection_matrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        background = (
            backgrounds[cid]
            if backgrounds is not None
            else torch.zeros(3, device=device)
        )

        fisher_color = kwargs.get("fisher_color", 1.0)
        fisher_depth = kwargs.get("fisher_depth", 1.0)
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            projmatrix_raw=projection_matrix,
            gradient_power=2,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
            fisher_color=fisher_color,
            fisher_depth=fisher_depth
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = torch.zeros_like(means, requires_grad=True, device=device)
        render_colors_ = []
        for i in range(0, channels, 3):
            _colors = colors[..., i : i + 3]
            if _colors.shape[-1] < 3:
                pad = torch.zeros(
                    _colors.shape[0], 3 - _colors.shape[-1], device=device
                )
                _colors = torch.cat([_colors, pad], dim=-1)
            
            grad_means3D, grad_sh, grad_opacities, grad_scales, grad_rotations, grad_theta, grad_rho, JTJ_tau, \
              _render_colors_, radii, depth, opacity, n_touched = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=_colors if colors.dim() == 3 else None,
                colors_precomp=_colors if colors.dim() == 2 else None,
                opacities=opacities[:, None],
                scales=scales,
                rotations=quats,
                cov3D_precomp=None,
            )
            if _colors.shape[-1] < 3:
                _render_colors_ = _render_colors_[:, :, : _colors.shape[-1]]
            render_colors_.append(_render_colors_)
            fishers.append((grad_means3D, grad_sh, grad_opacities, grad_scales, grad_rotations, grad_theta, grad_rho, JTJ_tau))
            depths.append(depth)

        render_colors_ = torch.cat(render_colors_, dim=-1)
        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]
        render_colors.append(render_colors_)
    
    render_colors = torch.stack(render_colors, dim=0)
    meta = {"fishers": fishers, "depths": depths}
    return render_colors, None, meta

def equal_hist(uncern):
    """
        Equalize the histogram of the uncertainty map for visualization.
            normalize the uncertainty map to [0, 1]
    """
    shape = uncern.shape
    # H, W = uncern.shape

    # Histogram equalization for visualization
    uncern = uncern.flatten()
    median = np.median(uncern)
    bins = np.append(np.linspace(uncern.min(), median, len(uncern)), 
                            np.linspace(median, uncern.max(), len(uncern)))
    # Do histogram equalization on uncern  
    # bins = np.linspace(uncern.min(), uncern.max(), len(uncern) // 20)
    hist, bins2 = np.histogram(uncern, bins=bins)
    # Compute CDF from histogram
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf = np.hstack(([0], cdf))
    cdf = cdf / cdf[-1]
    # Do equalization
    binnum = np.digitize(uncern, bins, True) - 1
    neg = np.where(binnum < 0)
    binnum[neg] = 0
    uncern_aeq = cdf[binnum] * bins[-1]

    uncern_aeq = uncern_aeq.reshape(shape)
    uncern_aeq = (uncern_aeq - uncern_aeq.min()) / (uncern_aeq.max() - uncern_aeq.min())
    return uncern_aeq 