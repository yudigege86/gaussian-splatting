#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import fov2focal

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
    packed=False,
    sparse_grad=False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)

    fx = fov2focal(viewpoint_camera.FoVx, width)
    fy = fov2focal(viewpoint_camera.FoVy, height)
    cx = width / 2.0
    cy = height / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=pc.get_xyz.device,
        dtype=torch.float32,
    )

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(pc.get_xyz.device)

    means3D = pc.get_xyz
    opacity = pc.get_opacity.squeeze(-1)
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    sh_degree = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(
                eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0
            )
            colors = colors_precomp
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
                colors = torch.cat((dc, shs), dim=1)
            else:
                colors = pc.get_features
            sh_degree = pc.active_sh_degree
    else:
        colors = override_color

    rasterize_mode = "antialiased" if pipe.antialiasing else "classic"
    render_mode = "RGB+ED"
    render_colors, render_alphas, info = rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=packed,
        sparse_grad=sparse_grad,
        rasterize_mode=rasterize_mode,
        render_mode=render_mode,
        backgrounds=bg_color[None],
    )

    info["width"] = width
    info["height"] = height
    info["n_cameras"] = 1

    rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1).clamp(0, 1)
    depth = render_colors[0, ..., 3]
    depth_image = 1.0 / depth.clamp_min(1e-6)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter": (info["radii"] > 0).all(dim=-1).nonzero(),
        "radii": info["radii"],
        "depth": depth_image,
        "info": info,
        }
    
    return out
