
from dataclasses import dataclass

import torch
from pytorch3d.renderer import MonteCarloRaysampler, NDCMultinomialRaysampler

from nerffeat.rendering.fine import EmissionAbsorptionRaymarcherStratified as FineRaymarcher
from nerffeat.rendering.fine import ImplicitRendererStratified as FineRenderer
from nerffeat.rendering.stratified import (
    EmissionAbsorptionRaymarcherStratified,
    ImplicitRendererStratified,
)


@dataclass
class NerfRendererBundle:
    coarse: ImplicitRendererStratified
    fine: FineRenderer
    eval_grid: ImplicitRendererStratified
    coarse_raymarcher: EmissionAbsorptionRaymarcherStratified
    fine_raymarcher: FineRaymarcher


@dataclass
class SingleRendererBundle:
    monte_carlo: ImplicitRendererStratified
    eval_grid: ImplicitRendererStratified
    raymarcher: EmissionAbsorptionRaymarcherStratified


def _mc_raysampler(config: dict, min_depth: float, max_depth: float) -> MonteCarloRaysampler:
    return MonteCarloRaysampler(
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        n_rays_per_image=int(config["rays_per_image"]),
        n_pts_per_ray=int(config["points_per_ray"]),
        min_depth=min_depth,
        max_depth=max_depth,
        stratified_sampling=True,
    )


def _grid_raysampler(config: dict, min_depth: float, max_depth: float) -> NDCMultinomialRaysampler:
    return NDCMultinomialRaysampler(
        image_height=int(config["image_size"]),
        image_width=int(config["image_size"]),
        n_pts_per_ray=int(config["points_per_ray"]),
        min_depth=min_depth,
        max_depth=max_depth,
    )


def build_nerf_renderers(
    *,
    config: dict,
    render_size: int,
    min_depth: float,
    max_depth: float,
    device: torch.device,
) -> NerfRendererBundle:
    coarse_raymarcher = EmissionAbsorptionRaymarcherStratified()
    fine_raymarcher = FineRaymarcher()
    coarse_raymarcher.threshold_mode = bool(config["threshold_mode"])
    fine_raymarcher.threshold_mode = bool(config["threshold_mode"])

    eval_grid_config = {**config["eval_grid"], "image_size": render_size}

    bundle = NerfRendererBundle(
        coarse=ImplicitRendererStratified(
            raysampler=_mc_raysampler(config["coarse"], min_depth, max_depth),
            raymarcher=coarse_raymarcher,
            device=device,
        ),
        fine=FineRenderer(
            raysampler=_mc_raysampler(config["fine"], min_depth, max_depth),
            raymarcher=fine_raymarcher,
            device=device,
        ),
        eval_grid=ImplicitRendererStratified(
            raysampler=_grid_raysampler(eval_grid_config, min_depth, max_depth),
            raymarcher=coarse_raymarcher,
            device=device,
        ),
        coarse_raymarcher=coarse_raymarcher,
        fine_raymarcher=fine_raymarcher,
    )
    bundle.coarse = bundle.coarse.to(device)
    bundle.fine = bundle.fine.to(device)
    bundle.eval_grid = bundle.eval_grid.to(device)
    return bundle


def build_single_renderers(
    *,
    config: dict,
    render_size: int,
    min_depth: float,
    max_depth: float,
    device: torch.device,
) -> SingleRendererBundle:
    raymarcher = EmissionAbsorptionRaymarcherStratified()
    eval_grid_config = {**config["eval_grid"], "image_size": render_size // 2}
    bundle = SingleRendererBundle(
        monte_carlo=ImplicitRendererStratified(
            raysampler=_mc_raysampler(config["monte_carlo"], min_depth, max_depth),
            raymarcher=raymarcher,
            device=device,
        ),
        eval_grid=ImplicitRendererStratified(
            raysampler=_grid_raysampler(eval_grid_config, min_depth, max_depth),
            raymarcher=raymarcher,
            device=device,
        ),
        raymarcher=raymarcher,
    )
    bundle.monte_carlo = bundle.monte_carlo.to(device)
    bundle.eval_grid = bundle.eval_grid.to(device)
    return bundle
