from typing import Callable, Tuple

import torch
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)
from pytorch3d.renderer.implicit.raysampling import RayBundle
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf

from nerffeat.utils import sample_images_at_mc_locs


class ImplicitRendererStratified(torch.nn.Module):
    """
    A class for rendering a batch of implicit surfaces. The class should
    be initialized with a raysampler and raymarcher class which both have
    to be a `Callable`.
    VOLUMETRIC_FUNCTION
    The `forward` function of the renderer accepts as input the rendering cameras
    as well as the `volumetric_function` `Callable`, which defines a field of opacity
    and feature vectors over the 3D domain of the scene.
    A standard `volumetric_function` has the following signature::
        def volumetric_function(
            ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]
    With the following arguments:
        `ray_bundle`: A RayBundle or HeterogeneousRayBundle object
            containing the following variables:
            `origins`: A tensor of shape `(minibatch, ..., 3)` denoting
                the origins of the rendering rays.
            `directions`: A tensor of shape `(minibatch, ..., 3)`
                containing the direction vectors of rendering rays.
            `lengths`: A tensor of shape
                `(minibatch, ..., num_points_per_ray)`containing the
                lengths at which the ray points are sampled.
            `xys`: A tensor of shape
                `(minibatch, ..., 2)` containing the
                xy locations of each ray's pixel in the screen space.
    Calling `volumetric_function` then returns the following:
        `rays_densities`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, opacity_dim)` containing
            the an opacity vector for each ray point.
        `rays_features`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, feature_dim)` containing
            the an feature vector for each ray point.
    Note that, in order to increase flexibility of the API, we allow multiple
    other arguments to enter the volumetric function via additional
    (optional) keyword arguments `**kwargs`.
    A typical use-case is passing a `CamerasBase` object as an additional
    keyword argument, which can allow the volumetric function to adjust its
    outputs based on the directions of the projection rays.
    Example:
        A simple volumetric function of a 0-centered
        RGB sphere with a unit diameter is defined as follows::
            def volumetric_function(
                ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
                **kwargs,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                rays_points_world = ray_bundle_to_ray_points(ray_bundle)
                rays_densities = torch.sigmoid(
                    -100.0 * rays_points_world.norm(dim=-1, keepdim=True)
                )
                rays_features = torch.nn.functional.normalize(
                    rays_points_world, dim=-1
                ) * 0.5 + 0.5
                return rays_densities, rays_features
    """

    def __init__(self, raysampler: Callable, raymarcher: Callable, device) -> None:
        super().__init__()

        if not callable(raysampler):
            raise ValueError('"raysampler" has to be a "Callable" object.')
        if not callable(raymarcher):
            raise ValueError('"raymarcher" has to be a "Callable" object.')

        self.raysampler = raysampler
        self.raymarcher = raymarcher
        self.device = device

    def compute_weights(
        self, cameras: CamerasBase, volumetric_function: Callable, **kwargs
    ) -> Tuple[torch.Tensor, RayBundle]:
        """
        Render a batch of images using a volumetric function
        represented as a callable (e.g. a Pytorch module).
        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.
            volumetric_function: A `Callable` that accepts the parametrizations
                of the rendering rays and returns the densities and features
                at the respective 3D of the rendering rays. Please refer to
                the main class documentation for details.
        Returns:
            images: A tensor of shape `(minibatch, ..., feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `Union[RayBundle, HeterogeneousRayBundle]` containing
                the parametrizations of the sampled rendering rays.
        """

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        ray_bundle = self.raysampler(
            cameras=cameras, volumetric_function=volumetric_function, **kwargs
        )
        rays_densities, rays_features = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )
        rays_densities = rays_densities[..., 0]

        eps = 1e-10
        surface_thickness = 1
        absorption = _shifted_cumprod((1.0 + eps) - rays_densities, shift=surface_thickness)
        weights = rays_densities * absorption
        return ray_bundle, weights

    def forward(
        self,
        cameras: CamerasBase,
        volumetric_function: Callable,
        mask_rays=False,
        mask_image=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, RayBundle]:
        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        with torch.no_grad():
            ray_bundle, weights = self.compute_weights(
                cameras=cameras, volumetric_function=volumetric_function
            )
            if mask_rays:
                mask_values = torch.where(
                    sample_images_at_mc_locs(mask_image.to(self.device), ray_bundle.xys)[..., 0]
                )
                ray_bundle = RayBundle(
                    origins=ray_bundle.origins[mask_values].unsqueeze(0),
                    directions=ray_bundle.directions[mask_values].unsqueeze(0),
                    lengths=ray_bundle.lengths[mask_values].unsqueeze(0),
                    xys=ray_bundle.xys[mask_values].unsqueeze(0),
                )

        rays_densities, rays_features = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )

        images, weights = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
            **kwargs,
        )
        return images, ray_bundle, weights


class EmissionAbsorptionRaymarcherStratified(torch.nn.Module):
    """
    Raymarch using the Emission-Absorption (EA) algorithm.
    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The feature values `rays_features` of shape
    `(..., n_points_per_ray, feature_dim)` represent the content of the
    point that is supposed to be rendered in case the given point is opaque
    (i.e. its density -> 1.0).
    EA first utilizes `rays_densities` to compute the absorption function
    along each ray as follows::
        absorption = cumprod(1 - rays_densities, dim=-1)
    The value of absorption at position `absorption[..., k]` specifies
    how much light has reached `k`-th point along a ray since starting
    its trajectory at `k=0`-th point.
    Each ray is then rendered into a tensor `features` of shape `(..., feature_dim)`
    by taking a weighed combination of per-ray features `rays_features` as follows::
        weights = absorption * rays_densities
        features = (rays_features * weights).sum(dim=-2)
    Where `weights` denote a function that has a strong peak around the location
    of the first surface point that a given ray passes through.
    Note that for a perfectly bounded volume (with a strictly binary density),
    the `weights = cumprod(1 - rays_densities, dim=-1) * rays_densities`
    function would yield 0 everywhere. In order to prevent this,
    the result of the cumulative product is shifted `self.surface_thickness`
    elements along the ray direction.
    """

    def __init__(self, surface_thickness: int = 1, threshold_mode=False, threshold=0.03) -> None:
        super().__init__()
        self.surface_thickness = surface_thickness
        self.threshold_mode = threshold_mode
        self.threshold = threshold

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.
        Returns:
            features_opacities: A tensor of shape `(..., feature_dim+1)`
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]

        if self.threshold_mode:
            density_mask = torch.zeros_like(rays_densities)
            density_mask[torch.where(rays_densities > self.threshold)] = 1
            rays_densities = density_mask

        absorption = _shifted_cumprod((1.0 + eps) - rays_densities, shift=self.surface_thickness)
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)

        return torch.cat((features, opacities), dim=-1), weights


class ProbabilisticRaysampler(torch.nn.Module):
    """
    Implements the importance sampling of points along rays.
    The input is a `RayBundle` object with a `ray_weights` tensor
    which specifies the probabilities of sampling a point along each ray.
    This raysampler is used for the fine rendering pass of NeRF.
    As such, the forward pass accepts the RayBundle output by the
    raysampling of the coarse rendering pass. Hence, it does not
    take cameras as input.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        stratified: bool,
        stratified_test: bool,
        add_input_samples: bool = True,
    ):
        """
        Args:
            n_pts_per_ray: The number of points to sample along each ray.
            stratified: If `True`, the input `ray_weights` are assumed to be
                sampled at equidistant intervals.
            stratified_test: Same as `stratified` with the difference that this
                setting is applied when the module is in the `eval` mode
                (`self.training==False`).
            add_input_samples: Concatenates and returns the sampled values
                together with the input samples.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._stratified = stratified
        self._stratified_test = stratified_test
        self._add_input_samples = add_input_samples

    def forward(
        self,
        input_ray_bundle: RayBundle,
        ray_weights: torch.Tensor,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            input_ray_bundle: An instance of `RayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.legths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.
        Returns:
            ray_bundle: A new `RayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additional sampled
                points per ray.
        """

        z_vals = input_ray_bundle.lengths
        batch_size = z_vals.shape[0]

        with torch.no_grad():
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid.view(-1, z_vals_mid.shape[-1]),
                ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1],
                self._n_pts_per_ray,
                det=not (
                    (self._stratified and self.training)
                    or (self._stratified_test and not self.training)
                ),
            ).view(batch_size, z_vals.shape[1], self._n_pts_per_ray)

        if self._add_input_samples:
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=input_ray_bundle.origins,
            directions=input_ray_bundle.directions,
            lengths=z_vals,
            xys=input_ray_bundle.xys,
        )
