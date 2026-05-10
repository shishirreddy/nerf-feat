import numpy as np
import torch
from pytorch3d.renderer import ray_bundle_to_ray_points
from pytorch3d.renderer.implicit.raysampling import RayBundle

from nerffeat.models.siren import Siren


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega_0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega_0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega_0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceFieldFeat(torch.nn.Module):
    """NeRF-Feat radiance field with a shared density/color MLP and a detachable feature branch.

    Args:
        n_harmonic_functions: Number of harmonic frequencies for positional encoding.
        hidden_dim: Hidden width for the density/color MLPs.
        feature_dim: Output dimensionality of the surface feature branch.
        use_siren: Use a SIREN network for the feature branch instead of a Softplus MLP.
        mode: Active inference branch — ``"color"``, ``"feature"``, or ``"color+feature"``.
    """

    def __init__(
        self,
        n_harmonic_functions=60,
        hidden_dim=256,
        feature_dim=12,
        use_siren=False,
        mode="color",
    ):
        super().__init__()
        self.mode = mode
        self.use_siren = use_siren
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        embedding_dim = n_harmonic_functions * 2 * 3

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Softplus(beta=10.0),
        )

        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(hidden_dim, 3),
            torch.nn.Sigmoid(),
        )

        if not use_siren:
            self.feature_layer = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hidden_dim),
                torch.nn.Softplus(beta=10.0),
                torch.nn.Linear(hidden_dim, feature_dim),
                torch.nn.Sigmoid(),
            )
        else:
            self.feature_layer = Siren(
                in_features=3,
                out_features=feature_dim,
                hidden_features=hidden_dim,
                hidden_layers=2,
            )
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Softplus(beta=10.0),
        )

        self.density_layer[0].bias.data[0] = -1.5

    def _get_densities(self, features):
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        dirs_emb = self.harmonic_embedding(
            torch.nn.functional.normalize(rays_directions, dim=-1)
        )
        dirs_emb = dirs_emb[..., None, :].expand(*features.shape[:-1], dirs_emb.shape[-1])
        return self.color_layer(torch.cat((features, dirs_emb), dim=-1))

    def _get_features_colors(self, features, rays_directions, embeds):
        dirs_emb = self.harmonic_embedding(
            torch.nn.functional.normalize(rays_directions, dim=-1)
        )
        dirs_emb = dirs_emb[..., None, :].expand(*features.shape[:-1], dirs_emb.shape[-1])
        color_out = self.color_layer(torch.cat((features, dirs_emb), dim=-1))
        return torch.cat([color_out, self.feature_layer(embeds)], dim=-1)

    def _get_features(self, features):
        return self.feature_layer(features)

    def forward(
        self,
        ray_bundle: RayBundle,
        cameras=None,  # passed by ImplicitRendererStratified; unused (directions come from ray_bundle)
    ):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        embeds = self.harmonic_embedding(rays_points_world)

        features = self.mlp(embeds)

        rays_densities = self._get_densities(features)
        if self.mode == "feature":
            rays_colors = self._get_features(rays_points_world if self.use_siren else embeds)
        elif self.mode == "color":
            rays_colors = self._get_colors(features, ray_bundle.directions)
        else:
            rays_colors = self._get_features_colors(features, ray_bundle.directions, embeds)

        return rays_densities, rays_colors

    def forward_features_at_points(self, rays_points_world):
        if self.use_siren:
            features = self._get_features(rays_points_world)
        else:
            features = self._get_features(self.harmonic_embedding(rays_points_world))
        dummy_density = torch.zeros(*features.shape[:-1], 1, device=features.device)
        return torch.cat([features, dummy_density], dim=-1)

    def forward_density_at_points(self, rays_points_world):
        embeds = self.harmonic_embedding(rays_points_world)

        features = self.mlp(embeds)

        rays_densities = self._get_densities(features)
        return rays_densities

    def batched_forward_features_at_points(
        self,
        rays_points_world,
        n_batches: int = 16,
    ):
        tot_samples = rays_points_world.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)
        spatial_size = [*rays_points_world.shape[:-1]]
        batch_outputs = [
            self.forward_features_at_points(rays_points_world.view(-1, 3)[batch_idx])
            for batch_idx in batches
        ]
        return torch.cat(batch_outputs, dim=0).view(*spatial_size, -1)

    def batched_forward(
        self,
        ray_bundle: RayBundle,
        n_batches: int = 16,
        cameras=None,  # passed by ImplicitRendererStratified; forwarded to self.forward
    ):

        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            )
            for batch_idx in batches
        ]

        rays_densities, rays_colors = [
            torch.cat([b[i] for b in batch_outputs], dim=0).view(*spatial_size, -1)
            for i in (0, 1)
        ]
        return rays_densities, rays_colors

    def batched_forward_density(
        self,
        ray_bundle: RayBundle,
        n_batches: int = 16,
    ) -> torch.Tensor:
        """Memory-efficient density-only forward pass, splitting rays into batches."""

        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        batch_outputs = [
            self.forward_density(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            )
            for batch_idx in batches
        ]

        return torch.cat(batch_outputs, dim=0).view(*spatial_size, -1)

    def forward_density(
        self,
        ray_bundle: RayBundle,
    ):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        embeds = self.harmonic_embedding(rays_points_world)

        features = self.mlp(embeds)

        rays_densities = self._get_densities(features)

        return rays_densities

    def extract_mesh(self, threshold=0.1):
        n_batches = 16
        grid_resolution = 128
        coordinates = np.linspace(-1, 1, grid_resolution)
        grid_points = np.asarray(
            [[z, y, x] for x in coordinates for y in coordinates for z in coordinates],
            dtype="float32",
        )

        model_device = next(self.parameters()).device
        grid_coords = torch.from_numpy(
            grid_points.reshape(grid_resolution, grid_resolution, grid_resolution, 3)
        ).to(model_device)

        batches = torch.chunk(torch.arange(grid_coords.view(-1, 3).shape[0]), n_batches)

        rays_densities = torch.cat(
            [self.forward_at_points(grid_coords.view(-1, 3)[batch_idx]) for batch_idx in batches],
            dim=0,
        ).view(grid_resolution, grid_resolution, grid_resolution, -1)

        import mcubes

        mvertices, mtriangles = mcubes.marching_cubes(
            rays_densities[:, :, :, 0].movedim(0, 2).movedim(1, 0).cpu().numpy(), threshold
        )
        return (mvertices - grid_resolution / 2) / (grid_resolution / 2), mtriangles

    def forward_at_points(self, rays_points_world) -> torch.Tensor:
        embeds = self.harmonic_embedding(rays_points_world)
        features = self.mlp(embeds)
        return self._get_densities(features)
