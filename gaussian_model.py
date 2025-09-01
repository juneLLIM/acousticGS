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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):

            L = build_scaling_rotation(
                scaling_modifier * scaling, rotation)
            covariance = L @ L.transpose(1, 2)

            return covariance

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.active_sh_degree = 0
        self.max_sh_degree = config.model.sh_degree
        self.seq_len = config.audio.seq_len
        self._mean = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.mean_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None

        self.gaussian_dim = config.model.gaussian_dim
        if self.gaussian_dim == 3:
            self.t = torch.empty(0)
            self.spatial_lr_scale = (
                config.rendering.coord_max - config.rendering.coord_min) * (3**0.5) / 2.0   # radius
            print("3D Gaussian model selected")
        elif self.gaussian_dim == 4:
            self.spatial_lr_scale = (
                (config.rendering.coord_max - config.rendering.coord_min)**2*3+self.seq_len**2)**0.5 / 2.0  # radius
            print("4D Gaussian model selected")
        else:
            raise ValueError("Invalid Gaussian dimension")

        self.setup_functions()

    def forward(self, network_pts, network_view=None, network_tx=None):
        return self.render_signal_at_points(network_pts)

    def capture(self):
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._mean,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.mean_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.t
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._mean,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.mean_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
            )

    def restore(self, model_args):
        if self.gaussian_dim == 3:
            (self.active_sh_degree,
             self._mean,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.mean_gradient_accum,
             self.denom,
             opt_dict,
             self.t) = model_args
        elif self.gaussian_dim == 4:
            (self.active_sh_degree,
             self._mean,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.mean_gradient_accum,
             self.denom,
             opt_dict,) = model_args
        self.setup_optimizer()
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self.gaussian_dim == 3:
            return self.rotation_activation(self._rotation)
        else:
            return self.rotation_activation(self._rotation.view(-1, 4, 2)).view(-1, 8)

    @property
    def get_mean(self):
        return self._mean

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def eval_features(self, query_points):

        # query_points: (B, 3)
        B = query_points.shape[0]
        N = self.get_mean.shape[0]

        # Pre-fetch gaussian features
        mean = self.get_mean[:, :3]    # (N, 3)
        sh = self.get_features  # (N, C)

        dir_pp = query_points.unsqueeze(1) - mean    # (B, N, 3)

        dir_pp_normalized = dir_pp / \
            (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-8)   # (B, N, 3)

        # Evaluate spherical harmonics (B, N, C)
        features = eval_sh(
            self.active_sh_degree,
            sh.repeat(B, 1, 1).unsqueeze(-2),  # (B, N, 1, C)
            dir_pp_normalized  # (B, N, 3)
        )

        return features.squeeze(-1)  # (B, N)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_random(self):

        count = self.config.model.initial_points

        if self.gaussian_dim == 3:
            mean = torch.rand(count, 3, device="cuda") * \
                (self.config.rendering.coord_max - self.config.rendering.coord_min) + \
                self.config.rendering.coord_min

            # Initialize with small scales
            scales = torch.ones((count, 3), device="cuda") * 0.01

            # Initialize as identity quaternions
            rots = torch.zeros((count, 4), device="cuda")
            rots[:, 0] = 1

            self.t = torch.randint(self.seq_len, (count,), device="cuda")
        else:
            mean = torch.rand(count, 4, device="cuda") * \
                (self.config.rendering.coord_max - self.config.rendering.coord_min) + \
                self.config.rendering.coord_min
            # Initialize with small scales
            scales = torch.ones((count, 4), device="cuda") * 0.01

            # Initialize as identity quaternions
            rots = torch.zeros((count, 8), device="cuda")
            rots[:, 0] = 1
            rots[:, 4] = 1

        # Initialize SH feature for magnitude
        # DC component is initialized to 1.0, rest are 0.
        features = torch.zeros(
            (count, (self.max_sh_degree + 1) ** 2), device="cuda")
        features[:, 0] = 1.0

        # Initialize with low opacity
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((count, 1), dtype=torch.float, device="cuda"))

        self._mean = nn.Parameter(mean.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, 0:1].requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, 1:].requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.mean_gradient_accum = torch.zeros((count, 1), device="cuda")
        self.denom = torch.zeros((count, 1), device="cuda")

        self.setup_optimizer()

        print(f"Created {count} random gaussians.")

    def setup_optimizer(self):

        l = [
            {'params': [self._mean], 'lr': self.config.optimizer.position_lr_init *
                self.spatial_lr_scale, "name": "mean"},
            {'params': [self._features_dc],
                'lr': self.config.optimizer.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest],
                'lr': self.config.optimizer.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity],
                'lr': self.config.optimizer.opacity_lr, "name": "opacity"},
            {'params': [self._scaling],
                'lr': self.config.optimizer.scaling_lr, "name": "scaling"},
            {'params': [self._rotation],
                'lr': self.config.optimizer.rotation_lr, "name": "rotation"},
        ]

        optimizer_type = self.config.optimizer.type.lower()

        if optimizer_type == "default" or optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.mean_scheduler_args = get_expon_lr_func(lr_init=self.config.optimizer.position_lr_init * self.spatial_lr_scale,
                                                     lr_final=self.config.optimizer.position_lr_final * self.spatial_lr_scale,
                                                     lr_delay_mult=self.config.optimizer.position_lr_delay_mult,
                                                     max_steps=self.config.optimizer.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mean":
                lr = self.mean_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(
                    group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._mean = optimizable_tensors["mean"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.mean_gradient_accum = self.mean_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_mean, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"mean": new_mean,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._mean = optimizable_tensors["mean"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.mean_gradient_accum = torch.zeros(
            (self.get_mean.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_mean.shape[0], 1), device="cuda")

    def densify_and_split(self, grads, N=2):
        n_init_points = self.get_mean.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= self.config.densification.min_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(
            self.get_scaling, dim=1).values > self.config.densification.threshold_scale)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros_like(self.get_mean[selected_pts_mask].repeat(N, 1))
        rots = build_rotation(
            self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        samples = torch.normal(mean=means, std=stds)
        new_mean = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
            self.get_mean[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_mean, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(
            grads, dim=-1) >= self.config.densification.min_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(
            self.get_scaling, dim=1).values <= self.config.densification.threshold_scale)

        new_mean = self._mean[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_mean, new_features_dc, new_features_rest, new_opacities,
                                   new_scaling, new_rotation)

    def densify_and_prune(self):

        min_opacity = self.config.densification.min_opacity
        max_scale = self.config.densification.max_scale

        grads = self.mean_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads)
        self.densify_and_split(grads)

        prune_mask = (self.get_opacity < min_opacity).squeeze() & (
            self.get_scaling.max(dim=1).values < max_scale)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, update_filter):
        self.mean_gradient_accum[update_filter] += torch.norm(
            self.get_mean.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render_signal_at_points(self, query_points):
        """
        Calculates the final signal at a set of 3D query points.

        Args:
            query_points (torch.Tensor): (B, 3) tensor of query points.

        Returns:
            final_signal (torch.Tensor): (B, seq_len) tensor of rendered audio signals in time domain.
        """
        B = query_points.shape[0]
        N = self.get_mean.shape[0]
        if N == 0:
            return torch.zeros(B, self.seq_len, device="cuda"), torch.zeros(B, device="cuda")

        speed = self.config.rendering.speed
        mean = self.get_mean[:, :3]    # (N, 3)
        t = self.get_mean[:, 3] if self.gaussian_dim == 4 else self.t   # (N)
        opacity = self.get_opacity  # (N, 1)
        covariance = self.get_covariance()  # (N, 4, 4)
        sh = self.eval_features(query_points)  # (B, N)

        diff = query_points.unsqueeze(1) - mean.unsqueeze(0)  # (B, N, 3)
        dist = torch.norm(diff, dim=-1)  # (B, N)
        jacobian = torch.cat(
            [diff/(dist.unsqueeze(-1)*speed), torch.ones_like(dist.unsqueeze(-1))], dim=-1)  # (B, N, 4)

        rasterized_mean = t + dist/speed  # (B, N)
        rasterized_var = torch.einsum("bnc,ncc,bnc->bn",  # (B, N)
                                      jacobian, covariance, jacobian)

        time = torch.arange(self.seq_len, device="cuda")  # (seq_len)

        opacity = opacity.unsqueeze(0)  # (1, N, 1)
        sh = sh.unsqueeze(-1)  # (B, N, 1)
        rasterized_mean = rasterized_mean.unsqueeze(-1)  # (B, N, 1)
        rasterized_var = rasterized_var.unsqueeze(-1)  # (B, N, 1)
        time = time.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)

        power = torch.exp(-0.5*(time-rasterized_mean) ** 2 / rasterized_var)
        final_signal = (opacity * sh * power).sum(dim=1)  # (B, seq_len)

        return final_signal  # (B, seq_len)
