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

from functools import partial
import torch
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling_rotation
from torch import nn
import torch.nn.functional as F
from utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):

            L = build_scaling_rotation(
                scaling_modifier * scaling, rotation, device=self.device)
            covariance = L @ L.transpose(1, 2)

            return covariance

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.stft = partial(
            torch.stft, n_fft=self.config.audio.n_fft, return_complex=True, hop_length=self.config.audio.hop_length, window=self.window)

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
        self.device = torch.device(config.device)
        self.span = config.rendering.coord_max - config.rendering.coord_min
        self.window = torch.hann_window(
            self.config.audio.n_fft, device=self.device)

        self.gaussian_version = config.model.gaussian_version
        if self.gaussian_version == 1:
            self.gaussian_dim = 3
            self.t = torch.empty(0)
            self.f = torch.empty(0)
            print("3D Gaussian model with xyz axis selected")
        elif self.gaussian_version == 2:
            self.gaussian_dim = 4
            self.f = torch.empty(0)
            print("4D Gaussian model with xyzt axis selected")
        elif self.gaussian_version == 3:
            self.gaussian_dim = 4
            self.t = torch.empty(0)
            print("4D Gaussian model with xyzf axis selected")
        elif self.gaussian_version == 4:
            self.gaussian_dim = 5
            print("5D Gaussian model with xyzft axis selected")
        else:
            raise ValueError("Invalid Gaussian version selected.")

        self.setup_functions()

        dummy_input = torch.randn(1, self.seq_len, device=self.device)
        dummy_stft = self.stft(dummy_input)
        _, self.f_len, self.t_len = dummy_stft.shape

    def forward(self, position_rx, network_view=None, position_tx=None):
        query_points = self.normalize_points(position_rx)
        return self.render_signal_at_points(query_points)

    def backward(self, total_loss):
        total_loss.backward()
        self.add_densification_stats(self.update_filter)

    def capture(self):
        model_data = [
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
        ]

        if self.gaussian_version == 1:
            model_data.extend([self.t, self.f])
        elif self.gaussian_version == 2:
            model_data.append(self.f)
        elif self.gaussian_version == 3:
            model_data.append(self.t)

        return tuple(model_data)

    def restore(self, model_args):
        (self.active_sh_degree,
         self._mean,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.mean_gradient_accum,
         self.denom,
         opt_dict) = model_args[:10]

        extra_args = model_args[10:]
        if self.gaussian_version == 1:
            self.t, self.f = extra_args
        elif self.gaussian_version == 2:
            self.f = extra_args[0]
        elif self.gaussian_version == 3:
            self.t = extra_args[0]

        self.setup_optimizer()
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self.gaussian_dim == 3:
            return self.rotation_activation(self._rotation)
        elif self.gaussian_dim == 4:
            return self.rotation_activation(self._rotation.view(-1, 4, 2)).view(-1, 8)
        elif self.gaussian_dim == 5:
            return self.rotation_activation(self._rotation)

    @property
    def get_mean(self):
        return self._mean

    @property
    def get_xyztf(self):
        if self.gaussian_version == 1:
            return torch.cat((self._mean, self.t.unsqueeze(-1), self.f.unsqueeze(-1)), dim=1)
        elif self.gaussian_version == 2:
            return torch.cat((self._mean, self.f.unsqueeze(-1)), dim=1)
        elif self.gaussian_version == 3:
            return torch.cat((self._mean, self.t.unsqueeze(-1)), dim=1)
        elif self.gaussian_version == 4:
            return self._mean

    @property
    def get_xyz(self):
        return self._mean[:, :3]

    @property
    def get_t(self):
        if self.gaussian_version in [1, 3]:
            return self.t
        elif self.gaussian_version in [2, 4]:
            return self._mean[:, 3]

    @property
    def get_f(self):
        if self.gaussian_version in [1, 2]:
            return self.f
        elif self.gaussian_version == 3:
            return self._mean[:, 3]
        elif self.gaussian_version == 4:
            return self._mean[:, 4]

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

    def eval_features(self, sh, dir_pp):

        dir_pp_normalized = dir_pp / \
            (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-8)   # (K, 3)

        # Evaluate spherical harmonics
        features = eval_sh(
            self.active_sh_degree,
            sh.unsqueeze(1),  # (K, 1, C)
            dir_pp_normalized  # (K, 3)
        )

        return features.squeeze(-1)  # (K)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def initialize(self, position_tx=None):
        if position_tx is not None and self.config.model.init_from_data is True:
            self.create_from_simulation(position_tx)
        else:
            self.create_random()

    def create_random(self):

        count = self.config.model.initial_points

        # Initialize mean, scaling(small scales), rotation(identity quaternions or rotors)
        if self.gaussian_dim == 3:
            mean = torch.rand(count, 3, device=self.device) * 2 - 1
            scales = torch.full((count, 3), 0.01, device=self.device).log()
            rots = torch.zeros((count, 4), device=self.device)
            rots[:, 0] = 1
        elif self.gaussian_dim == 4:
            mean = torch.rand(count, 4, device=self.device) * 2 - 1
            scales = torch.full((count, 4), 0.01, device=self.device).log()
            rots = torch.zeros((count, 8), device=self.device)
            rots[:, 0] = 1
            rots[:, 4] = 1
        elif self.gaussian_dim == 5:
            mean = torch.rand(count, 5, device=self.device) * 2 - 1
            scales = torch.full((count, 5), 0.01, device=self.device).log()
            rots = torch.zeros((count, 10), device=self.device)
        else:
            raise ValueError("Invalid Gaussian dimension.")

        # Initialize time and frequency
        if self.gaussian_version == 1:
            self.t = torch.randint(
                self.t_len, (count,), device=self.device) / self.t_len * 2 - 1
            self.f = torch.randint(
                self.f_len, (count,), device=self.device) / self.f_len * 2 - 1
        elif self.gaussian_version == 2:
            self.f = torch.randint(
                self.f_len, (count,), device=self.device) / self.f_len * 2 - 1
        elif self.gaussian_version == 3:
            self.t = torch.randint(
                self.t_len, (count,), device=self.device) / self.t_len * 2 - 1
        elif self.gaussian_version == 4:
            pass

        # Initialize SH feature to stft values
        # DC component is initialized to random magnitude in [0,1), and phase in [-pi,pi), rest are 0.
        features = torch.zeros(
            (count, (self.max_sh_degree + 1) ** 2), device=self.device, dtype=torch.complex64)
        magnitudes = torch.rand((count,), device=self.device)
        phases = (torch.rand((count,), device=self.device) * 2 - 1) * torch.pi
        real = magnitudes * torch.cos(phases)
        imag = magnitudes * torch.sin(phases)
        features[:, 0] = torch.complex(real, imag)

        # Initialize with low opacity
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((count, 1), dtype=torch.float, device=self.device))

        self._mean = nn.Parameter(mean.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, 0:1].requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, 1:].requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.mean_gradient_accum = torch.zeros((count, 1), device=self.device)
        self.denom = torch.zeros((count, 1), device=self.device)

        self.setup_optimizer()

        print(f"Created {count} random gaussians.")

    def create_from_simulation(self, position_tx):

        import pyroomacoustics as pra
        import time
        import numpy as np

        # Create initial points in a grid
        count_sqrt = int(self.config.model.initial_points ** 0.5)
        count = count_sqrt * count_sqrt
        mean = torch.rand(count_sqrt, 3, device=self.device) * 2 - 1
        t = torch.randint(self.t_len, (count,), device=self.device)
        f = torch.randint(self.f_len, (count,), device=self.device)

        # Simulate RIR at grid points
        chrono = time.time()
        room = pra.ShoeBox(
            (self.span, self.span, self.span),
            fs=self.config.audio.fs,
            absorption=0.2,
            air_absorption=True,
            max_order=3,
            ray_tracing=False,
            use_rand_ism=True)
        pos_src = position_tx - self.config.rendering.coord_min
        room.add_source(pos_src)
        room.add_microphone((mean.T.cpu() + 1) / 2 * self.span)
        room.compute_rir()
        print("Simulation done in", time.time() - chrono, "seconds.")

        # Initialize SH feature to stft values
        # DC component is initialized to RIR results, rest are 0.
        signals = torch.from_numpy(
            np.array([rir[0][:self.seq_len] for rir in room.rir])).to(device=self.device)
        idx = torch.arange(
            count_sqrt, device=self.device).repeat_interleave(count_sqrt)
        stft = self.stft(signals)[idx, f, t]

        features = torch.zeros(
            (count, (self.max_sh_degree + 1) ** 2), device=self.device, dtype=torch.complex64)

        features[:, 0] = stft

        # Fix format
        mean = mean.repeat_interleave(count_sqrt, dim=0)
        t = t / self.t_len * 2 - 1
        f = f / self.f_len * 2 - 1

        # Initialize scaling(small scales) and rotation(identity quaternions or rotors)
        if self.gaussian_dim == 3:
            scales = torch.full((count, 3), 0.01, device=self.device).log()
            rots = torch.zeros((count, 4), device=self.device)
            rots[:, 0] = 1
        elif self.gaussian_dim == 4:
            scales = torch.full((count, 4), 0.01, device=self.device).log()
            rots = torch.zeros((count, 8), device=self.device)
            rots[:, 0] = 1
            rots[:, 4] = 1
        elif self.gaussian_dim == 5:
            scales = torch.full((count, 5), 0.01, device=self.device).log()
            rots = torch.zeros((count, 10), device=self.device)
        else:
            raise ValueError("Invalid Gaussian dimension.")

        # Initialize mean, time, and frequency
        if self.gaussian_version == 1:
            self.t = t
            self.f = f
        elif self.gaussian_version == 2:
            mean = torch.cat((mean, t.unsqueeze(-1)), dim=1)
            self.f = f
        elif self.gaussian_version == 3:
            mean = torch.cat((mean, f.unsqueeze(-1)), dim=1)
            self.t = t
        elif self.gaussian_version == 4:
            mean = torch.cat((mean, t.unsqueeze(-1), f.unsqueeze(-1)), dim=1)

        # Initialize with low opacity
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((count, 1), dtype=torch.float, device=self.device))

        self._mean = nn.Parameter(mean.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, 0:1].requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, 1:].requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.mean_gradient_accum = torch.zeros((count, 1), device=self.device)
        self.denom = torch.zeros((count, 1), device=self.device)

        self.setup_optimizer()

        print(f"Created {count} random gaussians.")

    def setup_optimizer(self):

        l = [
            {'params': [self._mean],
                'lr': self.config.optimizer.position_lr_init, "name": "mean"},
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
        self.mean_scheduler_args = get_expon_lr_func(lr_init=self.config.optimizer.position_lr_init,
                                                     lr_final=self.config.optimizer.position_lr_final,
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

        if self.gaussian_version == 1:
            self.t = self.t[valid_points_mask]
            self.f = self.f[valid_points_mask]
        elif self.gaussian_version == 2:
            self.f = self.f[valid_points_mask]
        elif self.gaussian_version == 3:
            self.t = self.t[valid_points_mask]

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

    def densification_postfix(self, new_mean, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t=None, new_f=None):
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

        if self.gaussian_version == 1:
            self.t = torch.cat((self.t, new_t), dim=0)
            self.f = torch.cat((self.f, new_f), dim=0)
        elif self.gaussian_version == 2:
            self.f = torch.cat((self.f, new_f), dim=0)
        elif self.gaussian_version == 3:
            self.t = torch.cat((self.t, new_t), dim=0)

        self.mean_gradient_accum = torch.zeros(
            (self.get_mean.shape[0], 1), device=self.device)
        self.denom = torch.zeros(
            (self.get_mean.shape[0], 1), device=self.device)

    def densify_and_split(self, grads, N=2):
        n_init_points = self.get_mean.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= self.config.densification.min_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(
            self.get_scaling, dim=1).values > self.config.densification.threshold_scale)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros_like(self.get_mean[selected_pts_mask].repeat(N, 1))
        rots = build_rotation(
            self._rotation[selected_pts_mask], device=self.device).repeat(N, 1, 1)
        samples = torch.normal(mean=means, std=stds)
        new_mean = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
            self.get_mean[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_t = None
        new_f = None
        if self.gaussian_version == 1:
            new_t = self.t[selected_pts_mask].repeat(N)
            new_f = self.f[selected_pts_mask].repeat(N)
        elif self.gaussian_version == 2:
            new_f = self.f[selected_pts_mask].repeat(N)
        elif self.gaussian_version == 3:
            new_t = self.t[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_mean, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_f)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
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

        new_t = None
        new_f = None
        if self.gaussian_version == 1:
            new_t = self.t[selected_pts_mask]
            new_f = self.f[selected_pts_mask]
        elif self.gaussian_version == 2:
            new_f = self.f[selected_pts_mask]
        elif self.gaussian_version == 3:
            new_t = self.t[selected_pts_mask]

        self.densification_postfix(new_mean, new_features_dc, new_features_rest, new_opacities,
                                   new_scaling, new_rotation, new_t, new_f)

    def densify_and_prune(self):

        min_opacity = self.config.densification.min_opacity
        min_scale = self.config.densification.min_scale
        max_scale = self.config.densification.max_scale

        grads = self.mean_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads)
        self.densify_and_split(grads)

        prune_mask = (self.get_opacity < min_opacity).squeeze() \
            | (self.get_scaling.max(dim=1).values < min_scale) \
            | (self.get_scaling.max(dim=1).values > max_scale) \
            | ((self.get_mean > 1) | (self.get_mean < -1)).any(dim=1)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, update_filter):
        self.mean_gradient_accum[update_filter] += torch.norm(
            self.get_mean.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def normalize_speed(self, speed):
        # Both should be multiplied by 2 originally
        dist_ratio = 1 / (self.config.rendering.coord_max -
                          self.config.rendering.coord_min)
        time_ratio = self.config.audio.fs / self.seq_len
        return speed * dist_ratio / time_ratio

    def normalize_points(self, input_pts):
        return 2 * (input_pts - self.config.rendering.coord_min) / self.span - 1

    def denormalize_points(self, input_pts):
        return (input_pts + 1) / 2 * self.span + self.config.rendering.coord_min

    def project(self, query_points):
        """
        Projects gaussians to the given query points.

        Args:
            query_points (torch.Tensor): (B, 3) tensor of normalized query points.

        Returns:
            projected_mean (torch.Tensor): (B, N) tensor of projected gaussian means.
            projected_var (torch.Tensor): (B, N) tensor of projected gaussian variances.
        """

        xyz = self.get_xyz    # (N, 3)
        t = self.get_t  # (N)
        f = self.get_f  # (N)
        v = self.normalize_speed(self.config.rendering.speed)
        d = query_points.unsqueeze(1) - xyz.unsqueeze(0)  # (B, N, 3)
        l = torch.norm(d, dim=-1)  # (B, N)
        s = self.get_scaling    # (N, ?)

        # Far-field culling
        # Conic culling omitted since speed of sound is much faster than spatial distance
        with torch.no_grad():
            dist_mask = l + \
                s.max(dim=1)[0] < self.config.rendering.cull_distance
            if not dist_mask.any():
                return [torch.empty(0, device=self.device)] * 5
            b_idx, n_idx = dist_mask.nonzero(as_tuple=True)

            # Sort by distance within each batch
            _, b_cnt = torch.unique_consecutive(b_idx, return_counts=True)

            sorted_l_indices = torch.cat([
                torch.argsort(l_split) + offset for l_split, offset
                in zip(l[b_idx, n_idx].split(b_cnt.tolist()), F.pad(b_cnt[:-1], (1, 0)).cumsum(0))
            ])

        # Apply culling with l sorted
        b_idx = b_idx[sorted_l_indices]
        n_idx = n_idx[sorted_l_indices]
        t = t[n_idx]
        f = f[n_idx]
        d = d[b_idx, n_idx, :]
        l = l[b_idx, n_idx]
        s = s[n_idx, :]

        R = build_rotation(self._rotation[n_idx], device=self.device)

        # Projection
        if self.gaussian_version == 2:
            J0 = d[..., 0] / (v * l)
            J1 = d[..., 1] / (v * l)
            J2 = d[..., 2] / (v * l)

            std0 = (J0 * R[..., 0, 0] + J1 * R[..., 1, 0] +
                    J2 * R[..., 2, 0] + R[..., 3, 0]) * s[..., 0]
            std1 = (J0 * R[..., 0, 1] + J1 * R[..., 1, 1] +
                    J2 * R[..., 2, 1] + R[..., 3, 1]) * s[..., 1]
            std2 = (J0 * R[..., 0, 2] + J1 * R[..., 1, 2] +
                    J2 * R[..., 2, 2] + R[..., 3, 2]) * s[..., 2]
            std3 = (J0 * R[..., 0, 3] + J1 * R[..., 1, 3] +
                    J2 * R[..., 2, 3] + R[..., 3, 3]) * s[..., 3]

            projected_mean = torch.stack((t + (l / v), f), dim=-1)  # (K, 2)
            projected_var = std0 ** 2 + std1 ** 2 + std2 ** 2 + std3 ** 2
        elif self.gaussian_version == 4:

            J00 = d[..., 0] / (v * l)
            J01 = d[..., 1] / (v * l)
            J02 = d[..., 2] / (v * l)

            JR00 = (J00 * R[..., 0, 0] + J01 * R[..., 1, 0] +
                    J02 * R[..., 2, 0] + R[..., 3, 0])
            JR01 = (J00 * R[..., 0, 1] + J01 * R[..., 1, 1] +
                    J02 * R[..., 2, 1] + R[..., 3, 1])
            JR02 = (J00 * R[..., 0, 2] + J01 * R[..., 1, 2] +
                    J02 * R[..., 2, 2] + R[..., 3, 2])
            JR03 = (J00 * R[..., 0, 3] + J01 * R[..., 1, 3] +
                    J02 * R[..., 2, 3] + R[..., 3, 3])
            JR04 = (J00 * R[..., 0, 4] + J01 * R[..., 1, 4] +
                    J02 * R[..., 2, 4] + R[..., 3, 4])

            V00 = ((JR00 * s[..., 0]) ** 2 + (JR01 * s[..., 1]) ** 2 +
                   (JR02 * s[..., 2]) ** 2 + (JR03 * s[..., 3]) ** 2 + (JR04 * s[..., 4]) ** 2)
            Cov = (JR00 * R[..., 4, 0] * s[..., 0] ** 2 + JR01 * R[..., 4, 1] * s[..., 1] ** 2 +
                   JR02 * R[..., 4, 2] * s[..., 2] ** 2 + JR03 * R[..., 4, 3] * s[..., 3] ** 2 + JR04 * R[..., 4, 4] * s[..., 4] ** 2)
            V11 = ((R[..., 4, 0] * s[..., 0]) ** 2 + (R[..., 4, 1] * s[..., 1]) ** 2 +
                   (R[..., 4, 2] * s[..., 2]) ** 2 + (R[..., 4, 3] * s[..., 3]) ** 2 + (R[..., 4, 4] * s[..., 4]) ** 2)

            projected_mean = torch.stack((t + (l / v), f), dim=-1)  # (K, 2)
            projected_var = torch.stack((V00, V11, Cov), dim=-1)  # (K, 3)
        else:
            raise NotImplementedError

        return projected_mean, projected_var, d, b_idx, n_idx

    def render_signal_at_points(self, query_points):
        """
        Calculates the final signal at a set of 3D query points.

        Args:
            query_points (torch.Tensor): (B, 3) tensor of normalized query points.

        Returns:
            final_signal (torch.Tensor): (B, seq_len) tensor of rendered audio signals in time domain.
        """
        B = query_points.shape[0]
        N = self.get_mean.shape[0]
        L = self.seq_len
        T = self.t_len
        M = self.f_len
        final_stft = torch.zeros(
            B, M, T, device=self.device, dtype=torch.complex64)
        final_signal = torch.zeros(B, L, device=self.device)
        if N == 0:
            return final_signal

        # Project gaussians
        projected_mean, projected_var, d, b_idx, n_idx = self.project(
            query_points)

        if projected_mean.shape[0] == 0:
            return final_signal

        # Time bin
        t_pts = torch.linspace(-1, 1, T+1, device=self.device)[:T]  # (t_len)
        t_bin = t_pts[1] - t_pts[0]

        # Frequency bin
        f_pts = torch.linspace(-1, 1, M+1, device=self.device)[:M]  # (f_len)
        f_bin = f_pts[1] - f_pts[0]

        # Check gaussian overlay (3*sigma box) per bin
        with torch.no_grad():
            if self.gaussian_version == 2:
                std = torch.sqrt(projected_var)
                lower_bound = (projected_mean[..., 0] - 3 * std).unsqueeze(-1)
                upper_bound = (projected_mean[..., 0] + 3 * std).unsqueeze(-1)
                time_mask = torch.min(upper_bound, t_pts + t_bin / 2) - \
                    torch.max(lower_bound, t_pts - t_bin / 2) > 0

                if not time_mask.any():
                    return final_signal

                idx, t_idx = time_mask.nonzero(as_tuple=True)
                f_idx = ((projected_mean[idx, 1] + 1) * M / 2).int()

            elif self.gaussian_version == 4:
                V00 = projected_var[:, 0]
                V11 = projected_var[:, 1]
                Cov = projected_var[:, 2]
                det = V00 * V11 - Cov ** 2
                mid = (V00 + V11) / 2
                sqrt_D = torch.sqrt(torch.clamp(mid ** 2 - det, min=0.1))
                radius = 3 * torch.sqrt(torch.max(mid + sqrt_D,  mid - sqrt_D))

                lower_bound_t = (projected_mean[..., 0] - radius).unsqueeze(-1)
                upper_bound_t = (projected_mean[..., 0] + radius).unsqueeze(-1)
                time_mask = torch.min(upper_bound_t, t_pts + t_bin / 2) - \
                    torch.max(lower_bound_t, t_pts - t_bin / 2) > 0

                lower_bound_f = (projected_mean[..., 1] - radius).unsqueeze(-1)
                upper_bound_f = (projected_mean[..., 1] + radius).unsqueeze(-1)
                freq_mask = torch.min(upper_bound_f, f_pts + f_bin / 2) - \
                    torch.max(lower_bound_f, f_pts - f_bin / 2) > 0

                combined_mask = time_mask.unsqueeze(1) & freq_mask.unsqueeze(2)

                if not combined_mask.any():
                    return final_signal

                idx, f_idx, t_idx = combined_mask.nonzero(as_tuple=True)

            else:
                raise NotImplementedError

        # Apply bin culling
        t_pts = t_pts[t_idx]  # (K)
        f_pts = f_pts[f_idx]  # (K)
        b_idx = b_idx[idx]  # (K)
        n_idx = n_idx[idx]  # (K)
        projected_mean = projected_mean[idx]  # (K, 2)
        projected_var = projected_var[idx]  # (K)
        opacity = self.get_opacity[n_idx].squeeze(-1)  # (K)
        d = d[idx]  # (K, 3)

        # Calculate gaussian power
        if self.gaussian_version == 2:
            power = torch.exp(-0.5 * (t_pts - projected_mean[..., 0])
                              ** 2 / projected_var)  # (K)
        elif self.gaussian_version == 4:
            V00 = projected_var[:, 0]
            V11 = projected_var[:, 1]
            Cov = projected_var[:, 2]
            det = V00 * V11 - Cov ** 2
            dt = t_pts - projected_mean[..., 0]
            df = f_pts - projected_mean[..., 1]
            power = torch.exp(-0.5 *
                              (V11 * dt ** 2 + V00 * df ** 2 - 2 * Cov * dt * df) / det)
        else:
            raise NotImplementedError

        # Calculate alpha
        alpha = opacity * power  # (K)

        # Alpha culling
        with torch.no_grad():
            alpha_mask = alpha > 1/255
            if not alpha_mask.any():
                return final_signal

        alpha = alpha[alpha_mask]
        b_idx = b_idx[alpha_mask]
        n_idx = n_idx[alpha_mask]
        t_idx = t_idx[alpha_mask]
        f_idx = f_idx[alpha_mask]
        d = d[alpha_mask]

        # Index reference
        with torch.no_grad():
            bft_idx = b_idx * (M * T) + f_idx * T + t_idx

            # Sort in f, t order
            sort_idx = torch.argsort(bft_idx, stable=True)
            bft_idx = bft_idx[sort_idx]

            _, counts = torch.unique_consecutive(bft_idx, return_counts=True)

            idx_starts = F.pad(torch.cumsum(counts, dim=0)[:-1], (1, 0))
            inv_starts = torch.repeat_interleave(idx_starts, counts)

        # Sort all relevant tensors
        alpha = alpha[sort_idx]
        n_idx = n_idx[sort_idx]
        d = d[sort_idx]

        # Calculate log transmittance
        log_alpha = torch.log(1.0 - alpha + 1e-10)
        log_transmittance = torch.cumsum(log_alpha, dim=0)
        log_transmittance = log_transmittance - log_transmittance[inv_starts]

        # Transmittance culling
        with torch.no_grad():
            transmittance_mask = log_transmittance > torch.tensor(
                0.0001, device=self.device).log()
            if not transmittance_mask.any():
                return final_signal

        # Apply transmittance culling
        alpha = alpha[transmittance_mask]
        n_idx = n_idx[transmittance_mask]
        bft_idx = bft_idx[transmittance_mask]
        log_transmittance = log_transmittance[transmittance_mask]
        d = d[transmittance_mask]

        decay = 1/torch.norm(d, dim=-1)
        sh = self.eval_features(self.get_features[n_idx], d)
        transmittance = torch.exp(log_transmittance)

        # Calculate final contribution
        contribution = decay * sh * transmittance * alpha

        # Accumulate contributions to the final STFT
        final_stft.view(-1).index_add_(0, bft_idx, contribution)

        # Inverse STFT to time domain
        final_signal = torch.istft(
            final_stft, n_fft=self.config.audio.n_fft, hop_length=self.config.audio.hop_length,
            window=self.window, length=self.seq_len)

        # Archive update filter for densification
        with torch.no_grad():
            update_filter = torch.zeros(
                self.get_mean.shape[0], dtype=torch.bool, device=self.device)
            update_filter[torch.unique(n_idx)] = True
            self.update_filter = update_filter

        return final_signal  # (B, seq_len)
