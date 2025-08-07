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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        # Lie algebra parameters do not need normalization.
        self.rotation_activation = torch.nn.Identity()

    def __init__(self):
        self._xyzt = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyzt_gradient_accum = torch.empty(0)
        self.optimizer = None
        self.setup_functions()

    def capture(self):
        return (
            self._xyzt,
            self._scaling,
            self._rotation,
            self._opacity,
            self.xyzt_gradient_accum,
            self.optimizer.state_dict(),
        )

    def restore(self, model_args, training_args):
        (self._xyzt,
         self._scaling,
         self._rotation,
         self._opacity,
         xyzt_gradient_accum,
         denom,
         opt_dict) = model_args
        self.training_setup(training_args)
        self.xyzt_gradient_accum = xyzt_gradient_accum
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyzt(self):
        return self._xyzt

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_random(self, count, aabb):
        # aabb (Axis-Aligned Bounding Box) is in [min_x, min_y, min_z, max_x, max_y, max_z] format
        aabb_min = torch.tensor(aabb[:3], dtype=torch.float32, device="cuda")
        aabb_max = torch.tensor(aabb[3:], dtype=torch.float32, device="cuda")

        # Randomly generate spatial (xyz) and temporal (t) coordinates
        xyz = torch.rand(count, 3, device="cuda") * \
            (aabb_max - aabb_min) + aabb_min
        # Initialize time between 0 and 1
        t = torch.rand(count, 1, device="cuda")
        xyzt = torch.cat([xyz, t], dim=1)

        # Initialize with small scales
        scales = torch.ones((count, 4), device="cuda") * 0.01

        # Initialize 6 rotation parameters for Lie algebra, starting with zero (no rotation)
        rots = torch.zeros((count, 6), device="cuda")

        # Initialize with low opacity
        opacities = torch.ones((count, 1), device="cuda") * 0.1

        self._xyzt = nn.Parameter(xyzt.requires_grad_(True))
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scales).requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(
            self.inverse_opacity_activation(opacities).requires_grad_(True))

        print(f"Created {count} random gaussians.")

    def training_setup(self, training_args):
        self.xyzt_gradient_accum = torch.zeros(
            (self.get_xyzt.shape[0], 4), device="cuda")

        l = [
            {'params': [self._xyzt],
                'lr': training_args.position_lr_init, "name": "xyzt"},
            {'params': [self._opacity],
                'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling],
                'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation],
                'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyzt_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init,
                                                     lr_final=training_args.position_lr_final,
                                                     lr_delay_mult=training_args.position_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyzt":
                lr = self.xyzt_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 't', 'opacity']
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyzt = self._xyzt.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4')
                      for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyzt.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzt, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        # t 좌표가 없으면 0으로 초기화
        if 't' in plydata.elements[0].properties:
            t = np.asarray(plydata.elements[0]["t"])[..., np.newaxis]
        else:
            t = np.zeros((xyz.shape[0], 1))

        xyzt = np.hstack((xyz, t))

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyzt.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyzt.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyzt = nn.Parameter(torch.tensor(
            xyzt, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(
            opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(
            scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(
            rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(
                    group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]
                                     ] = stored_state if stored_state is not None else {}
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

        self._xyzt = optimizable_tensors["xyzt"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyzt_gradient_accum = self.xyzt_gradient_accum[valid_points_mask]

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

    def add_densification_stats(self, grads):
        self.xyzt_gradient_accum += grads

    def densification_postfix(self, new_xyzt, new_opacities, new_scaling, new_rotation):
        d = {"xyzt": new_xyzt,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyzt = optimizable_tensors["xyzt"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyzt_gradient_accum = torch.cat(
            [self.xyzt_gradient_accum, torch.zeros((new_xyzt.shape[0], 4), device="cuda")])

    def densify_and_clone(self, clone_mask):
        new_xyzt = self._xyzt[clone_mask]
        new_opacities = self._opacity[clone_mask]
        new_scaling = self._scaling[clone_mask]
        new_rotation = self._rotation[clone_mask]
        self.densification_postfix(
            new_xyzt, new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, split_mask, N=2):
        n_init_points = self.get_xyzt.shape[0]

        stds = self.get_scaling[split_mask].repeat(N, 1)
        means = self.get_xyzt[split_mask].repeat(N, 1)

        samples = torch.randn((stds.size(0), 4), device="cuda") * stds
        rots = build_rotation(self._rotation[split_mask]).repeat(N, 1, 1)
        new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + means

        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[split_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[split_mask].repeat(N, 1)
        new_opacity = self._opacity[split_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyzt, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.zeros(
            n_init_points, dtype=torch.bool, device="cuda")
        prune_filter[split_mask] = True
        self.prune_points(prune_filter)

    def densify_and_prune(self, densify_grad_threshold, prune_opacity_threshold, prune_scale_threshold, densify_scale_threshold):
        prune_mask = (self.get_opacity() < prune_opacity_threshold).squeeze()
        if prune_scale_threshold > 0:
            prune_mask = prune_mask | (
                torch.max(self.get_scaling, dim=1).values > prune_scale_threshold).squeeze()
        self.prune_points(prune_mask)

        grad_norms = torch.norm(self.xyzt_gradient_accum, dim=-1)
        grad_norms[grad_norms.isnan()] = 0.0

        if grad_norms.max() > 0:
            normalized_grads = grad_norms / grad_norms.max()
            clone_mask = (normalized_grads >= densify_grad_threshold) & (
                torch.max(self.get_scaling, dim=1).values < densify_scale_threshold)
            self.densify_and_clone(clone_mask)

            grad_norms_after_clone = torch.norm(
                self.xyzt_gradient_accum, dim=-1)
            if grad_norms_after_clone.max() > 0:
                normalized_grads_after_clone = grad_norms_after_clone / grad_norms_after_clone.max()
                split_mask = (normalized_grads_after_clone >= densify_grad_threshold) & (
                    torch.max(self.get_scaling, dim=1).values >= densify_scale_threshold)
                self.densify_and_split(split_mask)

        self.xyzt_gradient_accum = torch.zeros(
            (self.get_xyzt.shape[0], 4), device="cuda")

    def _render_signal_base(self, query_points_4d):
        M = query_points_4d.shape[0]
        N = self.get_xyzt.shape[0]
        if N == 0:
            return torch.zeros(M, device="cuda")

        means = self.get_xyzt
        scales = self.get_scaling
        rotations = self.get_rotation
        opacities = self.get_opacity

        query_points_expanded = query_points_4d.unsqueeze(1)
        means_expanded = means.unsqueeze(0)
        diff = query_points_expanded - means_expanded

        R = build_rotation(rotations)
        S_inv = torch.diag_embed(1.0 / scales)
        L_inv = S_inv @ R.transpose(1, 2)

        transformed_diff = (L_inv.unsqueeze(
            0) @ diff.unsqueeze(-1)).squeeze(-1)
        mahalanobis_sq = torch.sum(transformed_diff**2, dim=-1)

        log_det_sigma = 2 * torch.sum(torch.log(scales), dim=1)

        k = 4
        log_pdf = -0.5 * (k * np.log(2 * np.pi) +
                          log_det_sigma.unsqueeze(0) + mahalanobis_sq)

        gauss_contribution = torch.exp(
            log_pdf) * opacities.squeeze(-1).unsqueeze(0)
        final_signal = torch.sum(gauss_contribution, dim=1)

        return final_signal

    def render_signal_at_t(self, points_3d, t):
        t_tensor = torch.full_like(points_3d[:, :1], fill_value=t)
        points_4d = torch.cat([points_3d, t_tensor], dim=1)
        return self._render_signal_base(points_4d)

    def render_signal_waveform(self, points_3d, t_start, t_end, t_steps):
        M = points_3d.shape[0]

        t_samples = torch.linspace(
            t_start, t_end, t_steps, device=points_3d.device)

        points_3d_expanded = points_3d.unsqueeze(1)
        t_samples_expanded = t_samples.unsqueeze(0).unsqueeze(-1)

        points_3d_grid = points_3d_expanded.expand(M, t_steps, 3)
        t_samples_grid = t_samples_expanded.expand(M, t_steps, 1)

        points_4d_grid = torch.cat([points_3d_grid, t_samples_grid], dim=2)
        points_4d_flat = points_4d_grid.view(-1, 4)
        signal_flat = self._render_signal_base(points_4d_flat)
        waveform = signal_flat.view(M, t_steps)

        return waveform

    def reset_invalid_gaussians(self, mask):
        """Resets the parameters of invalid Gaussians."""
        # Replace NaNs with zeros to prevent errors in subsequent operations
        self._xyzt[mask] = 0.0
        self._scaling[mask] = 0.0
        self._rotation[mask] = 0.0
        self._opacity[mask] = 0.0
