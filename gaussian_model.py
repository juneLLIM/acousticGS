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
from utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel(nn.Module):

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

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.active_sh_degree = 0
        self.optimizer_type = config.optimizer.type.lower()
        self.max_sh_degree = config.model.sh_degree
        self._xyz = torch.empty(0)
        self._sh_dc = torch.empty(0)
        self._sh_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def forward(self, network_pts, network_view, network_tx):

        # Batchify the rendering to avoid memory issues
        batch_size = 1024 * 16  # Adjust batch size based on available memory
        all_signals = []
        all_opacities = []
        for i in range(0, network_pts.shape[0], batch_size):
            batch_pts = network_pts[i:i+batch_size]
            signal_batch, opacity_batch = self.render_signal_at_points(
                batch_pts)
            all_signals.append(signal_batch)
            all_opacities.append(opacity_batch)

        signal = torch.cat(all_signals, dim=0)
        opacity = torch.cat(all_opacities, dim=0)
        return opacity, signal

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._sh_dc,
            self._sh_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args):
        (self.active_sh_degree,
         self._xyz,
         self._sh_dc,
         self._sh_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup()
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_sh(self):
        sh_dc = self._sh_dc
        sh_rest = self._sh_rest
        return torch.cat((sh_dc, sh_rest), dim=1)

    @property
    def get_sh_dc(self):
        return self._sh_dc

    @property
    def get_sh_rest(self):
        return self._sh_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_magnitude(self, query_points):
        magnitudes = []
        for query_point in query_points:
            dir_pp = (self.get_xyz - query_point)  # (N, 3)
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            magnitude = eval_sh(self.active_sh_degree,
                                self.get_sh.unsqueeze(1), dir_pp_normalized)
            magnitude = torch.clamp_min(magnitude, 0.0)  # (N, 1)

            magnitudes.append(magnitude)
        magnitudes = torch.cat(magnitudes, dim=1)  # (N, M)
        return magnitudes.transpose(0, 1)  # (M, N)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_random(self, count: int, xyz_min: float, xyz_max: float):
        self.spatial_lr_scale = (xyz_max - xyz_min) / 2.0   # radius

        xyz = torch.rand(count, 3, device="cuda") * \
            (xyz_max - xyz_min) + xyz_min

        # Initialize SH feature for magnitude (single channel)
        # DC component is initialized to 1.0, rest are 0.
        sh = torch.zeros((count, (self.max_sh_degree + 1) ** 2), device="cuda")
        sh[:, 0] = 1.0

        # Initialize with small scales
        scales = torch.ones((count, 3), device="cuda") * 0.01

        # Initialize as identity quaternions
        rots = torch.zeros((count, 4), device="cuda")
        rots[:, 0] = 1

        # Initialize with low opacity
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((count, 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._sh_dc = nn.Parameter(
            sh[:, 0:1].requires_grad_(True))
        self._sh_rest = nn.Parameter(
            sh[:, 1:].requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        print(f"Created {count} random gaussians.")

    def training_setup(self):
        self.percent_dense = self.config.optimizer.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': self.config.optimizer.position_lr_init *
                self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._sh_dc],
                'lr': self.config.optimizer.magnitude_lr, "name": "sh_dc"},
            {'params': [self._sh_rest],
                'lr': self.config.optimizer.magnitude_lr / 20.0, "name": "sh_rest"},
            {'params': [self._opacity],
                'lr': self.config.optimizer.opacity_lr, "name": "opacity"},
            {'params': [self._scaling],
                'lr': self.config.optimizer.scaling_lr, "name": "scaling"},
            {'params': [self._rotation],
                'lr': self.config.optimizer.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default" or self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.config.optimizer.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=self.config.optimizer.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=self.config.optimizer.position_lr_delay_mult,
                                                    max_steps=self.config.optimizer.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('sh_dc')
        for i in range(self._sh_rest.shape[1]):
            l.append('sh_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        sh_dc = self._sh_dc.detach().cpu().numpy()
        sh_rest = self._sh_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4')
                      for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, sh_dc, sh_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp=False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        sh_dc = np.asarray(plydata.elements[0]["sh_dc"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("sh_rest_")]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == (self.max_sh_degree + 1) ** 2 - 1
        sh_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            sh_extra[:, idx] = np.asarray(
                plydata.elements[0][attr_name])

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(
            xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sh_dc = nn.Parameter(torch.tensor(
            sh_dc, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sh_rest = nn.Parameter(torch.tensor(
            sh_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(
            opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(
            scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(
            rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

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

        self._xyz = optimizable_tensors["xyz"]
        self._sh_dc = optimizable_tensors["sh_dc"]
        self._sh_rest = optimizable_tensors["sh_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_sh_dc, new_sh_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
             "sh_dc": new_sh_dc,
             "sh_rest": new_sh_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._sh_dc = optimizable_tensors["sh_dc"]
        self._sh_rest = optimizable_tensors["sh_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
            self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_sh_dc = self._sh_dc[selected_pts_mask].repeat(N, 1)
        new_sh_rest = self._sh_rest[selected_pts_mask].repeat(
            N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_sh_dc, new_sh_rest,
                                   new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(
            grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_sh_dc = self._sh_dc[selected_pts_mask]
        new_sh_rest = self._sh_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_sh_dc, new_sh_rest,
                                   new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(
                prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render_signal_at_points(self, query_points):
        """
        Calculates the contribution of all Gaussians at a set of 3D query points.

        Args:
            query_points (torch.Tensor): (M, 3) tensor of query points.

        Returns:
            A tuple of (signals, opacities) at the query points.
            - signals (torch.Tensor): (M,) tensor of acoustic sh.
            - opacities (torch.Tensor): (M,) tensor of opacities.
        """
        M = query_points.shape[0]
        N = self.get_xyz.shape[0]
        if N == 0:
            return torch.zeros(M, device="cuda"), torch.zeros(M, device="cuda")

        means = self.get_xyz    # (N, 3)
        scales = self.get_scaling   # (N, 3)
        rotations = self.get_rotation  # (N, 4)
        opacities = self.get_opacity  # (N, 1)
        magnitude = self.get_magnitude(query_points)  # (M, N)

        # Build covariance matrices and compute inverse
        R = build_rotation(rotations)  # (N, 3, 3)
        S_inv = torch.diag_embed(1.0 / scales)  # (N, 3, 3)
        L_inv = S_inv @ R.transpose(1, 2)  # (N, 3, 3)

        # Transform query points to the Gaussian's local coordinate system
        diff = query_points.unsqueeze(1) - means.unsqueeze(0)  # (M, N, 3)
        transformed_diff = (L_inv.unsqueeze(
            0) @ diff.unsqueeze(-1)).squeeze(-1)  # (M, N, 3)
        mahalanobis_sq = torch.sum(transformed_diff**2, dim=-1)  # (M, N)

        # Calculate contribution of each Gaussian at each query point
        # Contribution = sh * Opacity * PDF
        gaussians = torch.exp(-0.5 * mahalanobis_sq)  # (M, N)
        contribution = magnitude * \
            opacities.transpose(0, 1) * gaussians  # (M, N)

        # Sum contributions from all Gaussians for each query point
        final_signal = torch.sum(contribution, dim=1)  # (M,)

        # Also compute a combined opacity for alpha compositing
        opacity_contribution = opacities.transpose(0, 1) * gaussians  # (M, N)
        final_opacity = torch.sum(opacity_contribution, dim=1)  # (M,)

        return final_signal, final_opacity

    def reset_invalid_gaussians(self, mask):
        """Resets the parameters of invalid Gaussians."""
        # Replace NaNs with zeros to prevent errors in subsequent operations
        self._xyz[mask] = 0.0
        self._sh_dc[mask] = 0.0
        self._sh_rest[mask] = 0.0
        self._scaling[mask] = 0.0
        self._rotation[mask] = 0.0
        self._opacity[mask] = 0.0
