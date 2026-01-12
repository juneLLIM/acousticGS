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

import torch.nn as nn
import torch
from . import _C


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_points,
        means5D,
        shs,
        opacities,
        scales,
        rotations,
        T, M,
        sh_degree,
        speed,
        source_pos,
        config
    ):

        # Invoke C++/CUDA rasterizer
        num_rendered, num_buckets, phasor, out_additive, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(
            query_points,
            means5D,
            shs,
            opacities,
            scales,
            rotations,
            config.rendering.scale_modifier,
            T, M,
            sh_degree,
            config.rendering.antialiasing,
            config.model.gaussian_version,
            speed,
            config.rendering.cull_distance,
            config.rendering.sh_clamping_threshold,
            source_pos,
            config.rendering.ray_threshold,
            config.logging.cuda)

        # Keep relevant tensors for backward
        ctx.config = config
        ctx.speed = speed
        ctx.sh_degree = sh_degree
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.source_pos = source_pos
        ctx.save_for_backward(query_points, means5D, shs, opacities, scales, rotations,
                              out_additive, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)
        return phasor, radii

    @staticmethod
    def backward(ctx, grad_out_phasor, _,):

        # Restore necessary values from context
        config = ctx.config
        speed = ctx.speed
        sh_degree = ctx.sh_degree
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        query_points, means5D, shs, opacities, scales, rotations, out_additive, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # Compute gradients for relevant tensors by invoking backward method
        grad_means5D, grad_shs, grad_opacities, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(
            num_rendered,
            num_buckets,
            sh_degree,
            query_points,
            means5D,
            shs,
            opacities,
            scales,
            rotations,
            config.rendering.scale_modifier,
            radii,
            grad_out_phasor,
            out_additive,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sampleBuffer,
            config.rendering.antialiasing,
            config.model.gaussian_version,
            speed,
            config.rendering.cull_distance,
            config.rendering.sh_clamping_threshold,
            config.logging.cuda)

        grads = (
            None,
            grad_means5D,
            grad_shs,
            grad_opacities,
            grad_scales,
            grad_rotations,
            None, None,
            None,
            None,
            None,
            None
        )

        return grads


class GaussianRasterizer(nn.Module):
    def __init__(self, config, T, M, speed):
        super().__init__()
        self.config = config
        self.T = T
        self.M = M
        self.speed = speed

    def forward(self, query_points, means5D, shs, opacities, scales, rotations, sh_degree, source_pos):

        # Invoke C++/CUDA rasterizing routine

        if query_points.dim() == 2:
            stfts = []
            radiis = []
            for i in range(query_points.shape[0]):
                stft, radii = _RasterizeGaussians.apply(
                    query_points[i],
                    means5D,
                    shs,
                    opacities,
                    scales,
                    rotations,
                    self.T, self.M,
                    sh_degree,
                    self.speed,
                    source_pos,
                    self.config
                )
                stfts.append(stft)
                radiis.append(radii)
            return torch.stack(stfts), torch.stack(radiis)

        return _RasterizeGaussians.apply(
            query_points,
            means5D,
            shs,
            opacities,
            scales,
            rotations,
            self.T, self.M,
            sh_degree,
            self.speed,
            source_pos,
            self.config
        )


class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)

    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(
                    param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(
                    param, memory_format=torch.preserve_format)

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]

            # Handle complex parameters by viewing as float
            # This ensures C++ kernel receives float pointers and correct dimension M
            target_param = param
            target_grad = param.grad
            target_exp_avg = exp_avg
            target_exp_avg_sq = exp_avg_sq

            if param.is_complex():
                target_param = torch.view_as_real(
                    param).view(param.shape[0], -1)
                target_grad = torch.view_as_real(
                    param.grad).view(param.grad.shape[0], -1)
                target_exp_avg = torch.view_as_real(
                    exp_avg).view(exp_avg.shape[0], -1)
                target_exp_avg_sq = torch.view_as_real(
                    exp_avg_sq).view(exp_avg_sq.shape[0], -1)

            M = target_param.numel() // N
            _C.adamUpdate(target_param, target_grad, target_exp_avg, target_exp_avg_sq,
                          visibility, lr, 0.9, 0.999, eps, N, M)
