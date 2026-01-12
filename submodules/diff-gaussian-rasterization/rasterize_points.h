/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& micpos,
	const torch::Tensor& means5D,
	const torch::Tensor& shs,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const int t_length,
	const int f_length,
	const int sh_degree,
	const float antialiasing,
	const int gaussian_version,
	const float speed,
	const float cull_distance,
	const float sh_clamping_threshold,
	const torch::Tensor& source_pos,
	const float ray_threshold,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const int R,
	const int B,
	const int sh_degree,
	const torch::Tensor& micpos,
	const torch::Tensor& means5D,
	const torch::Tensor& shs,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& radii,
	const torch::Tensor& dL_dout_stft,
	const torch::Tensor& out_additive,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& sampleBuffer,
	const float antialiasing,
	const int gaussian_version,
	const float speed,
	const float cull_distance,
	const float sh_clamping_threshold,
	const bool debug);

void adamUpdate(
	torch::Tensor& param,
	torch::Tensor& param_grad,
	torch::Tensor& exp_avg,
	torch::Tensor& exp_avg_sq,
	torch::Tensor& visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
);

torch::Tensor
fusedssim(
	float C1,
	float C2,
	torch::Tensor& img1,
	torch::Tensor& img2
);

torch::Tensor
fusedssim_backward(
	float C1,
	float C2,
	torch::Tensor& img1,
	torch::Tensor& img2,
	torch::Tensor& dL_dmap
);
