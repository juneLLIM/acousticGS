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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>
#include "auxiliary.h"

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	template <int V, typename RotationModel>
	void preprocess(
		int P, int D, int M,
		const int W, int H,
		const glm::vec3* micpos,
		const float* means5D,
		const float* shs,
		const float* opacities,
		const float* scales,
		const RotationModel* rotations,
		const float scale_modifier,
		float* clamped,
		int* radii,
		float2* means2D,
		float* distances,
		float* phasors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool antialiasing,
		float speed,
		float cull_distance,
		float sh_clamping_threshold);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
		float* sampled_T, float* sampled_ar,
		int W, int H,
		const float2* means2D,
		const float* phasors,
		const float4* conic_opacity,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		float* out_stft,
		float* distances);
}


#endif
