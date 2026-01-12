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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		int W, int H, int R, int B,
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_bucket_tile_offset,
		const uint32_t* bucket_to_tile,
		const float* sampled_T, const float* sampled_ar,
		const float2* means2D,
		const float4* conic_opacity,
		const float* phasors,
		const float* distances,
		const uint32_t* n_contrib,
		const uint32_t* max_contrib,
		const float* pixel_phasors,
		const float* dL_dstft,
		float2* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dphasor,
		float* dL_ddistance,
		const float speed,
		const int seq_len);

	template <int V, typename RotationModel>
	void preprocess(
		int P, int D, int M, int W, int H,
		const glm::vec3* micpos,
		const float* means5D,
		const float* shs,
		const float* opacities,
		const float* scales,
		const RotationModel* rotations,
		const float scale_modifier,
		const float* clamped,
		const int* radii,
		const float2* dL_dmean2D,
		const float* dL_dconics,
		const float* dL_ddistance,
		float* dL_dopacity,
		float* dL_dmeans5D,
		float* dL_dphasor,
		float* dL_dsh,
		float* dL_dscale,
		float* dL_drot,
		float antialiasing,
		float speed,
		float cull_distance,
		float sh_clamping_threshold);
}

#endif
