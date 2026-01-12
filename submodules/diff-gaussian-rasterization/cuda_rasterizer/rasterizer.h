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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static std::tuple<int, int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			const int P, int D, int M, int W, int H,
			const float* micpos,
			const float* means5D,
			const float* shs,
			const float* opacities,
			const float* scales,
			const float* rotations,
			const float scale_modifier,
			int* radii,
			float* out_stft,
			float antialiasing,
			int gaussian_version,
			float speed,
			float cull_distance,
			float sh_clamping_threshold,
			int seq_len,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R, int B, int W, int H,
			const float* micpos,
			const float* means5D,
			const float* shs,
			const float* opacities,
			const float* scales,
			const float* rotations,
			const float scale_modifier,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			char* sample_buffer,
			const float* dL_dstft,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dphasor,
			float* dL_ddistance,
			float* dL_dmean5D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float antialiasing,
			int gaussian_version,
			float speed,
			float cull_distance,
			float sh_clamping_threshold,
			const int seq_len,
			bool debug);
	};
};

#endif