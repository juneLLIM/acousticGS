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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char* (size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::function<int* (size_t N)> resizeIntFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return t.contiguous().data_ptr<int>();
	};
	return lambda;
}

std::function<float* (size_t N)> resizeFloatFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return t.contiguous().data_ptr<float>();
	};
	return lambda;
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const bool antialiasing,
	const int gaussian_version,
	const float speed,
	const float cull_distance,
	const float sh_clamping_threshold,
	const bool debug) {
	if(means5D.ndimension() != 2 || means5D.size(1) != 5) {
		AT_ERROR("means5D must have dimensions (num_points, 5)");
	}

	const int P = means5D.size(0);
	const int W = t_length;
	const int H = f_length;

	auto int_opts = means5D.options().dtype(torch::kInt32);
	auto float_opts = means5D.options().dtype(torch::kFloat32);
	auto complex_opts = means5D.options().dtype(torch::kComplexFloat);

	torch::Tensor out_stft = torch::full({W, H}, 0.0, complex_opts);
	torch::Tensor radii = torch::full({P}, 0, means5D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
	std::function<char* (size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char* (size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char* (size_t)> imgFunc = resizeFunctional(imgBuffer);
	std::function<char* (size_t)> sampleFunc = resizeFunctional(sampleBuffer);

	int rendered = 0;
	int num_buckets = 0;
	if(P != 0) {
		int M = 0;
		if(shs.size(0) != 0) {
			M = shs.size(1);
		}

		auto tup = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			sampleFunc,
			P, sh_degree, M, W, H,
			micpos.contiguous().data_ptr<float>(),
			means5D.contiguous().data_ptr<float>(),
			reinterpret_cast<float*>(shs.contiguous().data_ptr<c10::complex<float>>()),
			opacities.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			scale_modifier,
			radii.contiguous().data_ptr<int>(),
			reinterpret_cast<float*>(out_stft.contiguous().data_ptr<c10::complex<float>>()),
			antialiasing,
			gaussian_version,
			speed,
			cull_distance,
			sh_clamping_threshold,
			debug);

		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
	}
	return std::make_tuple(rendered, num_buckets, out_stft, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer);
}

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
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& sampleBuffer,
	const bool antialiasing,
	const int gaussian_version,
	const float speed,
	const float cull_distance,
	const float sh_clamping_threshold,
	const bool debug) {

	const int P = means5D.size(0);
	const int W = dL_dout_stft.size(0);
	const int H = dL_dout_stft.size(1);

	int M = 0;
	if(shs.size(0) != 0) {
		M = shs.size(1);
	}

	torch::Tensor dL_dmeans5D = torch::zeros({P, 5}, means5D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 2}, means5D.options());
	torch::Tensor dL_dphasors = torch::zeros({P, NUM_CHANNELS}, means5D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means5D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means5D.options());
	torch::Tensor dL_ddistance = torch::zeros({P, 1}, means5D.options());

	auto complex_opts = means5D.options().dtype(torch::kComplexFloat);
	torch::Tensor dL_dshs = torch::zeros({P, M}, complex_opts);

	torch::Tensor dL_dscales = torch::zeros({P, 5}, means5D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 10}, means5D.options()); // bivector

	if(P != 0) {
		CudaRasterizer::Rasterizer::backward(
			P, sh_degree, M, R, B, W, H,
			micpos.contiguous().data_ptr<float>(),
			means5D.contiguous().data_ptr<float>(),
			reinterpret_cast<float*>(shs.contiguous().data_ptr<c10::complex<float>>()),
			opacities.contiguous().data_ptr<float>(),
			scales.data_ptr<float>(),
			rotations.data_ptr<float>(),
			scale_modifier,
			radii.contiguous().data_ptr<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
			reinterpret_cast<float*>(dL_dout_stft.contiguous().data_ptr<c10::complex<float>>()),
			dL_dmeans2D.contiguous().data_ptr<float>(),
			dL_dconic.contiguous().data_ptr<float>(),
			dL_dopacity.contiguous().data_ptr<float>(),
			dL_dphasors.contiguous().data_ptr<float>(),
			dL_ddistance.contiguous().data_ptr<float>(),
			dL_dmeans5D.contiguous().data_ptr<float>(),
			reinterpret_cast<float*>(dL_dshs.contiguous().data_ptr<c10::complex<float>>()),
			dL_dscales.contiguous().data_ptr<float>(),
			dL_drotations.contiguous().data_ptr<float>(),
			antialiasing,
			gaussian_version,
			speed,
			cull_distance,
			sh_clamping_threshold,
			debug);
	}

	return std::make_tuple(dL_dmeans5D, dL_dshs, dL_dopacity, dL_dscales, dL_drotations);
}

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
) {
	ADAM::adamUpdate(
		param.contiguous().data_ptr<float>(),
		param_grad.contiguous().data_ptr<float>(),
		exp_avg.contiguous().data_ptr<float>(),
		exp_avg_sq.contiguous().data_ptr<float>(),
		visible.contiguous().data_ptr<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}
