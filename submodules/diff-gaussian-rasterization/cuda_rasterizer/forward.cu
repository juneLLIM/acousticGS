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

#include <cub/cub.cuh>
#include "forward.h"
#include "auxiliary.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a complex phasor.
__device__ glm::vec2 computeSH(int idx, int deg, int max_coeffs, glm::vec3 micpos, const float* means, const float* shs, float* clamped, float clamping_threshold) {

	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = {means[5 * idx], means[5 * idx + 1], means[5 * idx + 2]};
	glm::vec3 dir = pos - micpos;
	dir = dir / glm::length(dir);

	glm::vec2* sh = ((glm::vec2*)shs) + idx * max_coeffs;
	glm::vec2 result = SH_C0 * sh[0];

	if(deg > 0) {
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if(deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if(deg > 2) {
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}

	// Magnitude clamping
	float magnitude = glm::length(result);
	float scale = 1.0f;
	if(clamping_threshold > 0.0f && magnitude > clamping_threshold) {
		scale = clamping_threshold / magnitude;
		result = result * scale;
	}
	clamped[NUM_CHANNELS * idx + 0] = scale;
	clamped[NUM_CHANNELS * idx + 1] = scale;

	return result;
}



template <int V, typename RotationModel>
__device__ bool project(
	int idx,
	const int W, int H,
	const glm::vec3& micpos,
	const float* means5D,
	const float* scales,
	const RotationModel* rotations,
	const float scale_modifier,
	const float speed,
	const float cull_distance,
	float2& mean2D,
	float& out_dist,
	float3& cov) {

	// Data loading
	const int base = idx * 5;
	const float mx = means5D[base];
	const float my = means5D[base + 1];
	const float mz = means5D[base + 2];
	const float mt = means5D[base + 3];
	const float mf = means5D[base + 4];

	// Distance calculation
	const float dx = mx - micpos.x;
	const float dy = my - micpos.y;
	const float dz = mz - micpos.z;
	const float dist_sq = dx * dx + dy * dy + dz * dz;

	// Distance culling
	if(dist_sq < 1e-12f) return false;

	// Calulate distance and inverse distance
	const float inv_dist = rsqrtf(dist_sq);
	const float dist = dist_sq * inv_dist;
	out_dist = dist;

	// Projected mean
	float proj_t = mt + dist / speed;
	float proj_f = mf;

	// Convert to pixels
	mean2D.x = (proj_t + 1.0f) * 0.5f * W;
	mean2D.y = (proj_f + 1.0f) * 0.5f * H;

	// Projected covariance
	if constexpr(V == 1) { // [XYZ]
		cov = make_float3(0.0f, 0.0f, 0.0f);
	}
	else if constexpr(V == 2) { // [XYZT]

		// Rotation matrix and scale
		float4 R[4];
		computeRotation(rotations[idx], R);
		const float* S_ptr = scales + idx * 4;

		// Jacobian coefficient
		const float inv_speed = 1.0f / speed;
		const float j_coeff = inv_speed * inv_dist;

		// Jacobian vector (Spatial part)
		const float jx = dx * j_coeff;
		const float jy = dy * j_coeff;
		const float jz = dz * j_coeff;

		// Initialize projected covariance components
		float var_time = 0.0f;

		// Project each column of R onto J
#pragma unroll
		for(int c = 0; c < 4; ++c) {

			float rx = R[c].x;
			float ry = R[c].y;
			float rz = R[c].z;
			float rt = R[c].w;

			float s = S_ptr[c];
			if(c < 3) s *= scale_modifier;

			float proj_time = (jx * rx + jy * ry + jz * rz + rt) * s;

			var_time += proj_time * proj_time;
		}
		cov = make_float3(var_time, 0.0f, 0.0f);
	}
	else if constexpr(V == 3) { // [XYZF]

		// Jacobian coefficient
		const float inv_speed = 1.0f / speed;
		const float j_coeff = inv_speed * inv_dist;

		// Jacobian vector (Spatial part)
		const float jx = dx * j_coeff;
		const float jy = dy * j_coeff;
		const float jz = dz * j_coeff;

		// Rotation matrix and scale
		float4 R[4];
		computeRotation(rotations[idx], R);
		const float* S_ptr = scales + idx * 4;

		// Initialize projected covariance components
		float var_time = 0.0f;
		float cov_tf = 0.0f;
		float var_freq = 0.0f;

		// Project each column of R onto J and Frequency axis
#pragma unroll
		for(int c = 0; c < 4; ++c) {
			float rx = R[c].x;
			float ry = R[c].y;
			float rz = R[c].z;
			float rf = R[c].w;

			float s = S_ptr[c];
			if(c < 3) s *= scale_modifier;

			// Projection
			float proj_time = (jx * rx + jy * ry + jz * rz) * s;
			float proj_freq = rf * s;

			var_time += proj_time * proj_time;
			cov_tf += proj_time * proj_freq;
			var_freq += proj_freq * proj_freq;
		}
		cov = make_float3(var_time, cov_tf, var_freq);
	}
	else if constexpr(V == 4) { // [XYZTF]

		// Jacobian coefficient
		const float inv_speed = 1.0f / speed;
		const float j_coeff = inv_speed * inv_dist;

		// Jacobian vector (Spatial part)
		const float jx = dx * j_coeff;
		const float jy = dy * j_coeff;
		const float jz = dz * j_coeff;

		// Rotation matrix and scale
		float R[25];
		computeRotation(rotations[idx], R);
		const float* S_ptr = scales + idx * 5;

		// Initialize projected covariance components
		float var_time = 0.0f;
		float cov_tf = 0.0f;
		float var_freq = 0.0f;

		// Project each column of R onto J and Frequency axis
#pragma unroll
		for(int c = 0; c < 5; ++c) {
			float rx = R[c];
			float ry = R[c + 5];
			float rz = R[c + 10];
			float rt = R[c + 15];
			float rf = R[c + 20];
			float s = S_ptr[c];
			if(c < 3) s *= scale_modifier;

			float proj_time = (jx * rx + jy * ry + jz * rz + rt) * s;
			float proj_freq = rf * s;

			var_time += proj_time * proj_time;
			cov_tf += proj_time * proj_freq;
			var_freq += proj_freq * proj_freq;
		}
		cov = make_float3(var_time, cov_tf, var_freq);
	}

	return true;
}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C, int V, typename RotationModel>
__global__ void preprocessCUDA(
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
	bool* on_ray,
	const dim3 grid,
	uint32_t* tiles_touched,
	float antialiasing,
	float speed,
	float cull_distance,
	float sh_clamping_threshold,
	const glm::vec3* source_pos,
	float ray_threshold) {
	auto idx = cg::this_grid().thread_rank();
	if(idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float2 mean2D;
	float distance;
	float3 cov;

	// Calculate on_ray status
	glm::vec3 pos = {means5D[5 * idx], means5D[5 * idx + 1], means5D[5 * idx + 2]};
	glm::vec3 source = *source_pos;
	glm::vec3 listener = *micpos;
	glm::vec3 ray_dir = listener - source;
	float len_sl = glm::length(ray_dir);
	if(len_sl > 1e-6f) ray_dir = ray_dir / len_sl;

	float v = glm::dot(pos - source, ray_dir);
	glm::vec3 proj = source + v * ray_dir;
	float dist_to_ray = glm::length(pos - proj);

	bool is_on_ray = (dist_to_ray < ray_threshold && v > 0.0f && v < len_sl);
	on_ray[idx] = is_on_ray;

	bool valid = project<V, RotationModel>(idx, W, H, *micpos, means5D, scales, rotations, scale_modifier, speed, cull_distance, mean2D, distance, cov);

	if(!valid) return;

	if(V == 4) { // TODO: Only implementing V=4 for now
		// Compute Conic
		float det = cov.x * cov.z - cov.y * cov.y;
		float h_convolution_scaling = 1.0f;

		if(antialiasing > 0.0f) {
			float h_var = antialiasing;
			cov.x += h_var;
			cov.z += h_var;
			const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
			h_convolution_scaling = sqrt(max(0.25f, det / det_cov_plus_h_cov)); // max for numerical stability (0.000025f in original GS)
			det = det_cov_plus_h_cov;
		}

		// Invert covariance (EWA algorithm)
		if(det <= 1e-6f)
			return;
		float det_inv = 1.f / det;
		float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

		// Bounding box
		float mid = 0.5f * (cov.x + cov.z);
		float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
		float radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

		uint2 rect_min, rect_max;
		getRect(mean2D, radius, rect_min, rect_max, grid);
		if((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
			return;

		// Convert spherical harmonics coefficients to complex phasor.
		glm::vec2 result = computeSH(idx, D, M, *micpos, means5D, shs, clamped, sh_clamping_threshold);
		((glm::vec2*)phasors)[idx] = result;

		// Store results
		distances[idx] = distance;
		radii[idx] = radius;
		means2D[idx] = mean2D;

		// Inverse 2D covariance and opacity neatly pack into one float4
		float opacity = opacities[idx];
		conic_opacity[idx] = {conic.x, conic.y, conic.z, opacity * h_convolution_scaling};
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset, uint32_t* __restrict__ bucket_to_tile,
	float* __restrict__ sampled_T, float* __restrict__ sampled_ar, float* __restrict__ sampled_additive,
	int W, int H,
	const float2* __restrict__ means2D,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	float* __restrict__ out_stft,
	float* __restrict__ out_additive,
	const float* __restrict__ distances,
	float cull_distance,
	const bool* __restrict__ on_ray) {
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = pix.y * W + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// what is the number of buckets before me? what is my offset?
	uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	// let's first quickly also write the bucket-to-tile mapping
	int num_buckets = (toDo + 31) / 32;
	for(int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
		int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
		if(bucket_idx < num_buckets) {
			bucket_to_tile[bbm + bucket_idx] = tile_id;
		}
	}

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_tf[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_dist[BLOCK_SIZE];
	__shared__ bool collected_on_ray[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};
	float C_add[CHANNELS] = {0};

	// Iterate over batches until all done or range is complete
	for(int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if(num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if(range.x + progress < range.y) {
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_tf[block.thread_rank()] = means2D[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_dist[block.thread_rank()] = distances[coll_id];
			collected_on_ray[block.thread_rank()] = on_ray[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for(int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
			// add incoming T value for every 32nd gaussian
			if(j % 32 == 0) {
				sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
				for(int ch = 0; ch < CHANNELS; ++ch) {
					sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
					sampled_additive[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()] = C_add[ch];
				}
				++bbm;
			}

			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 tf = collected_tf[j];
			float2 d = {tf.x - pixf.x, tf.y - pixf.y};
			float4 con_o = collected_conic_opacity[j];
			float decay = 1 / collected_dist[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if(power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, decay * con_o.w * exp(power));
			if(alpha < 1.0f / 255.0f)
				continue;

			if(collected_on_ray[j]) {
				for(int ch = 0; ch < CHANNELS; ch++) {
					float contrib = features[collected_id[j] * CHANNELS + ch] * alpha;
					C[ch] += contrib;
					C_add[ch] += contrib;
				}
			}
			else if(collected_dist[j] < cull_distance) {
				float test_T = T * (1 - alpha);
				if(test_T < 0.0001f) {
					done = true;
					continue;
				}

				// Eq. (3) from 3D Gaussian splatting paper.
				for(int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

				T = test_T;
			}

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if(inside) {
		n_contrib[pix_id] = last_contributor;
		for(int ch = 0; ch < CHANNELS; ch++) {
			out_stft[ch * H * W + pix_id] = C[ch];
			out_additive[ch * H * W + pix_id] = C_add[ch];
		}
	}

	// max reduce the last contributor
	typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
	if(block.thread_rank() == 0) {
		max_contrib[tile_id] = last_contributor;
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
	float* sampled_T, float* sampled_ar, float* sampled_additive,
	int W, int H,
	const float2* means2D,
	const float* phasors,
	const float4* conic_opacity,
	const bool* on_ray,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	float* out_stft,
	float* out_additive,
	float* distances,
	float cull_distance) {
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		per_tile_bucket_offset, bucket_to_tile,
		sampled_T, sampled_ar, sampled_additive,
		W, H,
		means2D,
		phasors,
		conic_opacity,
		n_contrib,
		max_contrib,
		out_stft,
		out_additive,
		distances,
		cull_distance,
		on_ray);
}

template <int V, typename RotationModel>
void FORWARD::preprocess(
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
	bool* on_ray,
	const dim3 grid,
	uint32_t* tiles_touched,
	float antialiasing,
	float speed,
	float cull_distance,
	float sh_clamping_threshold,
	const glm::vec3* source_pos,
	float ray_threshold) {
	preprocessCUDA<NUM_CHANNELS, V, RotationModel> << <(P + 255) / 256, 256 >> > (
		P, D, M, W, H,
		micpos,
		means5D,
		shs,
		opacities,
		scales,
		rotations,
		scale_modifier,
		clamped,
		radii,
		means2D,
		distances,
		phasors,
		conic_opacity,
		on_ray,
		grid,
		tiles_touched,
		antialiasing,
		speed,
		cull_distance,
		sh_clamping_threshold,
		source_pos,
		ray_threshold
		);
}

#define INSTANTIATE(V, RotationModel) \
template void FORWARD::preprocess<V, RotationModel>( \
	int P, int D, int M, \
	const int W, int H, \
	const glm::vec3* micpos, \
	const float* means5D, \
	const float* shs, \
	const float* opacities, \
	const float* scales, \
	const RotationModel* rotations, \
	const float scale_modifier, \
	float* clamped, \
	int* radii, \
	float2* means2D, \
	float* distances, \
	float* phasors, \
	float4* conic_opacity, \
	bool* on_ray, \
	const dim3 grid, \
	uint32_t* tiles_touched, \
	float antialiasing, \
	float speed, \
	float cull_distance, \
	float sh_clamping_threshold, \
	const glm::vec3* source_pos, \
	float ray_threshold)

INSTANTIATE(1, NoRotation);
INSTANTIATE(2, DoubleQuaternion);
INSTANTIATE(3, DoubleQuaternion);
INSTANTIATE(4, Bivector5D);

#undef INSTANTIATE