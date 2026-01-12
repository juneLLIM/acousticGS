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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }


// Backward pass for conversion of spherical harmonics to Phasor for each Gaussian.
__device__ void computeSH(int idx, int deg, int max_coeffs, glm::vec3 micpos, const float* means, const float* shs, const float* clamped, const glm::vec2* dL_dphasor, float* dL_dmeans, glm::vec2* dL_dshs) {
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = {means[5 * idx], means[5 * idx + 1], means[5 * idx + 2]};
	glm::vec3 dir_orig = pos - micpos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec2* sh = ((glm::vec2*)shs) + idx * max_coeffs;

	glm::vec2 dL_dPhasor = dL_dphasor[idx];

	// Apply the scaling factor stored during forward pass
	float scale = clamped[NUM_CHANNELS * idx];
	dL_dPhasor = dL_dPhasor * scale;

	glm::vec2 dPhasordx(0, 0);
	glm::vec2 dPhasordy(0, 0);
	glm::vec2 dPhasordz(0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec2* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dPhasordsh0 = SH_C0;
	dL_dsh[0] = dPhasordsh0 * dL_dPhasor;

	if(deg > 0) {
		float dPhasordsh1 = -SH_C1 * y;
		float dPhasordsh2 = SH_C1 * z;
		float dPhasordsh3 = -SH_C1 * x;
		dL_dsh[1] = dPhasordsh1 * dL_dPhasor;
		dL_dsh[2] = dPhasordsh2 * dL_dPhasor;
		dL_dsh[3] = dPhasordsh3 * dL_dPhasor;

		dPhasordx = -SH_C1 * sh[3];
		dPhasordy = -SH_C1 * sh[1];
		dPhasordz = SH_C1 * sh[2];

		if(deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dPhasordsh4 = SH_C2[0] * xy;
			float dPhasordsh5 = SH_C2[1] * yz;
			float dPhasordsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dPhasordsh7 = SH_C2[3] * xz;
			float dPhasordsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dPhasordsh4 * dL_dPhasor;
			dL_dsh[5] = dPhasordsh5 * dL_dPhasor;
			dL_dsh[6] = dPhasordsh6 * dL_dPhasor;
			dL_dsh[7] = dPhasordsh7 * dL_dPhasor;
			dL_dsh[8] = dPhasordsh8 * dL_dPhasor;

			dPhasordx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dPhasordy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dPhasordz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if(deg > 2) {
				float dPhasordsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dPhasordsh10 = SH_C3[1] * xy * z;
				float dPhasordsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dPhasordsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dPhasordsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dPhasordsh14 = SH_C3[5] * z * (xx - yy);
				float dPhasordsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dPhasordsh9 * dL_dPhasor;
				dL_dsh[10] = dPhasordsh10 * dL_dPhasor;
				dL_dsh[11] = dPhasordsh11 * dL_dPhasor;
				dL_dsh[12] = dPhasordsh12 * dL_dPhasor;
				dL_dsh[13] = dPhasordsh13 * dL_dPhasor;
				dL_dsh[14] = dPhasordsh14 * dL_dPhasor;
				dL_dsh[15] = dPhasordsh15 * dL_dPhasor;

				dPhasordx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dPhasordy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dPhasordz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dPhasordx, dL_dPhasor), glm::dot(dPhasordy, dL_dPhasor), glm::dot(dPhasordz, dL_dPhasor));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent phasor.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[5 * idx + 0] += dL_dmean.x;
	dL_dmeans[5 * idx + 1] += dL_dmean.y;
	dL_dmeans[5 * idx + 2] += dL_dmean.z;
}

// Backward pass for projection of 5D covariance from scale and rotation parameters.
template <int V, typename RotationModel>
__device__ void project_and_conic(
	int idx,
	const int W, int H,
	const glm::vec3& micpos,
	const float* means5D,
	const float* opacities,
	const float* scales,
	const RotationModel* rotations,
	const float scale_modifier,
	const float speed,
	const float cull_distance,
	float antialiasing,
	const float2* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_ddistance,
	float* dL_dmeans,
	float* dL_dscales,
	float* dL_drotations,
	float* dL_dopacity,
	const bool* on_ray
) {

	// =========================================================
	// Recompute forward state
	// =========================================================

	// Load mean position
	const int base = idx * 5;
	const float mx = means5D[base];
	const float my = means5D[base + 1];
	const float mz = means5D[base + 2];

	// Distance calculation
	const float dx = mx - micpos.x;
	const float dy = my - micpos.y;
	const float dz = mz - micpos.z;
	const float dist_sq = dx * dx + dy * dy + dz * dz;

	// Distance culling
	bool on_ray_val = on_ray[idx];
	if(!on_ray_val && (dist_sq < 1e-12f || dist_sq > cull_distance * cull_distance)) return;

	// Jacobian computation
	const float inv_speed = 1.0f / speed;
	const float inv_dist = rsqrtf(dist_sq);
	const float j_coeff = inv_speed * inv_dist;
	const float jx = dx * j_coeff;
	const float jy = dy * j_coeff;
	const float jz = dz * j_coeff;

	// Initialization
	float dL_djx = 0.0f, dL_djy = 0.0f, dL_djz = 0.0f;
	float3 dL_dC = {0};

	// =========================================================
	// Covariance and conic backpropagation
	// (dL/dconic, dL/ddistance) --> (dL/dj, dL/dscales, dL/drotations, dL/dopacity)
	// =========================================================

	if constexpr(V == 2) { // [XYZT]

		// Recompute rotation matrix and scale
		float4 R[4];
		computeRotation(rotations[idx], R);
		const float* S_ptr = scales + idx * 4;
		float* dL_dS = dL_dscales + idx * 4;
		float4 dL_dR[4] = {0};

#pragma unroll
		for(int c = 0; c < 4; ++c) {
			// Load rotation matrix column
			float rx = R[c].x;
			float ry = R[c].y;
			float rz = R[c].z;
			float rt = R[c].w;

			// Load scale
			float s = S_ptr[c];
			float modifier = (c < 3) ? scale_modifier : 1.0f;
			float s_final = s * modifier;

			// Recompute forward
			float p_time = (jx * rx + jy * ry + jz * rz + rt) * s_final;

			// Chain rule: dL/dp (using dL_dcov directly)
			float dL_dp_time = 2.0f * dL_dC.x * p_time;

			// Scale gradients
			float term_time = (jx * rx + jy * ry + jz * rz + rt);
			dL_dS[c] = dL_dp_time * term_time * modifier;

			// Jacobian gradients
			float common_j = dL_dp_time * s_final;
			dL_djx += common_j * rx;
			dL_djy += common_j * ry;
			dL_djz += common_j * rz;

			// Rotation gradients
			dL_dR[c].x += common_j * jx;
			dL_dR[c].y += common_j * jy;
			dL_dR[c].z += common_j * jz;
			dL_dR[c].w += common_j;
		}
		computeRotationBackward(rotations[idx], dL_dR, reinterpret_cast<float4*>(dL_drotations + idx * 8));
	}
	else if constexpr(V == 3) { // [XYZF]

		// Recompute rotation matrix and scale
		float4 R[4];
		computeRotation(rotations[idx], R);
		const float* S_ptr = scales + idx * 4;
		float* dL_dS = dL_dscales + idx * 4;
		float4 dL_dR[4] = {0};

#pragma unroll
		for(int c = 0; c < 4; ++c) {
			// Load rotation matrix column
			float rx = R[c].x;
			float ry = R[c].y;
			float rz = R[c].z;
			float rf = R[c].w;

			// Load scale
			float s = S_ptr[c];
			float modifier = (c < 3) ? scale_modifier : 1.0f;
			float s_final = s * modifier;

			// Recompute forward
			float p_time = (jx * rx + jy * ry + jz * rz) * s_final;
			float p_freq = rf * s_final;

			// Chain rule: dL/dp (using dL_dcov directly)
			float dL_dp_time = 2.0f * dL_dC.x * p_time + dL_dC.y * p_freq;
			float dL_dp_freq = 2.0f * dL_dC.z * p_freq + dL_dC.y * p_time;

			// Scale gradients
			float term_time = (jx * rx + jy * ry + jz * rz);
			float term_freq = rf;
			dL_dS[c] = (dL_dp_time * term_time + dL_dp_freq * term_freq) * modifier;

			// Jacobian gradients
			float common_j = dL_dp_time * s_final;
			dL_djx += common_j * rx;
			dL_djy += common_j * ry;
			dL_djz += common_j * rz;

			// Rotation gradients
			dL_dR[c].x += common_j * jx;
			dL_dR[c].y += common_j * jy;
			dL_dR[c].z += common_j * jz;
			dL_dR[c].w += dL_dp_freq * s_final;
		}
		computeRotationBackward(rotations[idx], dL_dR, reinterpret_cast<float4*>(dL_drotations + idx * 8));
	}
	else if constexpr(V == 4) { // [XYZTF]

		// Recompute rotation matrix and scale
		float R[25];
		computeRotation(rotations[idx], R);
		const float* S_ptr = scales + idx * 5;
		float* dL_dS = dL_dscales + idx * 5;
		float dL_dR[25] = {0};

		// Recompute covariance
		float3 cov = {0.0f, 0.0f, 0.0f};
		float cache_p_time[5];
		float cache_p_freq[5];

#pragma unroll
		for(int c = 0; c < 5; ++c) {
			float rx = R[c];
			float ry = R[c + 5];
			float rz = R[c + 10];
			float rt = R[c + 15];
			float rf = R[c + 20];

			float s = S_ptr[c];
			if(c < 3) s *= scale_modifier;

			// Forward Computation
			float p_time = (jx * rx + jy * ry + jz * rz + rt) * s;
			float p_freq = rf * s;

			// Cache for Backward pass
			cache_p_time[c] = p_time;
			cache_p_freq[c] = p_freq;

			cov.x += p_time * p_time;
			cov.y += p_time * p_freq;
			cov.z += p_freq * p_freq;
		}

		// Conic Gradient (dL_dconic -> dL_dC)
		{
			float c_xx = cov.x;
			float c_xy = cov.y;
			float c_yy = cov.z;
			float det = c_xx * c_yy - c_xy * c_xy;

			if(det <= 1e-6f) return;

			float d_inside_root = 0.f;

			if(antialiasing > 0.0f) {

				float h_var = antialiasing;

				// Backup values for correct differentiation (fixed from original code)
				const float x = c_xx;
				const float y = c_yy;
				const float z = c_xy;
				const float w = h_var;

				c_xx += h_var;
				c_yy += h_var;

				const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
				const float h_convolution_scaling = sqrt(max(0.25f, det / det_cov_plus_h_cov)); // max for numerical stability (0.000025f in original GS)
				const float dL_dopacity_v = dL_dopacity[idx];
				const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
				dL_dopacity[idx] *= h_convolution_scaling;
				d_inside_root = (det / det_cov_plus_h_cov) <= 0.25f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
				det = det_cov_plus_h_cov;

				// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
				// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
				const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
				const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
				const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
				const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
				dL_dC.x += dL_dx;
				dL_dC.y += dL_dy;
				dL_dC.z += dL_dz;
			}

			float det_inv = 1.0f / det;
			float common_factor = -det_inv * det_inv;

			float4 dL_dconic_val = ((const float4*)dL_dconic)[idx];
			float dL_dConicX = dL_dconic_val.x;
			float dL_dConicY = dL_dconic_val.y;
			float dL_dConicZ = dL_dconic_val.z;

			float dIdy = common_factor * (-2.0f * c_xy);

			dL_dC.x += dL_dConicX * (c_yy * common_factor * c_yy) +
				dL_dConicY * (-c_xy * common_factor * c_yy) +
				dL_dConicZ * (det_inv + c_xx * common_factor * c_yy);

			dL_dC.y += dL_dConicX * (c_yy * dIdy) +
				dL_dConicY * (-det_inv - c_xy * dIdy) +
				dL_dConicZ * (c_xx * dIdy);

			dL_dC.z += dL_dConicX * (det_inv + c_yy * common_factor * c_xx) +
				dL_dConicY * (-c_xy * common_factor * c_xx) +
				dL_dConicZ * (c_xx * common_factor * c_xx);
		}

		// Covariance backpropagation (dL_dC -> dL_dj, dL_dscales, dL_drotations)
#pragma unroll
		for(int c = 0; c < 5; ++c) {

			// Load rotation matrix column
			float rx = R[c];
			float ry = R[c + 5];
			float rz = R[c + 10];
			float rt = R[c + 15];
			float rf = R[c + 20];
			// Load scale
			float s = S_ptr[c];
			float modifier = (c < 3) ? scale_modifier : 1.0f;
			float s_final = s * modifier;

			// Load recomputed forward
			float p_time = cache_p_time[c];
			float p_freq = cache_p_freq[c];

			// Chain Rule: dL/dp
			float dL_dp_time = 2.0f * dL_dC.x * p_time + dL_dC.y * p_freq;
			float dL_dp_freq = 2.0f * dL_dC.z * p_freq + dL_dC.y * p_time;

			// Scale gradients
			float term_time = (jx * rx + jy * ry + jz * rz + rt);
			float term_freq = rf;
			dL_dS[c] += (dL_dp_time * term_time + dL_dp_freq * term_freq) * modifier;

			// Jacobian gradients
			float common_j = dL_dp_time * s_final;
			dL_djx += common_j * rx;
			dL_djy += common_j * ry;
			dL_djz += common_j * rz;

			// Rotation gradients
			dL_dR[c] += common_j * jx;
			dL_dR[c + 5] += common_j * jy;
			dL_dR[c + 10] += common_j * jz;
			dL_dR[c + 15] += common_j;
			dL_dR[c + 20] += dL_dp_freq * s_final;
		}
		computeRotationBackward(rotations[idx], dL_dR, dL_drotations + idx * 10);
	}

	// =========================================================
	// Mean backpropagation (dL_dmean2D -> dL_dmeans)
	// =========================================================

	const float2 dL_dM = dL_dmean2D[idx];
	const float dL_dproj_t = dL_dM.x * (0.5f * W);
	const float dL_dproj_f = dL_dM.y * (0.5f * H);

	float dL_ddist = dL_dproj_t * inv_speed + dL_ddistance[idx];
	const float dist_grad_factor = dL_ddist * inv_dist;

	// Correction for Jacobian dependency on distance: dL/dJ * dJ/dm
	const float dL_dj_dot_d = dL_djx * dx + dL_djy * dy + dL_djz * dz;
	const float j_correction = -(j_coeff / dist_sq) * dL_dj_dot_d;

	dL_dmeans[base + 0] += dx * dist_grad_factor + dL_djx * j_coeff + dx * j_correction;
	dL_dmeans[base + 1] += dy * dist_grad_factor + dL_djy * j_coeff + dy * j_correction;
	dL_dmeans[base + 2] += dz * dist_grad_factor + dL_djz * j_coeff + dz * j_correction;
	dL_dmeans[base + 3] += dL_dproj_t;
	dL_dmeans[base + 4] += dL_dproj_f;

}


template<int C, int V, typename RotationModel>
__global__ void preprocessCUDA(
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
	float* dL_dmean5D,
	float* dL_dphasor,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dopacity,
	const float* dL_dconic,
	const float* dL_ddistance,
	const bool* on_ray,
	float antialiasing,
	float speed,
	float cull_distance) {
	auto idx = cg::this_grid().thread_rank();
	if(idx >= P || !(radii[idx] > 0))
		return;

	// Compute gradient updates due to computing phasors from SHs
	if(shs)
		computeSH(idx, D, M, *micpos, means5D, shs, clamped, (glm::vec2*)dL_dphasor, dL_dmean5D, (glm::vec2*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if(scales) {
		project_and_conic<V, RotationModel>(
			idx, W, H, *micpos, means5D, opacities, scales, rotations, scale_modifier, speed, cull_distance, antialiasing,
			dL_dmean2D, dL_dconic, dL_ddistance, // Inputs
			dL_dmean5D, dL_dscale, dL_drot, dL_dopacity, // Outputs
			on_ray
		);
	}
}


template<uint32_t C>
__global__ void
PerGaussianRenderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int B,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	const uint32_t* __restrict__ bucket_to_tile,
	const float* __restrict__ sampled_T, const float* __restrict__ sampled_ar, const float* __restrict__ sampled_additive,
	const float2* __restrict__ means2D,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ phasors,
	const float* __restrict__ distances,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ max_contrib,
	const float* __restrict__ pixel_phasors,
	const float* __restrict__ pixel_additive,
	const float* __restrict__ dL_dstft,
	float2* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dphasor,
	float* __restrict__ dL_ddistance,
	const bool* __restrict__ on_ray,
	float cull_distance
) {
	// global_bucket_idx = warp_idx
	auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
	bool valid_bucket = global_bucket_idx < (uint32_t)B;
	if(!valid_bucket) return;

	bool valid_splat = false;

	uint32_t tile_id, bbm;
	uint2 range;
	int num_splats_in_tile, bucket_idx_in_tile;
	int splat_idx_in_tile, splat_idx_global;

	tile_id = bucket_to_tile[global_bucket_idx];
	range = ranges[tile_id];
	num_splats_in_tile = range.y - range.x;
	// What is the number of buckets before me? what is my offset?
	bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	bucket_idx_in_tile = global_bucket_idx - bbm;
	splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	splat_idx_global = range.x + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	// if first gaussian in bucket is useless, then others are also useless
	if(bucket_idx_in_tile * 32 >= max_contrib[tile_id]) {
		return;
	}

	// Load Gaussian properties into registers
	int gaussian_idx = 0;
	float2 xy = {0.0f, 0.0f};
	float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
	float c[C] = {0.0f};
	float decay = 0.0f;
	bool additive_only = false;
	if(valid_splat) {
		gaussian_idx = point_list[splat_idx_global];
		xy = means2D[gaussian_idx];
		con_o = conic_opacity[gaussian_idx];
		decay = 1.0f / distances[gaussian_idx];
		bool on_ray_val = on_ray[gaussian_idx];
		if(!on_ray_val && distances[gaussian_idx] >= cull_distance) valid_splat = false;
		additive_only = on_ray_val;
		for(int ch = 0; ch < C; ++ch)
			c[ch] = phasors[gaussian_idx * C + ch];
	}

	// Gradient accumulation variables
	float Register_dL_dmean2D_x = 0.0f;
	float Register_dL_dmean2D_y = 0.0f;
	float Register_dL_dconic2D_x = 0.0f;
	float Register_dL_dconic2D_y = 0.0f;
	float Register_dL_dconic2D_z = 0.0f;
	float Register_dL_dopacity = 0.0f;
	float Register_dL_ddistance = 0.0f;
	float Register_dL_dphasor[C] = {0.0f};

	// tile metadata
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const uint2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

	// values useful for gradient calculation
	float T;
	float last_contributor;
	float ar[C];
	float additive_chk[C];
	float dL_dpixel[C];
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// iterate over all pixels in the tile
	for(int i = 0; i < BLOCK_SIZE + 31; ++i) {
		if(false) {
			int idx = i - my_warp.thread_rank();
			const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
			bool valid_pixel = pix.x < W && pix.y < H;
			const float2 pixf = {(float)pix.x, (float)pix.y};

			if(valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
				if(W <= pix.x || H <= pix.y) continue;
				if(splat_idx_in_tile >= max_contrib[tile_id]) continue;
			}
		}
		// SHUFFLING

		// At this point, T already has my (1 - alpha) multiplied.
		// So pass this ready-made T value to next thread.
		T = my_warp.shfl_up(T, 1);
		last_contributor = my_warp.shfl_up(last_contributor, 1);
		for(int ch = 0; ch < C; ++ch) {
			ar[ch] = my_warp.shfl_up(ar[ch], 1);
			additive_chk[ch] = my_warp.shfl_up(additive_chk[ch], 1);
			dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
		}

		// which pixel index should this thread deal with?
		int idx = i - my_warp.thread_rank();
		const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
		const uint32_t pix_id = W * pix.y + pix.x;
		const float2 pixf = {(float)pix.x, (float)pix.y};
		bool valid_pixel = pix.x < W && pix.y < H;

		// every 32nd thread should read the stored state from memory
		// TODO: perhaps store these things in shared memory?
		if(valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
			T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
			last_contributor = n_contrib[pix_id];
			for(int ch = 0; ch < C; ++ch) {
				additive_chk[ch] = sampled_additive[global_bucket_idx * BLOCK_SIZE * C + ch * BLOCK_SIZE + idx];
				dL_dpixel[ch] = dL_dstft[ch * H * W + pix_id];
				dL_dpixel[ch] = dL_dstft[ch * H * W + pix_id];
			}
		}

		// do work
		if(valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
			if(W <= pix.x || H <= pix.y) continue;

			if(splat_idx_in_tile >= last_contributor) continue;

			// compute blending values
			const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if(power > 0.0f) continue;
			const float G = exp(power);
			const float alpha = min(0.99f, decay * con_o.w * G);
			if(alpha < 1.0f / 255.0f) continue;

			float weight;
			if(additive_only) {
				weight = alpha;
			}
			else {
				weight = alpha * T;
			}

			// add the gradient contribution of this pixel's phasor to the gaussian
			float dL_dalpha = 0.0f;
			for(int ch = 0; ch < C; ++ch) {
				ar[ch] += weight * c[ch]; // TODO: check
				if(additive_only) {
					additive_chk[ch] += weight * c[ch];
				}

				const float& dL_dchannel = dL_dpixel[ch];
				Register_dL_dphasor[ch] += weight * dL_dchannel;
				if(additive_only) {
					dL_dalpha += c[ch] * dL_dchannel;
				}
				else {
					float pixel_final_additive = pixel_additive[ch * H * W + pix_id];
					float background_additive = pixel_final_additive - additive_chk[ch];
					float background_total = -ar[ch];
					float background_transmittance = background_total - background_additive;
					dL_dalpha += ((c[ch] * T) - (1.0f / (1.0f - alpha)) * (background_transmittance)) * dL_dchannel;
				}
			}

			if(!additive_only) {
				T = T * (1.0f - alpha);
			}

			// Helpful reusable temporary variables
			const float dL_dG = decay * con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// accumulate the gradients
			const float tmp_x = dL_dG * dG_ddelx * ddelx_dx;
			Register_dL_dmean2D_x += tmp_x;
			const float tmp_y = dL_dG * dG_ddely * ddely_dy;
			Register_dL_dmean2D_y += tmp_y;

			Register_dL_dconic2D_x += -0.5f * gdx * d.x * dL_dG;
			Register_dL_dconic2D_y += -0.5f * gdx * d.y * dL_dG;
			Register_dL_dconic2D_z += -0.5f * gdy * d.y * dL_dG;
			Register_dL_dopacity += decay * G * dL_dalpha;
			Register_dL_ddistance += (con_o.w * G * dL_dalpha) * -(decay * decay);
		}
	}

	// finally add the gradients using atomics
	if(valid_splat) {
		atomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);
		atomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].x, Register_dL_dconic2D_x);
		atomicAdd(&dL_dconic2D[gaussian_idx].y, Register_dL_dconic2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].z, Register_dL_dconic2D_z);
		atomicAdd(&dL_dopacity[gaussian_idx], Register_dL_dopacity);
		atomicAdd(&dL_ddistance[gaussian_idx], Register_dL_ddistance);
		for(int ch = 0; ch < C; ++ch) {
			atomicAdd(&dL_dphasor[gaussian_idx * C + ch], Register_dL_dphasor[ch]);
		}
	}
}

template <int V, typename RotationModel>
void BACKWARD::preprocess(
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
	const bool* on_ray,
	float antialiasing,
	float speed,
	float cull_distance,
	float sh_clamping_threshold) {

	preprocessCUDA<NUM_CHANNELS, V, RotationModel> << < (P + 255) / 256, 256 >> > (
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
		dL_dmean2D,
		dL_dmeans5D,
		dL_dphasor,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dopacity,
		dL_dconics,
		dL_ddistance,
		on_ray,
		antialiasing,
		speed,
		cull_distance);
}

#define INSTANTIATE(V, RotationModel) \
template void BACKWARD::preprocess<V, RotationModel>( \
	int P, int D, int M, int W, int H, \
	const glm::vec3* micpos, \
	const float* means5D, \
	const float* shs, \
	const float* opacities, \
	const float* scales, \
	const RotationModel* rotations, \
	const float scale_modifier, \
	const float* clamped, \
	const int* radii, \
	const float2* dL_dmean2D, \
	const float* dL_dconics, \
	const float* dL_ddistance, \
	float* dL_dopacity, \
	float* dL_dmeans5D, \
	float* dL_dphasor, \
	float* dL_dsh, \
	float* dL_dscale, \
	float* dL_drot, \
	const bool* on_ray, \
	float antialiasing, \
	float speed, \
	float cull_distance, \
	float sh_clamping_threshold)

INSTANTIATE(1, NoRotation);
INSTANTIATE(2, DoubleQuaternion);
INSTANTIATE(3, DoubleQuaternion);
INSTANTIATE(4, Bivector5D);

#undef INSTANTIATE

void BACKWARD::render(
	int W, int H, int R, int B,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_bucket_tile_offset,
	const uint32_t* bucket_to_tile,
	const float* sampled_T,
	const float* sampled_ar,
	const float* sampled_additive,
	const float2* means2D,
	const float4* conic_opacity,
	const float* phasors,
	const float* distances,
	const uint32_t* n_contrib,
	const uint32_t* max_contrib,
	const float* pixel_phasors,
	const float* pixel_additive,
	const float* dL_dstft,
	float2* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dphasor,
	float* dL_ddistance,
	const bool* on_ray,
	float cull_distance) {
	const int THREADS = 32;
	PerGaussianRenderCUDA<NUM_CHANNELS> << <((B * 32) + THREADS - 1) / THREADS, THREADS >> > (
		ranges,
		point_list,
		W, H, B,
		per_bucket_tile_offset,
		bucket_to_tile,
		sampled_T, sampled_ar, sampled_additive,
		means2D,
		conic_opacity,
		phasors,
		distances,
		n_contrib,
		max_contrib,
		pixel_phasors,
		pixel_additive,
		dL_dstft,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dphasor,
		dL_ddistance,
		on_ray,
		cull_distance
		);
}
