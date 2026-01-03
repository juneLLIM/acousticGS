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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
 // Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S) {
    return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid) {
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    };
}

__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, dim3 grid) {
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
    };
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix) {
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
    return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix) {
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };
    return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix) {
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
    };
    return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix) {
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
    };
    return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
    float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

    float3 dnormvdv;
    dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
    dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
    dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

    float4 vdv = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
    float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
    float4 dnormvdv;
    dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
    dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
    dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
    dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
    return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

__forceinline__ __device__ void matMul5x5(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    float temp[25];
#pragma unroll
    for(int i = 0; i < 5; ++i) {
#pragma unroll
        for(int j = 0; j < 5; ++j) {
            float sum = 0.0f;
#pragma unroll
            for(int k = 0; k < 5; ++k) {
                sum += A[i * 5 + k] * B[k * 5 + j];
            }
            temp[i * 5 + j] = sum;
        }
    }
#pragma unroll
    for(int i = 0; i < 25; ++i) { C[i] = temp[i]; }
}
// Backward Helper: C += A * B^T * scale
__forceinline__ __device__ void matMulAdd_TB(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, float scale) {
    float temp[25];
#pragma unroll
    for(int i = 0; i < 5; ++i) {
#pragma unroll
        for(int j = 0; j < 5; ++j) {
            float sum = 0.0f;
#pragma unroll
            for(int k = 0; k < 5; ++k) {
                sum += A[i * 5 + k] * B[j * 5 + k];
            }
            temp[i * 5 + j] = sum * scale;
        }
    }
    for(int i = 0; i < 25; ++i) { C[i] += temp[i]; }
}
// Backward Helper: C += A^T * B * scale
__forceinline__ __device__ void matMulAdd_TA(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, float scale) {
    float temp[25];
#pragma unroll
    for(int i = 0; i < 5; ++i) {
#pragma unroll
        for(int j = 0; j < 5; ++j) {
            float sum = 0.0f;
#pragma unroll
            for(int k = 0; k < 5; ++k) {
                sum += A[k * 5 + i] * B[k * 5 + j];
            }
            temp[i * 5 + j] = sum * scale;
        }
    }
    for(int i = 0; i < 25; ++i) { C[i] += temp[i]; }
}

__device__ const float FACTORIAL_INV[5] = {
    1.0f,              // 0! (Placeholder)
    1.0f,              // 1!
    0.5f,              // 2!
    0.16666667f,       // 3!
    0.04166667f        // 4!
};

__forceinline__ __device__ void exp5x5(const float* __restrict__ K, float* __restrict__ E) {

    // Term 0: Identity
#pragma unroll
    for(int i = 0; i < 25; ++i) E[i] = (i % 6 == 0) ? 1.0f : 0.0f;

    // Term holds K^n. Initially K^1.
    float Term[25];
#pragma unroll
    for(int i = 0; i < 25; ++i) Term[i] = K[i];

    // Unified loop for orders 1 to 4
    // E = I + K/1! + K^2/2! + K^3/3! + K^4/4!
#pragma unroll
    for(int n = 1; n <= 4; ++n) {
        float coef = FACTORIAL_INV[n];

        // E += Term * coef
#pragma unroll
        for(int i = 0; i < 25; ++i) {
            E[i] += Term[i] * coef;
        }

        // Prepare next Term (Term *= K) only if needed (n < 4)
        if(n < 4) {
            matMul5x5(Term, K, Term);
        }
    }
}

__forceinline__ __device__ void exp5x5Backward(const float* __restrict__ K, const float* __restrict__ dL_dE, float* __restrict__ dL_dK) {
    // Precompute powers
    float K2[25], K3[25];
    matMul5x5(K, K, K2);      // K2 = K * K
    matMul5x5(K2, K, K3);     // K3 = K2 * K


    // Initialize output
    float accum_grad[25] = {0.0f};
#pragma unroll
    for(int i = 0; i < 25; ++i) dL_dK[i] = 0.0f;

    // Backward loop (n = 4 down to 1)
#pragma unroll
    for(int n = 4; n >= 1; --n) {

        // Accumulate gradient from dL_dE
        float coef = FACTORIAL_INV[n];
#pragma unroll
        for(int i = 0; i < 25; ++i) {
            accum_grad[i] += dL_dE[i] * coef;
        }

        // Compute gradient for K
        if(n > 1) {
            const float* curr_K_pow = (n == 4) ? K3 : ((n == 3) ? K2 : K);
            matMulAdd_TB(accum_grad, curr_K_pow, dL_dK, 1.0f);

            float temp[25] = {0.0f};
            matMulAdd_TA(K, accum_grad, temp, 1.0f);

            // Update accum
#pragma unroll
            for(int i = 0; i < 25; ++i) accum_grad[i] = temp[i];

        }
        else {
#pragma unroll
            for(int i = 0; i < 25; ++i) {
                dL_dK[i] += accum_grad[i];
            }
        }
    }
}

// float4 * float overloading
__forceinline__ __device__ float4 operator*(const float4& v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

struct NoRotation {};
struct DoubleQuaternion { float4 l; float4 r; };
struct Bivector5D { float v[10]; };

__forceinline__ __device__ void computeRotation(const NoRotation& rot, float* __restrict__ /*R*/) {}

__forceinline__ __device__ void computeRotation(const DoubleQuaternion& rot, float4* __restrict__ R) {

    // Load vectorized quaternions
    float4 l = rot.l;
    float4 r = rot.r;

    // Normalize quaternions
    float dot_l = l.x * l.x + l.y * l.y + l.z * l.z + l.w * l.w;
    float dot_r = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;

    float inv_l = rsqrtf(dot_l + 1e-8f);
    float inv_r = rsqrtf(dot_r + 1e-8f);

    l = l * inv_l;
    r = r * inv_r;

    // Precompute product
    float4 t1 = r * l.x; // (ap, aq, ar, as)
    float4 t2 = r * l.y; // (bp, bq, br, bs)
    float4 t3 = r * l.z; // (cp, cq, cr, cs)
    float4 t4 = r * l.w; // (dp, dq, dr, ds)

    // Fill matrix (L * v * R)
    // Row 0
    R[0] = make_float4(
        t1.x - t2.y - t3.z - t4.w,
        -t1.y - t2.x + t3.w - t4.z,
        -t1.z - t2.w - t3.x + t4.y,
        -t1.w + t2.z - t3.y - t4.x
    );

    // Row 1
    R[1] = make_float4(
        t1.y + t2.x - t3.w + t4.z,
        t1.x - t2.y + t3.z + t4.w,
        -t1.w - t2.z + t3.y - t4.x,
        t1.z - t2.w - t3.x - t4.y
    );

    // Row 2
    R[2] = make_float4(
        t1.z + t2.w + t3.x - t4.y,
        t1.w - t2.z + t3.y + t4.x,
        t1.x + t2.y - t3.z - t4.w,
        -t1.y + t2.x - t3.w + t4.z
    );

    // Row 3
    R[3] = make_float4(
        t1.w - t2.z + t3.y + t4.x,
        -t1.z + t2.w + t3.x + t4.y,
        t1.y + t2.x + t3.w + t4.z,
        t1.x + t2.y - t3.z - t4.w
    );
}

__forceinline__ __device__ void computeRotation(const Bivector5D& rot, float* __restrict__ R) {

    // Construct skew-symmetric matrix K
    float K[25];
    const float* p = rot.v;

    K[0] = 0.0f; K[6] = 0.0f; K[12] = 0.0f; K[18] = 0.0f; K[24] = 0.0f; // Diagonal
    K[1] = p[0]; K[2] = p[1]; K[3] = p[2]; K[4] = p[3]; // Row 0
    K[5] = -p[0]; K[7] = p[4]; K[8] = p[5]; K[9] = p[6]; // Row 1
    K[10] = -p[1]; K[11] = -p[4]; K[13] = p[7]; K[14] = p[8]; // Row 2
    K[15] = -p[2]; K[16] = -p[5]; K[17] = -p[7]; K[19] = p[9]; // Row 3
    K[20] = -p[3]; K[21] = -p[6]; K[22] = -p[8]; K[23] = -p[9]; // Row 4

    // Compute matrix exponential
    exp5x5(K, R);
}

__forceinline__ __device__ void computeRotationBackward(const NoRotation& rot, const float* __restrict__ /*dL_dR*/, float* __restrict__ /*dL_dparams*/) {}

__forceinline__ __device__ void computeRotationBackward(const DoubleQuaternion& rot, const float4* __restrict__ dL_dR, float4* __restrict__ dL_drot) {

    // Recompute forward
    float4 l = rot.l;
    float4 r = rot.r;

    // Normalize quaternions
    float dot_l = l.x * l.x + l.y * l.y + l.z * l.z + l.w * l.w;
    float dot_r = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;

    float inv_l = rsqrtf(dot_l + 1e-8f);
    float inv_r = rsqrtf(dot_r + 1e-8f);

    l = l * inv_l;
    r = r * inv_r;

    // Define coefficients
    // l = a + bi + cj + dk
    float a = l.x, b = l.y, c = l.z, d = l.w;
    // r = p + qi + rj + sk
    float p = r.x, q = r.y, r_val = r.z, s = r.w;

    // Load gradients
    float4 g0 = dL_dR[0]; // Row 0
    float4 g1 = dL_dR[1]; // Row 1
    float4 g2 = dL_dR[2]; // Row 2
    float4 g3 = dL_dR[3]; // Row 3

    // dL/da
    float da = (g0.x * p - g0.y * q - g0.z * r_val - g0.w * s) +
        (g1.x * q + g1.y * p - g1.z * s + g1.w * r_val) +
        (g2.x * r_val + g2.y * s + g2.z * p - g2.w * q) +
        (g3.x * s - g3.y * r_val + g3.z * q + g3.w * p);

    // dL/db
    float db = (-g0.x * q - g0.y * p - g0.z * s + g0.w * r_val) +
        (g1.x * p - g1.y * q - g1.z * r_val - g1.w * s) +
        (g2.x * s - g2.y * r_val + g2.z * q + g2.w * p) +
        (-g3.x * r_val + g3.y * s + g3.z * p + g3.w * q);

    // dL/dc
    float dc = (-g0.x * r_val + g0.y * s - g0.z * p - g0.w * q) +
        (-g1.x * s + g1.y * r_val + g1.z * q - g1.w * p) +
        (g2.x * p + g2.y * q - g2.z * r_val - g2.w * s) +
        (g3.x * q - g3.y * p + g3.z * s - g3.w * r_val);

    // dL/dd
    float dd = (-g0.x * s - g0.y * r_val + g0.z * q - g0.w * p) +
        (g1.x * r_val + g1.y * s - g1.z * p - g1.w * q) +
        (-g2.x * q + g2.y * p - g2.z * s + g2.w * r_val) +
        (g3.x * p + g3.y * q + g3.z * r_val + g3.w * s);

    // dL/dp
    float dp = (g0.x * a - g0.y * b - g0.z * c - g0.w * d) +
        (g1.x * b + g1.y * a - g1.z * d + g1.w * c) +
        (g2.x * c + g2.y * d + g2.z * a - g2.w * b) +
        (g3.x * d - g3.y * c + g3.z * b + g3.w * a);

    // dL/dq
    float dq = (-g0.x * b - g0.y * a + g0.z * d - g0.w * c) +
        (g1.x * a - g1.y * b + g1.z * c - g1.w * d) +
        (-g2.x * d + g2.y * c + g2.z * b - g2.w * a) +
        (g3.x * c + g3.y * d - g3.z * a + g3.w * b);

    // dL/dr
    float dr = (-g0.x * c - g0.y * d - g0.z * a + g0.w * b) +
        (g1.x * d + g1.y * c - g1.z * b + g1.w * a) +
        (g2.x * a - g2.y * b - g2.z * c + g2.w * d) +
        (-g3.x * b - g3.y * a + g3.z * d - g3.w * c);

    // dL/ds
    float ds = (-g0.x * d + g0.y * c - g0.z * b - g0.w * a) +
        (-g1.x * c + g1.y * d - g1.z * a - g1.w * b) +
        (g2.x * b + g2.y * a - g2.z * d - g2.w * c) +
        (g3.x * a + g3.y * b + g3.z * c - g3.w * d);

    // Left Quaternion normalization backward
    float dot_gl = da * a + db * b + dc * c + dd * d;
    float4 dL_dl;
    dL_dl.x = (da - dot_gl * a) * inv_l;
    dL_dl.y = (db - dot_gl * b) * inv_l;
    dL_dl.z = (dc - dot_gl * c) * inv_l;
    dL_dl.w = (dd - dot_gl * d) * inv_l;

    // Right Quaternion normalization backward
    float dot_gr = dp * p + dq * q + dr * r_val + ds * s;
    float4 dL_dr;
    dL_dr.x = (dp - dot_gr * p) * inv_r;
    dL_dr.y = (dq - dot_gr * q) * inv_r;
    dL_dr.z = (dr - dot_gr * r_val) * inv_r;
    dL_dr.w = (ds - dot_gr * s) * inv_r;

    // Save results
    DoubleQuaternion* out_grad = reinterpret_cast<DoubleQuaternion*>(dL_drot);
    out_grad->l = dL_dl;
    out_grad->r = dL_dr;
}

__forceinline__ __device__ void computeRotationBackward(const Bivector5D& rot, const float* __restrict__ dL_dR, float* __restrict__ dL_drot) {

    // Construct skew-symmetric matrix K
    float K[25];
    const float* p = rot.v;

    K[0] = 0.0f; K[6] = 0.0f; K[12] = 0.0f; K[18] = 0.0f; K[24] = 0.0f; // Diagonal
    K[1] = p[0]; K[2] = p[1]; K[3] = p[2]; K[4] = p[3]; // Row 0
    K[5] = -p[0]; K[7] = p[4]; K[8] = p[5]; K[9] = p[6]; // Row 1
    K[10] = -p[1]; K[11] = -p[4]; K[13] = p[7]; K[14] = p[8]; // Row 2
    K[15] = -p[2]; K[16] = -p[5]; K[17] = -p[7]; K[19] = p[9]; // Row 3
    K[20] = -p[3]; K[21] = -p[6]; K[22] = -p[8]; K[23] = -p[9]; // Row 4

    // Compute exp gradient (dL/dK)
    float dL_dK[25];
    exp5x5Backward(K, dL_dR, dL_dK);

    // Map dK back to bivector gradient (dL/drot)
    dL_drot[0] = dL_dK[1] - dL_dK[5];
    dL_drot[1] = dL_dK[2] - dL_dK[10];
    dL_drot[2] = dL_dK[3] - dL_dK[15];
    dL_drot[3] = dL_dK[4] - dL_dK[20];
    dL_drot[4] = dL_dK[7] - dL_dK[11];
    dL_drot[5] = dL_dK[8] - dL_dK[16];
    dL_drot[6] = dL_dK[9] - dL_dK[21];
    dL_drot[7] = dL_dK[13] - dL_dK[17];
    dL_drot[8] = dL_dK[14] - dL_dK[22];
    dL_drot[9] = dL_dK[19] - dL_dK[23];
}

#endif
