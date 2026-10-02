#ifndef SPLAT_MATH_H
#define SPLAT_MATH_H

// Math shared by the forward and backward kernels. Equations and section
// numbers follow docs/backward.pdf. Plain functions, no buffers, so every
// kernel computes the exact same thing.
//
// Parameters are stored the way gaussian.h has them: plain scale (not log),
// opacity in color.a (not logit), plain rgb (not SH), quaternion (x, y, z, w).

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN = 1.0 / 255.0;
static const float T_MIN = 1e-4;
static const float DILATION = 0.3;

// Raster backward's per-gaussian sums, 9 floats each:
// dL/dcolor (3), dL/dopacity (1), dL/dmean2d (2), dL/dconic xx, xy, yy (3)
static const int RASTER_GRAD_STRIDE = 9;

struct camera_t {
    float4x4 world_to_camera; // same matrix as project_gaussians' camera_transform
    float4 intrinsics;        // fx, fy, cx, cy, in pixels
    float4 limits;            // limx, limy = 1.3 tan(fov/2), near plane, unused
};

// What the raster forward saves per pixel for the raster backward.
struct pixel_state_t {
    float T_final;
    int stop; // how far down the sorted list the forward got
};

float3x3 camera_rotation(camera_t cam) {
    return float3x3(cam.world_to_camera[0].xyz, cam.world_to_camera[1].xyz, cam.world_to_camera[2].xyz);
}

float3 to_camera(camera_t cam, float3 mu) {
    return mul(cam.world_to_camera, float4(mu, 1.0)).xyz;
}

float3x3 diag3(float3 v) {
    return float3x3(v.x, 0, 0, 0, v.y, 0, 0, 0, v.z);
}

// q = (x, y, z, w), w is the real part, already normalized
float3x3 rotation_from_quat(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));
}

// Sigma = R diag(s^2) R^T
float3x3 covariance3d(float3x3 R, float3 s) {
    return mul(R, mul(diag3(s * s), transpose(R)));
}

// t clamped to 1.3x the field of view, only used inside J (section 4)
float2 clamped_xy(camera_t cam, float3 t) {
    return float2(clamp(t.x / t.z, -cam.limits.x, cam.limits.x) * t.z,
                  clamp(t.y / t.z, -cam.limits.y, cam.limits.y) * t.z);
}

float2x3 projection_jacobian(camera_t cam, float3 t) {
    float fx = cam.intrinsics.x, fy = cam.intrinsics.y;
    float2 tc = clamped_xy(cam, t);
    return float2x3(
        fx / t.z, 0, -fx * tc.x / (t.z * t.z),
        0, fy / t.z, -fy * tc.y / (t.z * t.z));
}

// Sigma' = J W Sigma W^T J^T + 0.3 I
float2x2 covariance2d(float2x3 J, float3x3 W, float3x3 Sigma) {
    float2x3 M = mul(J, W);
    float2x2 S2 = mul(M, mul(Sigma, transpose(M)));
    S2[0][0] += DILATION;
    S2[1][1] += DILATION;
    return S2;
}

float2x2 inverse2x2(float2x2 A) {
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    return float2x2(A[1][1], -A[0][1], -A[1][0], A[0][0]) / det;
}

// screen center in pixels
float2 project_mean(camera_t cam, float3 t) {
    return float2(cam.intrinsics.x * t.x / t.z + cam.intrinsics.z,
                  cam.intrinsics.y * t.y / t.z + cam.intrinsics.w);
}

// G = exp(-1/2 d^T Q d), d = p - m
float falloff(float2 d, float2x2 Q) {
    return exp(-0.5 * dot(d, mul(Q, d)));
}

struct gaussian_grad_t {
    float3 mean;
    float4 quaternion; // (x, y, z, w)
    float3 scale;
};

// Sections 3 (inverse step) through 6 for one gaussian. dm and dQ are the
// raster backward's sums for it: dL/dmean2d and dL/dconic (full symmetric).
gaussian_grad_t project_backward_math(camera_t cam, float3 mu, float4 q, float3 s, float2 dm, float2x2 dQ) {
    float fx = cam.intrinsics.x, fy = cam.intrinsics.y;
    float3x3 W = camera_rotation(cam);
    float3 t = to_camera(cam, mu);

    float qn = length(q);
    float4 qh = q / qn;
    float3x3 R = rotation_from_quat(qh);
    float3x3 Sigma = covariance3d(R, s);
    float2x3 J = projection_jacobian(cam, t);
    float2x3 M = mul(J, W);
    float2x2 Q = inverse2x2(covariance2d(J, W, Sigma));

    // section 3: through the inverse
    float2x2 V = -mul(Q, mul(dQ, Q));

    // section 4
    float3x3 U = mul(transpose(M), mul(V, M));
    float2x3 D = 2 * mul(V, mul(J, mul(W, mul(Sigma, transpose(W)))));
    float kx = abs(t.x / t.z) <= cam.limits.x ? 1.0 : 0.0;
    float ky = abs(t.y / t.z) <= cam.limits.y ? 1.0 : 0.0;
    float2 tc = clamped_xy(cam, t);
    float tz2 = t.z * t.z, tz3 = tz2 * t.z;
    float3 dt = float3(
        -fx / tz2 * D[0][2] * kx,
        -fy / tz2 * D[1][2] * ky,
        -fx / tz2 * D[0][0] - fy / tz2 * D[1][1]
            + (1 + kx) * fx * tc.x / tz3 * D[0][2]
            + (1 + ky) * fy * tc.y / tz3 * D[1][2]);

    // section 5
    dt += float3(fx / t.z * dm.x, fy / t.z * dm.y, -(fx * t.x * dm.x + fy * t.y * dm.y) / tz2);

    gaussian_grad_t g;
    g.mean = mul(transpose(W), dt);

    // section 6, for plain scale: dL/ds = dL/dl / s = 2 s (R^T U R)_kk
    float3x3 P = mul(transpose(R), mul(U, R));
    g.scale = 2 * s * float3(P[0][0], P[1][1], P[2][2]);

    float3x3 Gm = 2 * mul(U, mul(R, diag3(s * s)));
    float x = qh.x, y = qh.y, z = qh.z, w = qh.w;
    float dw = 2 * (z * (Gm[1][0] - Gm[0][1]) + y * (Gm[0][2] - Gm[2][0]) + x * (Gm[2][1] - Gm[1][2]));
    float dx = 2 * (y * (Gm[0][1] + Gm[1][0]) + z * (Gm[0][2] + Gm[2][0]) + w * (Gm[2][1] - Gm[1][2])) - 4 * x * (Gm[1][1] + Gm[2][2]);
    float dy = 2 * (x * (Gm[0][1] + Gm[1][0]) + w * (Gm[0][2] - Gm[2][0]) + z * (Gm[1][2] + Gm[2][1])) - 4 * y * (Gm[0][0] + Gm[2][2]);
    float dz = 2 * (w * (Gm[1][0] - Gm[0][1]) + x * (Gm[0][2] + Gm[2][0]) + y * (Gm[1][2] + Gm[2][1])) - 4 * z * (Gm[0][0] + Gm[1][1]);
    float4 dqh = float4(dx, dy, dz, dw); // matched by letter to (x, y, z, w)
    g.quaternion = (dqh - qh * dot(qh, dqh)) / qn;
    return g;
}

#endif // SPLAT_MATH_H
