#include "gaussian.h"
#include "splat_math.h"

// Projects each gaussian onto the image. Uses the same functions as
// project_backward (splat_math.h), so the backward differentiates exactly
// this projection (backward.pdf sections 4 and 5).

StructuredBuffer<gaussian_t> buffer_i;
RWStructuredBuffer<gaussian2d_t> buffer_o;

// same settings as project_backward
struct settings_t {
    camera_t camera; // world_to_camera, intrinsics (fx, fy, cx, cy), limits
	int gaussian_count;
	int pad0;
	int pad1;
	int pad2;
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024,1,1)]
void computeMain(uint3 threadId: SV_DispatchThreadID) {
    uint index = threadId.x;

	if (index >= settings_buffer[0].gaussian_count) {
		return;
	}

    camera_t cam = settings_buffer[0].camera;
    gaussian_t g = buffer_i[index];

    // camera space, z forward
    float3 t = to_camera(cam, g.mean.xyz);

    buffer_o[index].source_index = index;
	buffer_o[index].depth = t.z; // raster skips anything at or behind the near plane
    buffer_o[index].covariance_inv = float2x2(1, 0, 0, 1);
    buffer_o[index].mean = float2(0, 0);

    if (t.z <= cam.limits.z) {
        return;
    }

    // Sigma = R S^2 R^T, built from the quaternion and scale directly
    float3x3 R = rotation_from_quat(normalize(g.quaternion));
    float3x3 Sigma = covariance3d(R, g.scale.xyz);

    // 2D covariance J W Sigma W^T J^T + 0.3 I, then inverted. inverting the
    // 2x2 at the end matters: the 2x2 corner of an inverse covariance is a
    // slice through the gaussian, not its projection onto the image.
    float2x2 cov2d = covariance2d(projection_jacobian(cam, t), camera_rotation(cam), Sigma);
    buffer_o[index].covariance_inv = inverse2x2(cov2d);

    // center in pixels
    buffer_o[index].mean = project_mean(cam, t);
}
