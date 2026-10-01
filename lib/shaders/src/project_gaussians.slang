#include "gaussian.h"

StructuredBuffer<gaussian_t> buffer_i;
RWStructuredBuffer<gaussian2d_t> buffer_o;

struct settings_t {
    float4x4 camera_transform;  // world space to camera space
    float4x4 camera_transform_inv;  // camera space to world space
	float4x4 camera_projection; // camera space to clip space
	float4x4 camera_projection_inv; // clip space to camera space
	int buffer_size;
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024,1,1)]
void computeMain(uint3 threadId: SV_DispatchThreadID) {
    uint index = threadId.x;

	if (index >= settings_buffer[0].buffer_size) {
		return;
	}

    buffer_o[index].source_index = index;

    float4 pos = buffer_i[index].mean;
    pos.w = 1.0f;

    pos = mul(settings_buffer[0].camera_transform, pos);
    pos = mul(settings_buffer[0].camera_projection, pos);

    buffer_o[index].mean = pos.xy / pos.w;
	buffer_o[index].depth = pos.w;

    float4x4 cov = buffer_i[index].covariance_inv;

	// E_2^-1 = (W^-1)^T * E^-1 * (W^-1)
    cov = mul(transpose(settings_buffer[0].camera_transform_inv), mul(cov, settings_buffer[0].camera_transform_inv));

    // calculate jacobian
	// jacobian of the inverse function is equal to the inverse of the jacobian of the function
	// jacobian of the inverse of projection transform at point pos.
    float4x4 jacobian_inv = float4x4();
	
}