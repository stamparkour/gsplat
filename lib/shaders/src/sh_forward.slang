#include "gaussian.h"
#include "sh_math.h"

// Adds the view dependent color for one camera to every gaussian's rgb, one
// thread per gaussian. Runs before project_gaussians, so raster_forward and
// raster_backward see the camera's colors without any change to them. The
// plain rgb is saved in base_rgb first, and sh_backward writes it back after
// raster_backward. Only runs while spherical harmonics are on.

RWStructuredBuffer<gaussian_t> gaussians;
StructuredBuffer<float> sh;           // SH_STRIDE floats per gaussian
RWStructuredBuffer<float4> base_rgb;  // the plain rgb, a unused

struct settings_t {
    float4 camera_center; // world space, w unused
    int gaussian_count;
    int degree;           // 1 to 3
    int pad0;
    int pad1;
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    settings_t st = settings_buffer[0];
    int i = int(threadId.x);
    if (i >= st.gaussian_count) {
        return;
    }

    float4 color = gaussians[i].color.color;
    base_rgb[i] = float4(color.rgb, 0.0);

    float3 v = gaussians[i].mean.xyz - st.camera_center.xyz;
    float len = length(v);
    if (len <= 0) {
        return; // sitting on the camera, no direction to look from
    }

    float Y[SH_COEFFS];
    sh_basis(v / len, Y);
    int b = i * SH_STRIDE;
    float3 offset = float3(0.0);
    for (int k = 0; k < sh_active(st.degree); k++) {
        offset += Y[k] * float3(sh[b + 3 * k], sh[b + 3 * k + 1], sh[b + 3 * k + 2]);
    }
    color.rgb += offset;
    gaussians[i].color.color = color;
}
