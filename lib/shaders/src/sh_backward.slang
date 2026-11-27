#include "gaussian.h"
#include "splat_math.h"
#include "sh_math.h"

// Backward of sh_forward, one thread per gaussian. Runs after raster_backward
// and before project_backward.
//  - writes the plain rgb back from base_rgb, so the gaussians are exactly
//    what they were before sh_forward
//  - ADDS dL/dsh = Y_k(d) dL/dcolor into sh_grads, summed over images like
//    gradients, so clear sh_grads at the start of every step
//  - ADDS the part of dL/dmean that comes through the view direction into
//    gradients. dL/drgb is dL/dcolor itself, project_backward already adds it.

RWStructuredBuffer<gaussian_t> gaussians;
StructuredBuffer<float4> base_rgb;        // from sh_forward
StructuredBuffer<float> sh;               // SH_STRIDE floats per gaussian
StructuredBuffer<float> raster_grads;     // from raster_backward, dL/dcolor in the first 3
RWStructuredBuffer<float> sh_grads;       // SH_STRIDE floats per gaussian
RWStructuredBuffer<gaussian_t> gradients; // the same buffer project_backward adds into

struct settings_t {
    float4 camera_center; // the same one sh_forward used
    int gaussian_count;
    int degree;
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
    color.rgb = base_rgb[i].rgb;
    gaussians[i].color.color = color;

    float3 v = gaussians[i].mean.xyz - st.camera_center.xyz;
    float len = length(v);
    if (len <= 0) {
        return;
    }
    float3 d = v / len;

    int r = i * RASTER_GRAD_STRIDE;
    float3 dc = float3(raster_grads[r], raster_grads[r + 1], raster_grads[r + 2]);

    float Y[SH_COEFFS];
    float3 G[SH_COEFFS];
    sh_basis(d, Y);
    sh_basis_grad(d, G);

    int b = i * SH_STRIDE;
    float3 dd = float3(0.0);
    for (int k = 0; k < sh_active(st.degree); k++) {
        int j = b + 3 * k;
        sh_grads[j] += Y[k] * dc.x;
        sh_grads[j + 1] += Y[k] * dc.y;
        sh_grads[j + 2] += Y[k] * dc.z;
        dd += G[k] * dot(float3(sh[j], sh[j + 1], sh[j + 2]), dc);
    }

    // through d = v / |v|
    float3 dmu = (dd - d * dot(d, dd)) / len;
    gradients[i].mean += float4(dmu, 0.0);
}
