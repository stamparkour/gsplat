#include "gaussian.h"
#include "splat_math.h"

// Project backward, one thread per gaussian (backward.pdf sections 2 to 6).
// Turns the raster backward's sums into gradients in the same layout as
// gaussian_t, which is what apply_gradient reads.
//
// Adds into gradients instead of overwriting, so it sums over every image run
// since the host last cleared the buffer (docs/shaders.md, "Training loop").
// One thread owns each gaussian, so no atomics.

StructuredBuffer<gaussian_t> gaussians;
StructuredBuffer<float> raster_grads;      // from raster_backward, this image only
RWStructuredBuffer<gaussian_t> gradients;  // to apply_gradient, summed over images

struct settings_t {
    camera_t camera;
    int gaussian_count;
    int pad0;
    int pad1;
    int pad2;
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

    gaussian_t g = gaussians[i];

    float3 t = to_camera(st.camera, g.mean.xyz);
    if (t.z <= st.camera.limits.z) {
        return; // behind this camera, so it adds nothing
    }

    int b = i * RASTER_GRAD_STRIDE;
    float3 dc = float3(raster_grads[b + 0], raster_grads[b + 1], raster_grads[b + 2]);
    float drho = raster_grads[b + 3];
    float2 dm = float2(raster_grads[b + 4], raster_grads[b + 5]);
    float2x2 dQ = float2x2(raster_grads[b + 6], raster_grads[b + 7],
                           raster_grads[b + 7], raster_grads[b + 8]);

    gaussian_grad_t gg = project_backward_math(st.camera, g.mean.xyz, g.quaternion, g.scale.xyz, dm, dQ);
    gradients[i].mean += float4(gg.mean, 0.0);
    gradients[i].quaternion += gg.quaternion;
    gradients[i].scale += float4(gg.scale, 0.0);
    gradients[i].color.color += float4(dc, drho); // plain rgb and opacity, sections 1 and 2
}
