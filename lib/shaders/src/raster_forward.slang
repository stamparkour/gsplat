#include "gaussian.h"
#include "splat_math.h"

// Renders the sorted gaussians into an image, one thread per pixel, and saves
// what the raster backward needs per pixel (backward.pdf section 1).

StructuredBuffer<gaussian2d_t> gaussians_2d;   // from project_gaussians: mean in pixels, covariance_inv = Q
StructuredBuffer<gaussian_t> gaussians;        // color.rgb = color, color.a = opacity
StructuredBuffer<int> sorted_index;            // from sort_gaussians, front to back
RWStructuredBuffer<float4> image_out;          // rgb, a unused
RWStructuredBuffer<pixel_state_t> pixel_state; // for the raster backward

struct settings_t {
    float4 background;
    int width;
    int height;
    int gaussian_count;
    float near_plane;
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    settings_t st = settings_buffer[0];
    int index = int(threadId.x);
    if (index >= st.width * st.height) {
        return;
    }
    float2 p = float2(float(index % st.width), float(index / st.width)) + 0.5;

    float T = 1.0;
    float3 C = float3(0.0);
    int stop = st.gaussian_count;
    for (int k = 0; k < st.gaussian_count; k++) {
        gaussian2d_t g2 = gaussians_2d[sorted_index[k]];
        if (g2.depth <= st.near_plane) {
            continue; // behind the camera
        }
        float4 rgba = gaussians[g2.source_index].color.color;
        float alpha = min(ALPHA_MAX, rgba.a * falloff(p - g2.mean, g2.covariance_inv));
        if (alpha < ALPHA_MIN) {
            continue; // too faint to count
        }
        float test_T = T * (1 - alpha);
        if (test_T < T_MIN) {
            stop = k; // pixel is nearly opaque, this gaussian isn't blended
            break;
        }
        C += rgba.rgb * alpha * T;
        T = test_T;
    }

    image_out[index] = float4(C + T * st.background.rgb, 1.0);
    pixel_state_t ps;
    ps.T_final = T;
    ps.stop = stop;
    pixel_state[index] = ps;
}
