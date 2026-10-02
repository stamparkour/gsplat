#include "gaussian.h"
#include "splat_math.h"

// Raster backward, one thread per pixel (backward.pdf sections 1 to 3).
// Walks the pixel's list back to front and adds into each gaussian's sums.
// raster_grads must be cleared to 0 before this runs.

StructuredBuffer<gaussian2d_t> gaussians_2d;
StructuredBuffer<gaussian_t> gaussians;
StructuredBuffer<int> sorted_index;
StructuredBuffer<pixel_state_t> pixel_state; // from raster_forward
StructuredBuffer<float4> dL_dC;              // from loss_l1
RWStructuredBuffer<float> raster_grads;      // RASTER_GRAD_STRIDE floats per gaussian

struct settings_t {
    float4 background;
    int width;
    int height;
    int gaussian_count;
    float near_plane;
};

StructuredBuffer<settings_t> settings_buffer;

// Many pixels add into the same gaussian at once, so every add is atomic.
void add(int i, float v) {
    InterlockedAdd(raster_grads[i], v);
}

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

    pixel_state_t ps = pixel_state[index];
    float T = ps.T_final;
    float3 S = st.background.rgb;  // color of everything behind the current gaussian
    float3 gC = dL_dC[index].rgb;

    for (int k = ps.stop - 1; k >= 0; k--) {
        gaussian2d_t g2 = gaussians_2d[sorted_index[k]];
        if (g2.depth <= st.near_plane) {
            continue; // same tests as the forward, so both walk the same list
        }
        int i = g2.source_index;
        float4 rgba = gaussians[i].color.color;
        float2 d = p - g2.mean;
        float G = falloff(d, g2.covariance_inv);
        float alpha = min(ALPHA_MAX, rgba.a * G);
        if (alpha < ALPHA_MIN) {
            continue;
        }

        // section 1
        T = T / (1 - alpha);
        float3 dc = alpha * T * gC;
        float dalpha = T * dot(rgba.rgb - S, gC);
        S = alpha * rgba.rgb + (1 - alpha) * S;

        // section 2 (the 0.99 cap passes the gradient straight through)
        float drho = G * dalpha;
        float dG = rgba.a * dalpha;

        // section 3
        float w = G * dG;
        float2 dm = w * mul(g2.covariance_inv, d);

        int b = i * RASTER_GRAD_STRIDE;
        add(b + 0, dc.x);
        add(b + 1, dc.y);
        add(b + 2, dc.z);
        add(b + 3, drho);
        add(b + 4, dm.x);
        add(b + 5, dm.y);
        add(b + 6, -0.5 * w * d.x * d.x);
        add(b + 7, -0.5 * w * d.x * d.y);
        add(b + 8, -0.5 * w * d.y * d.y);
    }
}
