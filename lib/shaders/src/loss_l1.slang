// L1 loss against the photo, one thread per pixel. Writes dL/dC for the
// raster backward, and each pixel's share of the loss for the CPU to sum.

StructuredBuffer<float4> image;           // from raster_forward
StructuredBuffer<uint> target;            // the photo, RGBA8 packed like data::pixel
RWStructuredBuffer<float4> dL_dC;         // to raster_backward
RWStructuredBuffer<float> loss_per_pixel; // sum these for the loss value

struct settings_t {
    int width;
    int height;
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    settings_t st = settings_buffer[0];
    int n = st.width * st.height;
    int index = int(threadId.x);
    if (index >= n) {
        return;
    }

    uint v = target[index];
    float3 photo = float3(float(v & 0xFF), float((v >> 8) & 0xFF), float((v >> 16) & 0xFF)) / 255.0;
    float3 diff = image[index].rgb - photo;
    float scale = 1.0 / (3.0 * float(n));

    dL_dC[index] = float4(sign(diff) * scale, 0.0);
    loss_per_pixel[index] = (abs(diff.x) + abs(diff.y) + abs(diff.z)) * scale;
}
