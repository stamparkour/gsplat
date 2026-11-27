// sh -= learning_rate * sh_grads, one thread per float. Plain SGD with no
// clip, the host passes a smaller rate than apply_gradient gets (the 3DGS
// reference uses 1/20 of the base color's rate for these).

RWStructuredBuffer<float> sh;
StructuredBuffer<float> sh_grads;

struct settings_t {
    float learning_rate;
    int count; // floats in sh
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    int i = int(threadId.x);
    if (i >= settings_buffer[0].count) {
        return;
    }
    sh[i] -= settings_buffer[0].learning_rate * sh_grads[i];
}
