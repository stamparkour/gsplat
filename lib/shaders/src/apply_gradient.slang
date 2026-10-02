#include "gaussian.h"

RWStructuredBuffer<gaussian_t> buffer_io;
StructuredBuffer<gaussian_t> buffer_gradient;

struct settings_t {
    float learning_rate;
    int buffer_size;
};
StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;

    if(index >= settings_buffer[0].buffer_size) {
        return;
    }

    float l = settings_buffer[0].learning_rate;

    buffer_io[index].quaternion -= buffer_gradient[index].quaternion * l;
    buffer_io[index].scale -= buffer_gradient[index].scale * l;
    buffer_io[index].mean -= buffer_gradient[index].mean * l;
    buffer_io[index].color.color -= buffer_gradient[index].color.color * l;
}