#include "gaussian.h"

RWStructuredBuffer<gaussian_t> buffer_io;

struct settings_t {
    int buffer_io_size;
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    if(index >= settings_buffer[0].buffer_io_size) {
        return;
    }
    
    gaussian_t v = buffer_io[index];
    
    float4 s = v.scale;
    float4x4 scale_2 = float4x4(
        1/(s.x*s.x),0,0,0,
        0,1/(s.y*s.y),0,0,
        0,0,1/(s.z*s.z),0,
        0,0,0,0
    );

    float4 q = normalize(v.quaternion);
    v.quaternion = q;

    float4x4 rot = float4x4(
        1-2*(q.y*q.y+q.z*q.z),2*(q.x*q.y-q.z*q.w),2*(q.x*q.z+q.y*q.w),0,
        2*(q.x*q.y+q.z*q.w),1-2*(q.x*q.x+q.z*q.z),2*(q.y*q.z-q.x*q.w),0,
        2*(q.x*q.z-q.y*q.w),2*(q.y*q.z+q.x*q.w),1-2*(q.x*q.x+q.y*q.y),0,
        0,0,0,0
    );

    // E = RS^2R^T
    // E^-1 = RS^-2R^T
    buffer_io[index].covariance_inv = mul(rot, mul(scale_2, transpose(rot)));
}