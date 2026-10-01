#include "gaussian.h"

RWStructuredBuffer<gaussian_t> buffer_io;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    gaussian_t v = buffer_io[index];
    
    float4 s = v.scale;
    float4x4 scale_2 = float4x4(
        s.x*s.x,0,0,0,
        0,s.y*s.y,0,0,
        0,0,s.z*s.z,0,
        0,0,0,0
    );
    float4 q = v.quaternion;
    float4x4 rot = float4x4(
        1-2*(q.y*q.y+q.z*q.z),0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0
    );

    buffer_io[index].covariance = rot * scale_2 * transpose(rot);
}