StructuredBuffer<int> buffer0;
RWStructuredBuffer<int> buffer1;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    buffer1[index] = -buffer0[index];
}