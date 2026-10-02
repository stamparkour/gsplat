#include "gaussian.h"

StructuredBuffer<gaussian2d_t> buffer_i;
RWStructuredBuffer<int> index_buffer;

struct settings_t {
    int odd_even_state;
	int buffer_size;
};

StructuredBuffer<settings_t> settings_buffer;

[shader("compute")]
[numthreads(1024,1,1)]
void computeMain(uint3 threadId: SV_DispatchThreadID) {
    uint index = threadId.x;

    int i = index * 2 + settings_buffer[0].odd_even_state;
    int j = i + 1;

    if (j >= settings_buffer[0].buffer_size) {
        return;
    }

    int index_i = index_buffer[i];
    int index_j = index_buffer[j];

    if (buffer_i[index_i].depth > buffer_i[index_j].depth) {
        index_buffer[i] = index_j;
        index_buffer[j] = index_i;
    }
}