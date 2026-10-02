#ifndef GAUSSIAN_H
#define GAUSSIAN_H

struct gaussian2d_t {
	int source_index;
	float depth;
	float2x2 covariance_inv;
	float2 mean;
};

struct gaussian_color_t {
	float4 color; // r,g,b,a
	// additional arguments.
};

struct gaussian_t {
	float4x4 covariance_inv; // 3x3 but requires 4x4 alignment
	float4 mean;
	float4 quaternion; // i,j,k,r
	float4 scale; // only uses 3 values
	gaussian_color_t color;
};

#endif // GAUSSIAN_H