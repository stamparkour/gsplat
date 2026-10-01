#ifndef GAUSSIAN_H
#define GAUSSIAN_H

struct gaussian2d_t {
	int source_index;
	int _unused0;
	float2x2 covariance;
	float2x2 mean;
};

struct gaussian_color_t {
	float4 color; // r,g,b,a
	// additional arguments.
};

struct gaussian_t {
	float4x4 covariance; // 3x3 but requires 4x4 alignment
	float4 mean;
	float4 quaternion; // i,j,k,r
	float4 scale; // only uses 3 values
	gaussian_color_t color;
};

#endif // GAUSSIAN_H