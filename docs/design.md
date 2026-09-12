# goal:
- faster performance that original gaussian splatting lib.
- modular design for easy modification
- utilize Vulkan for rendering and training

questions
- do we use AI during development?


components:
- vulkan
- dataset
- rendering
- training
- cli/gui

# Dataset
tool that takes in data and standardizes it.

planned inputs:
- Colmap

output structure:
- per image:
	- camera pose
	- point cloud 2d position
- sparse/dense initializing cloud
```cpp


class image_point {
	const vector2f& position() const;
}

class image_point_map {
public:
	//iterator->first : point_token
	//iterator->second : image_point
	const_iterator find(point_token) const;
	const_iterator begin() const;
	const_iterator end() const;
	const_iterator cbegin() const;
	const_iterator cend() const;
}

class image {
public:
	image_token token() const;
	const file& image_file() const;
	const vulkan::texture2d& image_vulkan() const;
	const matrix3f& camera_pose() const;
	const image_point_map& point_map() const;
};

class image_map {
public:
	//iterator->first : image_token
	//iterator->second : image
	const_iterator find(image_token) const;
	const_iterator begin() const;
	const_iterator end() const;
	const_iterator cbegin() const;
	const_iterator cend() const;
};

class point {
public:
	point_token token() const;
	const vector3f& position() const;
}

class point_map {
public:
	//iterator->first : point_token
	//iterator->second : point
	const_iterator find(point_token) const;
	const_iterator begin() const;
	const_iterator end() const;
	const_iterator cbegin() const;
	const_iterator cend() const;
};

class dataset {
public:
	const image_map& images() const;
	const point_map& points() const;
}

dataset colmap(const std::filesystem::path& colmap_dir);
```





design requirements:
- modular input design
- modular extra data per image.

extra info:
- undistort image https://colmap.github.io/cli.html

# Rendering

takes in a set of 3d gaussians and camera pose, outputs 2d render.

each 3d gaussians contains:
- 3x3 covariance matrix.
- 3d position
- directional color information

camera pose is:
- camera space transformation
- camera projection transformation



requirements:
- ability to run multiple instances of rendering

```cpp
struct gaussian_vulkan {
	vector3f position;
	matrix4f covariance;
	/* color info */
}

void render_gaussians(
	vulkan::render_buffer& out_buffer,
	const vulkan::array1<gaussian_vulkan>& gaussians, 
	const matrix4f& camera_projection, 
	const matrix4f& camera_transform);
```

# Training

iterates over each input image and refines gaussians.

utilizes parts of renderer to train and back propagate.


```cpp
// sorts gaussians by depth after transforming with camera_transform
void sort_gaussians(
	vulkan::array1i& out_buffer,
	const vulkan::array1<gaussian_vulkan>& gaussians, 
	const matrix4f& camera_transform);

struct gaussian2d_vulkan {
	vector2f position;
	matrix2f covariance;
	/* color info */
};

// transforms 3d gaussians to 2d gaussians in out_buffer using sort_buffer as ordering
void convert_2d_gaussians(
	vulkan::array1<gaussian2d_vulkan>& out_buffer, 
	const vulkan::array1i& sort_buffer, 
	const vulkan::array1<gaussian_vulkan>& gaussians, 
	const matrix4f& camera_transform,
	const matrix4f& camera_projection);
	

// calculates gradient descent per 2d gaussian
void gradient_gaussian(
	vulkan::array1<gaussian2d_vulkan>& out_buffer, 
	const vulkan::array1<gaussian2d_vulkan>& gaussians2d);

// reverses the transformation and generates 3d gaussians for out_buffer
void inverse_to_3d_gaussian(
	vulkan::array1<gaussian_vulkan>& out_buffer
	const vulkan::array1<gaussian2d_vulkan>& gradient,
	const vulkan::array1i& sort_buffer, 
	const matrix4f& camera_transform,
	const matrix4f& camera_projection);

// applyies gradient to gaussians_buffer
void apply_gradient_gaussian(
	vulkan::array1<gaussian_vulkan>& gaussians_buffer,
	const vulkan::array1<gaussian_vulkan>& gradient);	
	
// adds and removes gaussians using parameters (cpu side)
void modify_gaussian(
	/* cpu side gausian list */ gaussians,
	const vulkan::array1<gaussian_vulkan>& gradient);

// need add two arrays
// need multiply by coef

```

**ideas**
the only paramters that can be affected per image are:
- left-right-up-down relative to camera. forward and back both affect scaling nonlinearly and affect sorting non-continuously.
- 2d covariance relative to camera. 2d covariance can easily be proven to be positive semidefinite (all eigenvalues are positive: det>0)

if position gradient is above some threshold, split the gaussian.
if the transparency or alpha value is too low, remove the gaussian.
# GUI

uses dear Im Gui

takes output of renderer and places it into dear im gui stuff

## CLI

train model and ouput gaussians

# Vulkan

needs uniform buffer array, 1d array, 2d array(image), 3d array, render buffer, uniform matrix, compute shader