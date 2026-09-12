# Goal
The goal is to create a modular 3DGS, Gaussian Splatting, to provide intuitive API for developing experimental workflows. Specific experiments include learning shadows and directional lighting. Further, we are designing this tool to be faster than the original Gaussian Splatting repository.

# Data
The dataset will be www.tanksandtemples.org.
For the shadow and direction lighting tests, we will use the satellite Digital Twin Environment.
# Plan
The plan is to design and create a modular 3DGS tool and gaussian rendering tool in C++. 

The pipeline for 3DGS is:
1. Standardize input data (COLMAP).
2. Initialize gaussians with input point cloud.
3. Iterative gradient descent with parameters position and rotation.
4. Add/remove gaussians as needed.
5. Calculate loss from testing image set.

The hyperparameters used are:
- learning rate
- when to split a gaussian
- when to remove a gaussian
- method to get gaussian color

The project will use CMake for building. The project will use the libraries DeerIMGUI, Eigen, and Vulkan.
## Training
The model is trained with gradient descent. Loss is calculated by per-pixel difference of true images vs. model output images. 

The dataset will be split into training and testing sets. The testing set will be the in-between views predicted by the model.

# Team
The team for the project is:
- Elijah Clark
- Nick Welsh

We will need a third person for the project for the Computer Vision assignment.