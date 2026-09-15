# Design

## Backend
Vulkan, training pipeline, renderer, database

## Frontend
CLI, GUI

# Subsystems
## Database

functionality:
- store gaussians in gpu.
- store images in gpu.
- generate gaussians from point cloud and images.
- transfer gaussians to cpu

Modular:
- input data source - colmaps
- output file - point cloud with gaussians. **research this**
- what attributes to store - Should attributes be dynamic or static?

Depends on:
- vulkan
- core

Questions
- Array-of-structs or structs-of-arrays?
- utilize intense colmaps library or use external tool to generate text file format for colmaps data?
- what image manipulation library to use?

External Libraries:
- OpenCV - image loading and manipulation
- COLMAPS?

Research:
- output file type

## Vulkan

functionality
- initialize vulkan context
- invoke compute shader
- store buffer of arbitrary arrays of data
- store image
- transfer data cpu to gpu
- transfer data from gpu to cpu

Depends on:
- core

External Libraries:
- glfw
- vulkan sdk

Research:
- storing data in Vulkan Buffers.
- storing images in Vulkan.
- should we design for multiple GPUs?

## Training Pipeline

Functionality:
- project 3d gaussians to 2d gaussians
- sort by depth
- project 2d gaussians to 3d gaussians (applying gradient descent)
- calculate gradient descent for $\Sigma'$ and $\mu'$, compute shader.
- gradient descent for color
- splitting gaussians
- removing gaussians

Modular:
- color
- splitting gaussians
- removing gaussians
- parameters for configuring per number of iteration  (batching, when to split/remove gaussians)

Depends on:
- vulkan
- database
- core

Questions:
- what should we make modular
- Should we do multi-threading?

Future:
- optimize by  splitting gaussians into screen buckets

## CLI/Traing frontend

functionality:
- call gradient descent
- config files
- load chosen input database
- load checkpoints
- choose export file type
- verify testing set of images.

Modular:
- Database provides names and info of database type automatically for cli

Depends on:
- training
- database
- core

Questions:
- should CLI manage and run GUI, or should there be a "training" and "viewing" app?
- Should we design for python bindings?
- should there be a GUI version of the CLI training app

External library:
- Boost.Program_options

## GUI/Viewing results frontend

functionality:
- render image to vulkan image buffer, compute shader
- load gaussian file
- move camera around
- go to preset locations determined by input images and provide Loss information at view.
- load testing image set and compare.
- render and export chosen perspectives as video/images.
- Sort gaussians by depth
- project 3d gaussians to 2d gaussians
- calculate gaussian color based on perspective

Modular:
- load chosen gaussian file type

Depends on:
- database
- vulkan
- core

Question:
- what should it look like?
- should we develop graphics for general use or more complex use?

External libraries:
- DearImgui






