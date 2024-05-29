
# Project Name

## Overview
This project requires the use of libtorch, the PyTorch C++ library. Please follow the instructions below to download and integrate libtorch using CMake.

## Prerequisites
- CMake (version 3.10 or higher)
- A C++ compiler compatible with C++14 or higher

## Instructions

### Step 1: Download libtorch
Download the latest version of libtorch from the official PyTorch website:
[Libtorch Download](https://pytorch.org/get-started/locally/)

Choose the appropriate version for your operating system and download the ZIP archive.

### Step 2: Extract libtorch
Extract the downloaded libtorch ZIP archive to a directory of your choice. For example:
```sh
unzip libtorch-shared-with-deps-latest.zip -d /path/to/your/libs/
```

### Step 3: Add libtorch to Your CMake Project
In your project's `CMakeLists.txt`, add the following lines to include libtorch:

```cmake
cmake_minimum_required(VERSION 3.10)

project(ProjectName)

# Set the path to libtorch
set(CMAKE_PREFIX_PATH "/path/to/your/libs/libtorch")

# Find libtorch package
find_package(Torch REQUIRED)

# Add your executable
add_executable(your_executable_name main.cpp)

# Link libtorch
target_link_libraries(your_executable_name "${TORCH_LIBRARIES}")

# Set C++ standard
set_property(TARGET your_executable_name PROPERTY CXX_STANDARD 14)
```

Replace `/path/to/your/libs/libtorch` with the actual path where you extracted libtorch.

### Step 4: Build Your Project
Navigate to your project's root directory and create a build directory:
```sh
mkdir build
cd build
```

Run CMake to configure your project:
```sh
cmake ..
```

Build your project using the generated Makefile:
```sh
make
```

### Step 5: Run Your Project
Once the build process is complete, you can run your executable:
```sh
./your_executable_name
```

## Additional Information
For more details on using libtorch with C++, please refer to the [PyTorch C++ documentation](https://pytorch.org/cppdocs/).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
