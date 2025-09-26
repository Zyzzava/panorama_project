# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(PanoramaProject)

# Locate the OpenCV library package
find_package(OpenCV REQUIRED)

# Add the OpenCV include directories to the compiler’s search path.
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable target and its source files.
add_executable(PanoramaProject
    main.cpp
    FeatureProcessing.cpp
    Report.cpp
)
# Link the executable against the required OpenCV libraries.
target_include_directories(PanoramaProject PRIVATE include)