cmake_minimum_required(VERSION 3.20)
project(sfm_with_cpp)

find_package(OpenCV REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(include)

# Add executable and source files
add_executable(sfm_with_cpp src/main.cpp )

# Link libraries
target_link_libraries(sfm_with_cpp PRIVATE 
    ${OpenCV_LIBS} 
    Ceres::ceres 
    Eigen3::Eigen
)