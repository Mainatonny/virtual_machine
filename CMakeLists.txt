cmake_minimum_required(VERSION 3.10)
project(PointCloudApp)

# Find required packages
find_package(Qt5Core REQUIRED)
find_package(Qt5Network REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the executable
add_executable(PointCloudApp main.cpp)

# Link the libraries
target_link_libraries(PointCloudApp
    Qt5::Core
    Qt5::Network
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)