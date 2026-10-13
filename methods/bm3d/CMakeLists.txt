cmake_minimum_required(VERSION 3.12)

project(bm3d CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(bm3d bm3d.cpp)

target_link_libraries(bm3d
    ${OpenCV_LIBS}
)
