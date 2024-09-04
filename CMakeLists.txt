cmake_minimum_required(VERSION 3.9)
project(yolosense)

set(CMAKE_CXX_STANDARD 14)

find_package(realsense2 REQUIRED)

# pcl
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# boost
find_package(Boost REQUIRED COMPONENTS thread)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(-DBOOST_ALL_DYN_LINK)

# OpenMP
find_package(TBB REQUIRED)

add_library (template_alignment SHARED template_alignment.cpp)

target_link_libraries (template_alignment pthread ${realsense2_LIBRARY} ${PCL_LIBRARIES} ${Boost_LIBRARIES} TBB::tbb)