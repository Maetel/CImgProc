######################################################
## CImgProc, header-only image processing project   ##
##  * Author : Wonjun Hwang                         ##
##  * E-mail : iamjam4944@gmail.com                 ##
######################################################


######################################################
## set basic info
######################################################
cmake_minimum_required(VERSION 3.8)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CIMGPROC_INCLUDE_PATH "${SRC_DIR}/include")
set(EXAMPLE_DIR "${SRC_DIR}/example")
set(RESOURCES_DIR "${SRC_DIR}/resources")
add_definitions(-DRESOURCES_DIR="${RESOURCES_DIR}")

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

message("Source directory : ${SRC_DIR}")
message("CImgProc include directory : ${CIMGPROC_INCLUDE_PATH}")

######################################################
## 3rd party
######################################################
set(3RD_PARTY "${SRC_DIR}/3rdparty")

#Eigen
set(EIGEN_VERSION "3.3.7")
set(EIGEN_SRC "${3RD_PARTY}/eigen/${EIGEN_VERSION}")

#OpenCL
set(OpenCL_VERSION "1.2")
set(OpenCL_INCLUDE "${3RD_PARTY}/OpenCL/OpenCL${OpenCL_VERSION}/include")
set(OpenCL_STATIC_x64 "${3RD_PARTY}/OpenCL/OpenCL${OpenCL_VERSION}/lib/x64/OpenCL.lib")
#set(OpenCL_STATIC_x86 "${3RD_PARTY}/OpenCL/OpenCL${OpenCL_VERSION}/lib/Win32/OpenCL.lib")

#OpenCV
set(OPENCV_DIR "C:/work/srcs/opencv")
set(OPENCV_VERSION "4.1.1")
string(REPLACE "." "" OPENCV_VERSION_CLEARED ${OPENCV_VERSION})
set(OPENCV_BUILD "${OPENCV_DIR}/opencv-${OPENCV_VERSION}/build")
if(NOT EXISTS ${OPENCV_BUILD})
    message(FATAL_ERROR "Please set OpenCV path and version")
endif()
set(OPENCV_INCLUDE "${OPENCV_BUILD}/install/include")
set(OPENCV_LIB_DIR "${OPENCV_BUILD}/lib")
set(OPENCV_LIB_OPTIMIZED "${OPENCV_LIB_DIR}/Release/opencv_world${OPENCV_VERSION_CLEARED}.lib")
set(OPENCV_LIB_DEBUG "${OPENCV_LIB_DIR}/Debug/opencv_world${OPENCV_VERSION_CLEARED}d.lib")

######################################################
## macros
######################################################
macro(set_as_project)
    
    #set project
    get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
    string(REPLACE " " "_" ProjectId ${ProjectId})
    set(current_project ${ProjectId})
    project(${current_project} CXX)

    #add files
    #FILE(GLOB CPP_FILES "*.cpp")
    #FILE(GLOB HEADER_FILES "*.h" "*.hpp")
    #add_executable(${current_project} "${CPP_FILES}" "${HEADER_FILES}")
    FILE(GLOB current_project_srcs "*.cpp" "*.h" "*.hpp")
    add_executable(${current_project} ${current_project_srcs})

    message("Project : ${current_project} added")
endmacro()

macro(set_base_project)
    get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
    project(${ProjectId})
    set(current_project ${ProjectId})
endmacro()

######################################################
## link
######################################################
macro(link_eigen)
    include_directories(${current_project} "${EIGEN_SRC}")
    message("${current_project} : Eigen Linked")
endmacro()

macro(link_opencl)    
    include_directories(${current_project} ${OpenCL_INCLUDE})
    target_link_libraries(${current_project} ${OpenCL_STATIC_x64})

    message("OpenCL_INCLUDE ${OpenCL_INCLUDE}")
    message("${current_project} : OpenCL Linked")
endmacro()

macro(link_opencv)
    include_directories(${current_project} ${OPENCV_INCLUDE})
    target_link_libraries(${current_project}
        debug ${OPENCV_LIB_DEBUG}
        optimized ${OPENCV_LIB_OPTIMIZED}
    )
    message("${current_project} : OpenCV world Linked")
endmacro()

macro(link_CImgProc)
    include_directories(${current_project} ${CIMGPROC_INCLUDE_PATH})
    message("${current_project} : CImgProc Linked")
endmacro()

######################################################
## Finally add example
######################################################
set_base_project()
add_subdirectory(${EXAMPLE_DIR})