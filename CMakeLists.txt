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
set(CMAKE_SCRIPT_DIR "${SRC_DIR}/scripts")

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_INSTALL_PREFIX ${SRC_DIR})

message("Source directory : ${SRC_DIR}")
message("Binary directory : ${CMAKE_CURRENT_BINARY_DIR}")

######################################################
## CLImgProc
######################################################

set(CLIMAGEALG_DIR "${SRC_DIR}/CLImgProc")
set(CLIMGPROC_INSTALL_PATH "${SRC_DIR}/CLImgProc_prebuilt")
set(CLIMGPROC_INCLUDE  "${CLIMGPROC_INSTALL_PATH}/include")

set(CLImgProc_build_shared true CACHE BOOL "Build shaed libs")
option(CLImgProc_build_shared "CLImageProc build shaed libs" ON)
if(CLImgProc_build_shared)
    set(CLIMGPROC_LIB "${CLIMGPROC_INSTALL_PATH}/bin")
else()
    set(CLIMGPROC_LIB "${CLIMGPROC_INSTALL_PATH}/lib")
endif()
message("CLIMGPROC_LIB ${CLIMGPROC_LIB}")

######################################################
## 3rd party
######################################################
set(3RD_PARTY "${SRC_DIR}/3rdparty")

FILE(GLOB CMAKE_SCRIPTS "${CMAKE_SCRIPT_DIR}/*.*")

FOREACH(SCRIPT ${CMAKE_SCRIPTS})
    include(${SCRIPT})
ENDFOREACH()

set(JAM_FACE_API_KEY "${SRC_DIR}/private/jam_face_api.key")

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
    FILE(GLOB current_project_srcs "./*.cpp" "./*.h" "./*.hpp")
    add_executable(${current_project} ${current_project_srcs})

    message("Project : ${current_project} added")
endmacro()

macro(set_as_lib)
    #set project
    get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
    string(REPLACE " " "_" ProjectId ${ProjectId})
    set(current_project ${ProjectId})
    
    FILE(GLOB current_project_srcs "*.cpp" "*.h" "*.hpp")
    
    set(LIB_TYPE "")

    if(CLImgProc_build_shared)
        add_library(${current_project} SHARED ${current_project_srcs})
        set(LIB_TYPE "Shared")
    else()
        add_library(${current_project} STATIC ${current_project_srcs})
        set(LIB_TYPE "Static")
    endif()
    
    message("${LIB_TYPE} Library : ${current_project} added")
endmacro()

macro(set_base_project)
    get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
    project(${ProjectId})
    set(current_project ${ProjectId})
endmacro()

######################################################
## link
######################################################
macro(link_CImgProc)
    target_include_directories(${current_project} PUBLIC ${CIMGPROC_INCLUDE_PATH})
    message("${current_project} : CImgProc Linked")
endmacro()

######################################################
## Finally add example
######################################################
set_base_project()
add_subdirectory(${EXAMPLE_DIR})
add_subdirectory(${CLIMAGEALG_DIR})