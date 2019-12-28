######################################################
## CImgProc, header-only image processing project   ##
##  * Author : Wonjun Hwang                         ##
##  * E-mail : iamjam4944@gmail.com                 ##
######################################################

######################################################
## OpenCV
######################################################

#set path
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


#set macro
macro(link_opencv)
    target_include_directories(${current_project} PUBLIC ${OPENCV_INCLUDE})
    target_link_libraries(${current_project} PUBLIC
        debug ${OPENCV_LIB_DEBUG}
        optimized ${OPENCV_LIB_OPTIMIZED}
    )
    add_definitions(-D CIMG_LINK_OPENCV=1)
    message("${current_project} : OpenCV world Linked")
endmacro()