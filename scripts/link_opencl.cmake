######################################################
## CImgProc, header-only image processing project   ##
##  * Author : Wonjun Hwang                         ##
##  * E-mail : iamjam4944@gmail.com                 ##
######################################################

######################################################
## OpenCL 1.2
######################################################

#set path
set(OpenCL_VERSION "1.2")
set(OpenCL_INCLUDE "${3RD_PARTY}/OpenCL/OpenCL${OpenCL_VERSION}/include")
set(OpenCL_STATIC_x64 "${3RD_PARTY}/OpenCL/OpenCL${OpenCL_VERSION}/lib/x64/OpenCL.lib")
#set(OpenCL_STATIC_x86 "${3RD_PARTY}/OpenCL/OpenCL${OpenCL_VERSION}/lib/Win32/OpenCL.lib")
set(OpenCL_SRC_DIRPATH "${SRC_DIR}/cl_srcs/")


#set macro
macro(link_opencl)    
    target_include_directories(${current_project} PUBLIC ${OpenCL_INCLUDE})
    target_link_libraries(${current_project} PUBLIC ${OpenCL_STATIC_x64})

    #add_definitions(/D "CL_SRC_DIRPATH=\\\"${OpenCL_SRC_DIRPATH}\\\"")
    add_definitions(/D "CL_SRC_DIRPATH=\\\"/cl_srcs/\\\"")
    add_definitions(/D "CIMG_LINK_OPENCL=1")
    message("${current_project} : OpenCL Linked")
endmacro()