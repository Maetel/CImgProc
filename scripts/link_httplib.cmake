######################################################
## CImgProc, header-only image processing project   ##
##  * Author : Wonjun Hwang                         ##
##  * E-mail : iamjam4944@gmail.com                 ##
######################################################

######################################################
## Httplib
## https://github.com/yhirose/cpp-httplib
######################################################

#set path
set(HTTPLIB_PATH "${3RD_PARTY}/httplib")
set(HTTPLIB_INCLUDE "${HTTPLIB_PATH}/include")

#set macro
macro(link_httplib)
    target_include_directories(${current_project} PUBLIC ${HTTPLIB_INCLUDE})
    add_definitions(-D CIMG_LINK_HTTPLIB=1)
    if(EXISTS "${JAM_FACE_API_KEY}")
        add_definitions(/D "CIMG_FACE_API_KEY=\\\"${JAM_FACE_API_KEY}\\\"")
    endif()
    message("${current_project} : HttpLib Linked")
endmacro()