######################################################
## CImgProc, header-only image processing project   ##
##  * Author : Wonjun Hwang                         ##
##  * E-mail : iamjam4944@gmail.com                 ##
######################################################

######################################################
## Eigen
######################################################

#set path
set(EIGEN_VERSION "3.3.7")
set(EIGEN_SRC "${3RD_PARTY}/eigen/${EIGEN_VERSION}")


#set macro
macro(link_eigen)
    include_directories(${current_project} "${EIGEN_SRC}")

    add_definitions(/D "CIMG_LINK_EIGEN = 1")
    message("${current_project} : Eigen Linked")
endmacro()