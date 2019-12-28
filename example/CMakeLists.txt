######################################################
## CImgProc, header-only image processing project   ##
##  * Author : Wonjun Hwang                         ##
##  * E-mail : iamjam4944@gmail.com                 ##
######################################################
set_as_project()
link_CImgProc()
link_opencv()
#link_eigen()
link_opencl()
link_httplib()
link_CLImgProc()

#target_link_libraries (${current_project} LINK_PUBLIC CLImgProc)
#target_link_libraries (${current_project} PUBLIC CLImgProc)
#target_include_directories (${current_project} PUBLIC CLImgProc)