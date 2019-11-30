//////////////////////////////////////////////////////
// CImgProc, header-only image processing project   //
//  * Author : Wonjun Hwang                         //
//  * E-mail : iamjam4944@gmail.com                 //
//////////////////////////////////////////////////////

//includes.cl

#ifndef CIMGPROC_INCLUDES_CL
#define CIMGPROC_INCLUDES_CL

#define _SET_GLOBAL_ID_1D \
const int x_g = get_global_id(0); \
if(x_g >= length) \
    return; \
const int curPx = x_g;

#define _SET_GLOBAL_ID_2D \
const int x_g = get_global_id(0); \
const int y_g = get_global_id(1); \
if(x_g >= wid || y_g >= hi) \
    return; \
const int curPx = x_g + y_g*wid;

#define _ABS(x) ((x)<0 ? -(x) : x)

#endif //!CIMGPROC_INCLUDES_CL
