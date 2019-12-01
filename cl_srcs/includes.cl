//////////////////////////////////////////////////////
// CImgProc, header-only image processing project   //
//  * Author : Wonjun Hwang                         //
//  * E-mail : iamjam4944@gmail.com                 //
//////////////////////////////////////////////////////

//////////////////////////////////////////////////////
// includes
//////////////////////////////////////////////////////

#ifndef CIMGPROC_INCLUDES_CL
#define CIMGPROC_INCLUDES_CL

//////////////////////////////////////////////////////
// kernel size related
//////////////////////////////////////////////////////
#ifndef LOCAL_WID
#define LOCAL_WID 8
#endif
#ifndef LOCAL_HI
#define LOCAL_HI 8
#endif
//#ifndef KERN_WID
//#define KERN_WID 3
//#endif
#define KERN_WID_OFFSET (KERN_WID/2)
//#ifndef KERN_HI
//#define KERN_HI 3
//#endif
#define KERN_HI_OFFSET (KERN_HI/2)
#define KERN_SIZE (KERN_WID * KERN_HI)
#define KERN_SIZE_HALF (KERN_SIZE/2)
#define LOCAL_KERN_HI (LOCAL_HI + 2*KERN_HI_OFFSET)
#define LOCAL_KERN_WID (LOCAL_WID + 2*KERN_WID_OFFSET)


//////////////////////////////////////////////////////
// utils
//////////////////////////////////////////////////////
#define __CONVERT(_type, concat) _type ## concat
#define _CONVERT(_type, concat) __CONVERT(_type, concat)
#define CONVERTER(_type) _CONVERT(convert_, _type)

#define _ABS(x) ((x)<0 ? -(x) : x)


//////////////////////////////////////////////////////
// global&local coordinate macros
//////////////////////////////////////////////////////
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

#define _SET_LOCAL_ID_2D \
const int x_l = get_local_id(0); \
const int y_l = get_local_id(1); \
const int y_kern_u = get_group_id(1) * get_local_size(1) - KERN_HI_OFFSET; \
const int x_kern_l = get_group_id(0) * get_local_size(0) - KERN_WID_OFFSET; \

// These macros must be defined in advance
// KERN_WID, KERN_HI
#define _COPY_LOCAL_ID_2D(LType, inArr) \
__local LType lmem[LOCAL_KERN_HI][LOCAL_KERN_WID]; \
for(int _y = y_l; _y < LOCAL_KERN_HI; _y += LOCAL_HI ) \
{ \
    const int _cur_y_g = y_kern_u + _y; \
    for(int _x = x_l; _x < LOCAL_KERN_WID; _x += LOCAL_WID ) \
    { \
        const int _cur_x_g = x_kern_l + _x; \
        bool isNaN = true; \
        int _cur_px = 0; \
        if(_cur_x_g >= 0 || _cur_x_g < wid || \
            _cur_y_g >= 0 || _cur_y_g < hi) \
        { \
            _cur_px = _cur_x_g + _cur_y_g * wid; \
            isNaN = !isfinite(convert_float(inArr[_cur_px])); \
        } \
        lmem[_y][_x] = isNaN ? 0 : CONVERTER(LType)(inArr[_cur_px]); \
    } \
} \
barrier(CLK_LOCAL_MEM_FENCE)


#endif //!CIMGPROC_INCLUDES_CL
