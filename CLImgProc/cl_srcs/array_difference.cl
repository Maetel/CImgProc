R"(
//////////////////////////////////////////////////////
// CImgProc, header-only image processing project   //
//  * Author : Wonjun Hwang                         //
//  * E-mail : iamjam4944@gmail.com                 //
//////////////////////////////////////////////////////

//////////////////////////////////////////////////////
// array difference
//////////////////////////////////////////////////////
#include "includes.cl"

#ifndef _TIn
#define _TIn uchar
#endif

#ifndef _TOut
#define _TOut uchar
#endif

__kernel void array_difference
(
    __global _TIn const* input1,
    __global _TIn const* input2,
    __global _TOut *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    output[curPx] = _ABS(input1[curPx] - input2[curPx]);
}
)"