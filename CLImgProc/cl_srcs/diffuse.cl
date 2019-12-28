R"(
//////////////////////////////////////////////////////
// CImgProc, header-only image processing project   //
//  * Author : Wonjun Hwang                         //
//  * E-mail : iamjam4944@gmail.com                 //
//////////////////////////////////////////////////////

//////////////////////////////////////////////////////
// diffuse
//////////////////////////////////////////////////////
#external_options
#include "includes.cl"

#ifndef TIn
#define TIn uchar
#endif

#ifndef TOut
#define TOut uchar
#endif

#ifndef TMask
#define TMask float
#endif

__kernel void diffuse_ratio
(
    __global TIn const *input1,
    __global TIn const *input2,
    __global TOut *output,
    float ratio_1to2,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    const TIn val1 = input1[curPx];
    const TIn val2 = input2[curPx];
    output[curPx] = CONVERTER(TOut)(convert_float(val1 * ratio_1to2) + convert_float(val2 * (1.f - ratio_1to2)));
}

__kernel void diffuse_mask
(
    __global TIn const *input1,
    __global TIn const *input2,
    __global TOut *output,
    __global TMask *mask,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    const TIn val1 = input1[curPx];
    const TIn val2 = input2[curPx];
    const TMask maskVal = mask[curPx];
    output[curPx] = CONVERTER(TOut)(convert_float(val1 * maskVal) + convert_float(val2 * (1.f - maskVal)));
}
)"