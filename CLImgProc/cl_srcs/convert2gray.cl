R"(
//////////////////////////////////////////////////////
// CImgProc, header-only image processing project   //
//  * Author : Wonjun Hwang                         //
//  * E-mail : iamjam4944@gmail.com                 //
//////////////////////////////////////////////////////

//////////////////////////////////////////////////////
// convert to gray
//////////////////////////////////////////////////////
#include "includes.cl"

#ifndef TIn
#define TIn uchar
#endif

#ifndef TOut
#define TOut uchar
#endif

__kernel void rgb2gray
(
    __global TIn  const* input,
    __global TOut *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar3 data = vload3(curPx, input);
    output[curPx] = CONVERTER(TOut)(
        data.x * 0.2126f +
        data.y * 0.7152f +
        data.z * 0.0722f
    );
}

__kernel void bgr2gray
(
    __global TIn const* input,
    __global TOut *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar3 data = vload3(curPx, input);
    output[curPx] = CONVERTER(TOut)(
        data.x * 0.0722f +
        data.y * 0.7152f +
        data.z * 0.2126f
    );
}

__kernel void rgba2gray
(
    __global TIn const* input,
    __global TOut *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar4 data = vload4(curPx, input);
    output[curPx] = CONVERTER(TOut)(
        data.x * 0.2126f +
        data.y * 0.7152f +
        data.z * 0.0722f
    );
}

__kernel void bgra2gray
(
    __global TIn const* input,
    __global TOut *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar4 data = vload4(curPx, input);
    output[curPx] = CONVERTER(TOut)(
        data.x * 0.0722f +
        data.y * 0.7152f +
        data.z * 0.2126f
    );
}
)"