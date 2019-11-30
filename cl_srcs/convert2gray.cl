//////////////////////////////////////////////////////
// CImgProc, header-only image processing project   //
//  * Author : Wonjun Hwang                         //
//  * E-mail : iamjam4944@gmail.com                 //
//////////////////////////////////////////////////////

//rgb2gray.cl
#include "includes.cl"

__kernel void rgb2gray
(
    __global uchar const* input,
    __global uchar *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar3 data = vload3(curPx, input);
    output[curPx] = convert_uchar(
        data.x * 0.2126f +
        data.y * 0.7152f +
        data.z * 0.0722f
    );
}

__kernel void bgr2gray
(
    __global uchar const* input,
    __global uchar *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar3 data = vload3(curPx, input);
    output[curPx] = convert_uchar(
        data.x * 0.0722f +
        data.y * 0.7152f +
        data.z * 0.2126f
    );
}

__kernel void rgba2gray
(
    __global uchar const* input,
    __global uchar *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar4 data = vload4(curPx, input);
    output[curPx] = convert_uchar(
        data.x * 0.2126f +
        data.y * 0.7152f +
        data.z * 0.0722f
    );
}

__kernel void bgra2gray
(
    __global uchar const* input,
    __global uchar *output,
    int length
)
{
    _SET_GLOBAL_ID_1D;

    uchar4 data = vload4(curPx, input);
    output[curPx] = convert_uchar(
        data.x * 0.0722f +
        data.y * 0.7152f +
        data.z * 0.2126f
    );
}
