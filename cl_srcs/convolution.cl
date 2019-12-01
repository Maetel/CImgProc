//////////////////////////////////////////////////////
// CImgProc, header-only image processing project   //
//  * Author : Wonjun Hwang                         //
//  * E-mail : iamjam4944@gmail.com                 //
//////////////////////////////////////////////////////

//////////////////////////////////////////////////////
// convolution
//////////////////////////////////////////////////////
#external_options
#include "includes.cl"

// These two must be defined in advance;
// KERN_WID, KERN_HI

#ifndef TIn
#define TIn uchar
#endif

#ifndef TOut
#define TOut uchar
#endif

__kernel void convolution
(
    __global TIn const *input,
    __global float const *filter,
    __global TOut *output,
    int wid, int hi,
    int normalize
)
{
    _SET_GLOBAL_ID_2D;
    _SET_LOCAL_ID_2D;
    _COPY_LOCAL_ID_2D(float, input);

    float valSum = 0.;
    float normSum = 0.;

//#pragma unroll
    for(int kern_y = 0; kern_y < KERN_HI; ++kern_y)
    {
        const int cur_y = kern_y + y_l;
        for(int kern_x = 0; kern_x < KERN_WID; ++kern_x)
        {
            const int cur_x = kern_x + x_l;
            const float curVal = lmem[cur_y][cur_x];
            const float filterVal = filter[kern_x + kern_y * KERN_WID];
            valSum += curVal * filterVal;
            if(normalize)
                normSum += filterVal;
        }
    }
    if(normalize)
        valSum /= normSum;
    output[curPx] = CONVERTER(TOut)(valSum);
}
