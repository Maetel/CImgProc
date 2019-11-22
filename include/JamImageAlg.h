//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#ifndef CIMGPROC_IMAGEALG_H
#define CIMGPROC_IMAGEALG_H

#include "JamMath.h"

namespace CIMGPROC
{
	namespace ImageAlg
	{
		template <typename T>
		inline double NCC(T const* input1, T const* input2, int length)
		{
			double numerator = 0., denominator_l = 0., denominator_r = 0.;
			for (int idx = 0; idx < length; ++idx)
			{
				const double val_l = input1[idx];
				const double val_r = input2[idx];

				numerator += val_l * val_r;
				denominator_l += val_l * val_l;
				denominator_r += val_r * val_r;
			}

			return numerator / std::sqrt(denominator_l * denominator_r);
		}

		template <typename T>
		inline double ZNCC(T const* input1, T const* input2, int length)
		{
			RunningAggregate agg1, agg2;

			for (int idx = 0; idx < length; ++idx)
			{
				agg1.push(input1[idx]);
				agg2.push(input2[idx]);
			}

			double mean1, mean2, var1, var2;
			agg1.result(0, &mean1, &var1);
			agg2.result(0, &mean2, &var2);
			const double dev1 = std::sqrt(var1);
			const double dev2 = std::sqrt(var2);

			double numerator = 0.;

			for (int idx = 0; idx < length; ++idx)
			{
				numerator += (input1[idx] - mean1) * (input2[idx] - mean2);
			}

			return numerator / double(length * dev1 * dev2);
		}

		template <typename T = double>
		inline void gaussianKernelGeneration(T * output, int kernelSize = 3, double sigma = 1.0)
		{
			// intialising standard deviation to 1.0 
			double r, s = 2.0 * sigma * sigma;

			// sum is for normalization 
			double sum = 0.0;

			const int kernOffset = kernelSize/2;

			// generating 5x5 kernel 
			for (int x = -kernOffset; x <= kernOffset; x++) {
				for (int y = -kernOffset; y <= kernOffset; y++) {
					r = std::sqrt(x * x + y * y);
					const int dst = x + kernOffset + (y + kernOffset) * kernelSize;
					output[dst] = (std::exp(-(r * r) / s)) / (JAM::PI * s);
					sum += output[dst];
				}
			}

			// normalizing the Kernel 
			for (int i = 0; i < kernelSize; ++i)
				for (int j = 0; j < kernelSize; ++j)
					output[i + (j * kernelSize)] /= sum;
		}

		template <typename Tin, typename Tout = Tin, typename T_KERN = double>
		inline void convolution(Tin const* input, Tout * output, int wid, int hi, const T_KERN * kernel, int kern_wid, int kern_hi)
		{
			const int kern_y_offset = kern_hi / 2;
			const int kern_x_offset = kern_wid / 2;

			// decide whether to normalize
			// case i) derivative - sum of kernel components is 0
			// case ii) gaussian blur - sum of kernel components is non-zero
			bool normalize = true;
			T_KERN norm_decider = 0;
			for (int kern_y = 0; kern_y < kern_hi; ++kern_y)
				for (int kern_x = 0; kern_x < kern_wid; ++kern_x)
					norm_decider += kernel[kern_x + kern_y * kern_wid];

			if (0 == norm_decider)
				normalize = false;

			for (int y = 0; y < hi; ++y)
			{
				for (int x = 0; x < wid; ++x)
				{
					double normalizer = 0;
					double filler = 0;
					for (int kern_y = -kern_y_offset; kern_y <= kern_y_offset; ++kern_y)
					{
						const int kern_offset = (kern_y + kern_y_offset) * kern_wid;
						for (int kern_x = -kern_x_offset; kern_x <= kern_x_offset; ++kern_x)
						{
							const int cur_y = y + kern_y, cur_x = x + kern_x;

							//boundary exception
							if (cur_y < 0 || cur_y > hi || cur_x < 0 || cur_x > wid)
								continue;

							double kern_val = kernel[(kern_x + kern_x_offset) + kern_offset];
							if (normalize)
								normalizer += static_cast<double>(kern_val);

							const double val = static_cast<double>(input[cur_x + cur_y * wid] * kern_val);
							filler += val;
						}
					}

					const Tout val = static_cast<Tout>(normalize ? (filler / double(normalizer)) : filler);
					output[x + y * wid] = val;
				}
			}
		}
	}
}
#endif //!CIMGPROC_IMAGEALG_H