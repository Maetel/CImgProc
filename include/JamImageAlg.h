//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#ifndef CIMGPROC_IMAGEALG_H
#define CIMGPROC_IMAGEALG_H

#include "JamMath.h"
#include <utility>

namespace CIMGPROC
{
	namespace ImageAlg
	{
		// TODO list
		//	* otsu binarization
		//	* up/down sampling
		//	* demosaic (bayer2rgb)

		using uint8_t = unsigned char;
		using uchar = uint8_t;

		enum Convert2Gray { RGB2GRAY = 0, BGR2GRAY, RGBA2GRAY, BGRA2GRAY};

		template <int ConvertType = RGB2GRAY, int Channel = 3>
		inline void convert2Gray(uint8_t const* input, uint8_t * output, int length)
		{
			if (nullptr == input || nullptr == output || 0 >= length)
				return;
			
			constexpr int _R =
				(RGB2GRAY == ConvertType) || (RGBA2GRAY == ConvertType) ? 0 :
				(BGR2GRAY == ConvertType) || (BGRA2GRAY == ConvertType) ? 2 : 0;
			constexpr int _G =
				(RGB2GRAY == ConvertType) || (RGBA2GRAY == ConvertType) ? 1 :
				(BGR2GRAY == ConvertType) || (BGRA2GRAY == ConvertType) ? 1 : 0;
			constexpr int _B =
				(RGB2GRAY == ConvertType) || (RGBA2GRAY == ConvertType) ? 2 :
				(BGR2GRAY == ConvertType) || (BGRA2GRAY == ConvertType) ? 0 : 0;

			for (int idx = 0; idx < length; ++idx)
			{
				output[idx] = static_cast<uint8_t>(
					static_cast<float>(input[(idx * Channel) + _R]) * 0.2126f +
					static_cast<float>(input[(idx * Channel) + _G]) * 0.7152f +
					static_cast<float>(input[(idx * Channel) + _B]) * 0.0722f
					);
			}
		}

		template <int Channel = 1, typename T = int>
		inline void histogram(uint8_t const* input, int length, T* output)
		{
			if (nullptr == input || nullptr == output || 0 >= length)
				return;

			std::fill(output, output + (256 * Channel), 0);

			for (int idx = 0; idx < length; ++idx)
			{
				if constexpr (1 == Channel)
				{
					++output[input[idx]];
				}
				else
					for (int ch = 0; ch < Channel; ++ch)
						++output[input[idx*Channel + ch] * Channel + ch];
			}
		}

		// NCC
		//	* Compare partially
		//	* will alocate output pointer if not null
		//	* Compute only finite pixel values
		template <typename T1, typename T2 = double>
		inline void NCC(
			T1 const* input1, int input1_wid, int input1_hi,
			T1 const* input2, int input2_wid, int input2_hi,
			T2* &output,
			std::pair<int,int>* point_of_the_highest_match = nullptr, double* value_of_the_highest_match = nullptr,
			int* output_wid = nullptr, int* output_hi = nullptr
		)
		{
			if (
				nullptr == input1 || 0 >= input1_wid || 0 >= input1_hi ||
				nullptr == input2 || 0 >= input2_wid || 0 >= input2_hi
				)
				return;

			const int out_wid = input1_wid - input2_wid + 1;
			const int out_hi = input1_hi - input2_hi + 1;
			
			// size exception
			if (out_wid <= 0 || out_hi <= 0)
				return;

			if (nullptr != output_wid)
				*output_wid = out_wid;
			if (nullptr != output_hi)
				*output_hi = out_hi;

			if (nullptr == output)
				output = new T2[out_wid * out_hi];

			double highest_val = 0.;
			int highest_val_x = -1;
			int highest_val_y = -1;

			for (int y = 0; y < out_hi; ++y)
			{
				for (int x = 0; x < out_wid; ++x)
				{
					const int startPoint = x + y * input1_wid;

					double numerator = 0., denominator_l = 0., denominator_r = 0.;
					for (int kern_y = 0; kern_y < input2_hi; ++kern_y)
					{
						for (int kern_x = 0; kern_x < input2_wid; ++kern_x)
						{
							const int dst1 = startPoint + kern_x + kern_y * input1_wid;
							const int dst2 = kern_x + kern_y * input2_wid;
							const double val_l = isfinite(static_cast<double>(input1[dst1])) ? input1[dst1] : 0.;
							const double val_r = isfinite(static_cast<double>(input2[dst2])) ? input2[dst2] : 0.;

							numerator += val_l * val_r;
							denominator_l += val_l * val_l;
							denominator_r += val_r * val_r;
						}
					}

					const double val = numerator / std::sqrt(denominator_l * denominator_r);
					if (val > highest_val)
					{
						highest_val = val;
						highest_val_x = x;
						highest_val_y = y;
					}

					output[x + y * out_wid] = static_cast<T2>(val);
				}
			}

			if (nullptr != point_of_the_highest_match)
			{
				*point_of_the_highest_match = { highest_val_x, highest_val_y };
			}
			if (nullptr != value_of_the_highest_match)
			{
				*value_of_the_highest_match = highest_val;
			}
		}
		// NCC : Compare fully of the same size
		template <typename T>
		inline double NCC(T const* input1, T const* input2, int length)
		{
			if (nullptr == input1 || nullptr == input2 || 0 >= length)
				return 0.;

			double* pRetval = 0;
			NCC(input1, length, 1, input2, length, 1, pRetval);
			double retval = *pRetval;
			delete pRetval;
			return retval;
		}

		// skips nan?
		template <typename T1 = uint8_t, typename T2 = double, int Channel = 1, bool SkipNaN = true>
		inline void mean(T1 const* input, int length, T2* outputs)
		{
			if (nullptr == input || 0 >= length || nullptr == outputs)
				return;

			double retvals[Channel];
			std::fill(retvals, retvals + Channel, 0.);

			for (int idx = 0; idx < length; ++idx)
			{
				for (int ch = 0; ch < Channel; ++ch)
				{
					if constexpr (!SkipNaN)
					{
						retvals[ch] += input[idx * Channel + ch];
					}
					else
					{
						const auto val = static_cast<double>(input[idx * Channel + ch]);
						retvals[ch] += isfinite(val) ? val : 0. ;
					}
						
				}
			}

			for (int ch = 0; ch < Channel; ++ch)
			{
				outputs[ch] = static_cast<T2>(retvals[ch] / double(length));
			}
		}

		template <typename T1 = uint8_t, typename T2 = double, typename T3 = T2, int Channel = 1, bool SkipNaN = true>
		inline void variance(T1 const* input, int length, T2* variances, T3* means = nullptr)
		{
			if (nullptr == input || 0 >= length || nullptr == variances)
				return;

			RunningAggregate<SkipNaN> agg[Channel];

			for (int idx = 0; idx < length; ++idx)
				for(int ch = 0; ch < Channel; ++ch)
					agg[ch].push(input[idx * Channel + ch]);

			for (int ch = 0; ch < Channel; ++ch)
			{
				double mean, var;
				agg[ch].result(0, &mean, &var);
				variances[ch] = static_cast<T2>(var);
				if (nullptr != means)
					means[ch] = static_cast<T3>(mean);
			}
		}

		template <typename T1 = uint8_t, typename T2 = double, typename T3 = T2, int Channel = 1, bool SkipNaN = true>
		inline void standardDeviation(T1 const* input, int length, T2* deviations, T3* means = nullptr)
		{
			if (nullptr == input || 0 >= length || nullptr == deviations)
				return;

			double variances[Channel];
			variance(input, length, variances, means);
			for (int ch = 0; ch < Channel; ++ch)
				deviations[ch] = static_cast<T2>(std::sqrt(variances[ch]));
		}

		template <typename T>
		inline double ZNCC(T const* input1, T const* input2, int length)
		{
			if (nullptr == input1 || nullptr == input2 || 0 >= length)
				return HUGE_VAL;

#if 0
			//compute standard deviation manually
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
#else
			double mean1, mean2, dev1, dev2;
			standardDeviation(input1, length, &dev1, &mean1);
			standardDeviation(input2, length, &dev2, &mean2);
#endif
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

			// generating kernel 
			for (int x = -kernOffset; x <= kernOffset; x++) {
				for (int y = -kernOffset; y <= kernOffset; y++) {
					r = std::sqrt(x * x + y * y);
					const int dst = x + kernOffset + (y + kernOffset) * kernelSize;
					output[dst] = (std::exp(-(r * r) / s)) / (CIMGPROC::PI * s);
					sum += output[dst];
				}
			}

			// normalizing the Kernel 
			for (int i = 0; i < kernelSize; ++i)
				for (int j = 0; j < kernelSize; ++j)
					output[i + (j * kernelSize)] /= sum;
		}

		template <typename T1, typename T2 = T1, typename T_KERN = double, bool SkipNaN = true>
		inline void convolution(T1 const* input, T2 * output, int wid, int hi, const T_KERN * kernel, int kern_wid, int kern_hi)
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

			if (0.0001 >= std::abs(norm_decider))
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
							if (cur_y < 0 || cur_y >= hi || cur_x < 0 || cur_x >= wid)
								continue;

							const auto curVal = input[cur_x + cur_y * wid];

							if constexpr (SkipNaN && (std::is_same<T1, float>::value || std::is_same<T1, double>::value))
								if (!isfinite(curVal))
									continue;

							double kern_val = kernel[(kern_x + kern_x_offset) + kern_offset];
							if (normalize)
								normalizer += static_cast<double>(kern_val);

							const double val = static_cast<double>(curVal * kern_val);
							filler += val;
						}
					}

					const T2 val = static_cast<T2>(normalize ? (filler / double(normalizer)) : filler);
					output[x + y * wid] = val;
				}
			}
		}

		template <typename T1, typename T2 = T1>
		inline void gaussianConvolution(T1 const* input, T2* output, int wid, int hi, int kernelSize = 3, double sigma = 1.0)
		{
			if (nullptr == input || nullptr == output || 0 >= wid || 0 >= hi || 3 > kernelSize || 0. > sigma)
				return;

			std::vector<double> gaussianKernel(kernelSize * kernelSize);
			gaussianKernelGeneration(gaussianKernel.data(), kernelSize, sigma);

			convolution(input, output, wid, hi, gaussianKernel.data(), kernelSize, kernelSize);
		}
	} //!ImageAlg
}
#endif //!CIMGPROC_IMAGEALG_H