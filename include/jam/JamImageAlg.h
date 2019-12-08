//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#ifndef CIMGPROC_IMAGEALG_H
#define CIMGPROC_IMAGEALG_H

#include "Jam/JamMath.h"
#include <utility>

namespace CIMGPROC
{
	namespace ImageAlg
	{
		// TODO list
		//	* up/down sampling (bilinear, bicubic)
		//	* demosaic (bayer2rgb)
		//	* bin packing
		//	* morphology (erosion & dilation)
		//	* image pyramid
		//	* median filter to Running median

		using uint8_t = unsigned char;
		using uchar = uint8_t;

		enum Convert2Gray { RGB2GRAY = 0, BGR2GRAY, RGBA2GRAY, BGRA2GRAY, };
		enum Binarize { THRESHOLD = 0, OTSU, };
		enum ImageTransform {
			None = 0,		// input.size == output.size
			MirrorLeftRight,// input.size == output.size, (x,y) -> (wid-x, y)
			MirrorUpDown,	// input.size == output.size, (x,y) -> (x, hi-y)
			Rotate180,		// input.size == output.size, (x,y) -> (wid-x, hi-y)
			Rotate90CW,				// input.size.transpose() == output.size, (x,y) -> (hi-y, x)
			Rotate90CCW,			// input.size.transpose() == output.size, (x,y) -> (y, wid-x)
			Rotate90CW_MirrorUpDown,// input.size.transpose() == output.size, (x,y) -> (hi-y, wid-x)
			Rotate90CCW_MirrorUpDown,//input.size.transpose() == output.size, (x,y) -> (y,x)
		};

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
		} //!convert2Gray

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
		} //!histogram


		// NCC
		//	* Compare partially
		//	* will alocate output pointer if not null
		//	* Compute only finite pixel values
		template <typename T1, typename T2 = double>
		inline void NCC(
			T1 const* input1, int input1_wid, int input1_hi,
			T1 const* input2, int input2_wid, int input2_hi,
			T2* &output, int increment = 3,
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

			for (int y = 0; y < out_hi; y += increment)
			{
				for (int x = 0; x < out_wid; x += increment)
				{
					const int startPoint = x + y * input1_wid;

					double numerator = 0., denominator_l = 0., denominator_r = 0.;
					for (int kern_y = 0; kern_y < input2_hi; kern_y += increment)
					{
						for (int kern_x = 0; kern_x < input2_wid; kern_x += increment)
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
		} //!NCC

		// NCC : Compare fully of the same size
		template <typename T>
		inline double NCC(T const* input1, T const* input2, int length, int increment = 1)
		{
			if (nullptr == input1 || nullptr == input2 || 0 >= length)
				return 0.;

			double* pRetval = 0;
			NCC(input1, length, 1, input2, length, 1, pRetval, increment);
			double retval = *pRetval;
			delete pRetval;
			return retval;
		} //!NCC

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
		} //!mean

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
		} //!variance

		template <typename T1 = uint8_t, typename T2 = double, typename T3 = T2, int Channel = 1, bool SkipNaN = true>
		inline void standardDeviation(T1 const* input, int length, T2* deviations, T3* means = nullptr)
		{
			if (nullptr == input || 0 >= length || nullptr == deviations)
				return;

			double variances[Channel];
			variance(input, length, variances, means);
			for (int ch = 0; ch < Channel; ++ch)
				deviations[ch] = static_cast<T2>(std::sqrt(variances[ch]));
		} //!standardDeviation

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
		} //!ZNCC

		//https://www.geeksforgeeks.org/gaussian-filter-generation-c/
		template <typename T = double>
		inline void gaussianKernelGeneration(T * output, int kernelSize = 3, double sigma = 1.0)
		{
			// intialising standard deviation to 1.0 
			double s = 2.0 * sigma * sigma;

			// sum is for normalization 
			double sum = 0.0;

			const int kernOffset = kernelSize/2;

			// generating kernel 
			for (int x = -kernOffset; x <= kernOffset; x++) {
				for (int y = -kernOffset; y <= kernOffset; y++) {
					const int dst = x + kernOffset + (y + kernOffset) * kernelSize;
					output[dst] = (std::exp(-1 * (x * x + y * y) / s)) / (CIMGPROC::PI * s);
					sum += output[dst];
				}
			}

			// normalizing the Kernel 
			for (int i = 0; i < kernelSize; ++i)
				for (int j = 0; j < kernelSize; ++j)
					output[i + (j * kernelSize)] /= sum;
		} //!gaussianKernelGeneration

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
		} //!convolution

		template <typename T1, typename T2 = T1>
		inline void gaussianConvolution(T1 const* input, T2* output, int wid, int hi, int kernelSize = 3, double sigma = 1.0)
		{
			if (nullptr == input || nullptr == output || 0 >= wid || 0 >= hi || 3 > kernelSize || 0. > sigma)
				return;

			std::vector<double> gaussianKernel(kernelSize * kernelSize);
			gaussianKernelGeneration(gaussianKernel.data(), kernelSize, sigma);

			convolution(input, output, wid, hi, gaussianKernel.data(), kernelSize, kernelSize);
		} //!gaussianConvolution

		template <typename T1, typename T2 = T1>
		inline void difference(T1 const* input1, T1 const* input2, int length, T2 * output, double* sum_of_distance = nullptr)
		{
			distance(input1, input2, length, output, sum_of_distance);
		} //!difference
		template <typename T1, typename T2 = T1>
		inline void distance(T1 const* input1, T1 const* input2, int length, T2 * output, double* sum_of_distance = nullptr)
		{
			if (nullptr == input1 || nullptr == input2 || 0 >= length || nullptr == output)
				return;
			
			const bool doSum = nullptr != sum_of_distance;
			double summer = 0.;

			for (int idx = 0; idx < length; ++idx)
			{
				const double val = std::abs(double(input1[idx]) - double(input2[idx]));
				output[idx] = T2(val);
				if (doSum)
					summer += val;
			}
			
			if (doSum)
				* sum_of_distance = summer;
		} //!distance

		// @param threshold : in case of THRESHOLD, input. in case of OTSU, output
		template <int BinType = OTSU, typename T = uint8_t>
		inline void binarize(uint8_t const* input, uint8_t* output, int length, T* threshold = nullptr)
		{
			if (nullptr == input || nullptr == output || 0 >= length)
				return;

			T _thres = 0;
			T& thres = (nullptr == threshold) ? _thres : *threshold;

			// Otsu binarization
			//https://www.ipol.im/pub/art/2016/158/
			if constexpr (BinType == OTSU)
			{
				int hist[256];
				histogram(input, length, hist);

				// Compute threshold
				// Init variables
				float sum = 0;
				float sumB = 0;
				int q1 = 0;
				int q2 = 0;
				float varMax = 0;

				// Auxiliary value for computing m2
				for (int i = 0; i <= 255; i++)
					sum += i * ((int)hist[i]);
					
				for (int i = 0; i <= 255; i++)
				{
					// Update q1
					q1 += hist[i];
					if (q1 == 0)
						continue;
					// Update q2
					q2 = length - q1;

					if (q2 == 0)
						break;
					// Update m1 and m2
					sumB += (float)(i * ((int)hist[i]));
					float m1 = sumB / q1;
					float m2 = (sum - sumB) / q2;

					// Update the between class variance
					float varBetween = (float)q1 * (float)q2 * (m1 - m2) * (m1 - m2);

					// Update the threshold if necessary
					if (varBetween > varMax)
					{
						varMax = varBetween;
						thres = static_cast<uint8_t>(i);
					}
				}
			}
			else if constexpr (BinType == THRESHOLD)
			{
				if (nullptr == threshold)
					return;
			}

			std::fill(output, output + length, 0);
			for (int idx = 0; idx < length; ++idx)
				if (input[idx] > thres)
					output[idx] = 255;
		} //!binarize

		template <typename T1 = uint8_t, typename T2 = T1>
		inline void differenceOfGaussian(uint8_t const* input, T1* output, int wid, int hi, int gauss_size_l = 3, int gauss_size_r = 7, double gauss_sigma_l = 1.0, double gauss_sigma_r = 2.0)
		{
			if (nullptr == input || nullptr == output ||  0 > wid || 0 > hi)
				return;

			const int length = wid * hi;
			T2 *gauss_l, *gauss_r;
			gauss_l = new T2[length];
			gauss_r = new T2[length];
			ImageAlg::gaussianConvolution(input, gauss_l, wid, hi, gauss_size_l, gauss_sigma_l);
			ImageAlg::gaussianConvolution(input, gauss_r, wid, hi, gauss_size_r, gauss_sigma_r);
			ImageAlg::difference(gauss_l, gauss_r, length, output);
		} //!differenceOfGaussian

		template <typename T1 = uint8_t, typename T2 = T1, bool Binarize = true>
		inline void dilation(T1 const* input, T2 * output, int wid, int hi, int kern_size = 3)
		{
			const int kern_length = kern_size * kern_size;
			std::vector<uint8_t> dilation_kernel(kern_length);
			std::fill(dilation_kernel.begin(), dilation_kernel.end(), 1);
			
			const int zero = 0;
			if constexpr (Binarize && std::is_same<T1, uint8_t>::value && std::is_same<T2, uint8_t>::value)
			{
				T2* buffer = new T2[wid * hi];
				convolution(input, buffer, wid, hi, dilation_kernel.data(), kern_size, kern_size);
				binarize<THRESHOLD>(buffer, output, wid * hi, &zero);
				delete buffer;
			}
			else
				convolution(input, output, wid, hi, dilation_kernel.data(), kern_size, kern_size);
		} //!dilation

		// not affine transformation
		template <typename T = uint8_t>
		inline void transformation(T const* input, T *output, int wid, int hi, ImageTransform transform)
		{
			if (nullptr == input || nullptr == output || 0 >= wid || 0 >= hi)
				return;

			constexpr int typeSize = sizeof(T);
			const int length = wid * hi;

#define PxVal(_arr, _x, _y, _stride) ((_arr)[(_x) + (_y) * (_stride)])

			//case char || uchar, suitable for memcpy
			switch (transform)
			{
			case None: // input.size == output.size
				memcpy(output, input, typeSize * length);
				break;
			case MirrorLeftRight: // input.size == output.size, (x,y) -> (wid-x, y)
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						PxVal(output, wid - 1 - x, y, wid) = PxVal(input, x, y, wid);
				break;
			case MirrorUpDown: // input.size == output.size, (x,y) -> (x, hi-y)
				for (int y = 0; y < hi; ++y)
					memcpy(output + (y * wid), input + ((hi - 1 - y) * wid), typeSize * wid);
				break;
			case Rotate180: // input.size == output.size, (x,y) -> (wid-x, hi-y)
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						PxVal(output, wid - 1 - x, hi - 1 - y, wid) = PxVal(input, x, y, wid);
				break;
			case Rotate90CW: // input.size.transpose() == output.size, (x,y) -> (hi-y, x)
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						PxVal(output, hi - 1 - y, x, hi) = PxVal(input, x, y, wid);
				break;
			case Rotate90CCW: // input.size.transpose() == output.size, (x,y) -> (y, wid-x)
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						PxVal(output, y, wid - 1 - x, hi) = PxVal(input, x, y, wid);
				break;
			case Rotate90CW_MirrorUpDown: // input.size.transpose() == output.size, (x,y) -> (hi-y, wid-x)
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						PxVal(output, hi - 1 - y, wid - 1 - x, hi) = PxVal(input, x, y, wid);
				break;
			case Rotate90CCW_MirrorUpDown: //input.size.transpose() == output.size, (x,y) -> (y,x)
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						PxVal(output, y, x, hi) = PxVal(input, x, y, wid);
				break;
			}
#undef PxVal
		} // !transformation parameter type
		
		//returns <dst_x, dst_y, stride>
		template<int Transform>
		constexpr std::tuple<int, int, int> _transform_dst(int wid, int hi, int x, int y)
		{
			using Result = std::tuple<int, int, int>;
			if constexpr (None == Transform)
				return Result(x, y, wid);
			else if constexpr (MirrorLeftRight == Transform)
				return Result(wid - 1 - x, y, wid);
			else if constexpr (MirrorUpDown == Transform)
				return Result(x, hi - 1 - y, wid);
			else if constexpr (Rotate180 == Transform)
				return Result(wid - 1 - x, hi - 1 - y, wid);
			else if constexpr (Rotate90CW == Transform)
				return Result(hi - 1 - y, x, hi);
			else if constexpr (Rotate90CCW == Transform)
				return Result(y, wid - 1 - x, hi);
			else if constexpr (Rotate90CW_MirrorUpDown == Transform)
				return Result(hi - 1 - y, wid - 1 - x, hi);
			else if constexpr (Rotate90CCW_MirrorUpDown == Transform)
				return Result(y, x, hi);
		}

		// not affine transformation
		template <int Transform, typename T = uint8_t>
		inline void transformation(T const* input, T * output, int wid, int hi)
		{
			if (nullptr == input || nullptr == output || 0 >= wid || 0 >= hi)
				return;

			constexpr int typeSize = sizeof(T);
			const int length = wid * hi;

			if constexpr (None == Transform)
			{
				memcpy(output, input, typeSize * length);
			}
			else if constexpr (MirrorUpDown == Transform)
			{
				for (int y = 0; y < hi; ++y)
					memcpy(output + (y * wid), input + ((hi - 1 - y) * wid), typeSize * wid);
			}
			else
			{
#define PxVal(_arr, _x, _y, _stride) ((_arr)[(_x) + (_y) * (_stride)])
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						PxVal(output,
							std::get<0>(_transform_dst<Transform>(wid, hi, x, y)), // dst_x
							std::get<1>(_transform_dst<Transform>(wid, hi, x, y)), // dst_y
							std::get<2>(_transform_dst<Transform>(wid, hi, x, y))  // stride
							) = PxVal(input, x, y, wid);
#undef PxVal
			}
		} // ! transformation template type

		template <typename T1, typename T2, bool SkipNaN = true>
		void medianFilter(T1 const* input, T2 *output, int wid, int hi, int kernelSize = 3)
		{
			if (nullptr == input || nullptr == output || 0 >= wid || 0 >= hi || 3 > kernelSize)
				return;

			const int kern_offset = kernelSize / 2;

			for (int y = 0; y < hi; ++y)
			{
				for (int x = 0; x < wid; ++x)
				{
					std::vector<T1> values;
					int kern_count = 0;
					for (int kern_y = -kern_offset; kern_y <= kern_offset; ++kern_y)
					{
						for (int kern_x = -kern_offset; kern_x <= kern_offset; ++kern_x)
						{
							const int cur_x = x + kern_x, cur_y = y + kern_y;
							
							//boundary exception
							if (cur_x < 0 || cur_x >= wid || cur_y < 0 || cur_y >= hi)
								continue;
							const T1 curval = input[cur_x + cur_y * wid];
							
							if constexpr (SkipNaN && (std::is_same<T1, float>::value || std::is_same<T1, double>::value))
								if (!isfinite(curval))
									continue;

							auto it = std::upper_bound(values.begin(), values.end(), curval);
							values.insert(it, curval);
							kern_count++;
						}
					}
					const auto midVal = static_cast<T2>(values.at(kern_count/2));
					output[x + y * wid] = midVal;
				}
			}
		}

		template <typename TIn, typename TOut, typename TMask>
		void diffuse(TIn const* input1, TIn const* input2, TMask const* mask, TOut *output, int length)
		{
			if (nullptr == input1 || nullptr == input2 || nullptr == mask || nullptr == output || length <= 0)
				return;

			constexpr auto MaskMax =
				std::is_same<TMask, uint8_t>::value ? 255 :
				std::is_same<TMask, float>::value ? 1.0f :
				//std::is_same<TMask, double>::value ? 1.0 :
				1.0;

			for (int idx = 0; idx < length; ++idx)
			{
				const auto val1 = input1[idx];
				const auto val2 = input2[idx];
				const auto maskVal = mask[idx];
				output[idx] = val1 * (1. - double(maskVal / double(MaskMax))) + val2 * double(maskVal /double(MaskMax));
			}
		}

		template <typename TIn, typename TOut>
		void diffuse(TIn const* input1, TIn const* input2, double ratio_1to2, TOut* output, int length)
		{
			if (nullptr == input1 || nullptr == input2 || ratio_1to2 < 0. || ratio_1to2 > 1. || nullptr == output || length <= 0)
				return;

			for (int idx = 0; idx < length; ++idx)
			{
				const auto val1 = input1[idx];
				const auto val2 = input2[idx];
				output[idx] = TOut(double(val1 * (ratio_1to2)) + double(val2 * (1. - ratio_1to2)));
			}
		}

	} //!ImageAlg
}
#endif //!CIMGPROC_IMAGEALG_H