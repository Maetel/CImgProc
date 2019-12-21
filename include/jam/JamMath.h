//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#ifndef CIMGPROC_MATH_H
#define CIMGPROC_MATH_H

namespace CIMGPROC
{
	//16 digits
	constexpr double PI = 3.1415926535897932;
#ifndef FLT_EPSILON
	constexpr float FLT_EPSILON = 0.00001f;
#endif

	// * Computes mean and variance(standard deviation) on the run
	// * If there is any NaN input, output will be always NaN
	// * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

	template <bool SkipNaN = true>
	struct RunningAggregate
	{
		using uint64_t = unsigned long long;
		uint64_t count = 0;
		double mean = 0.;
		double var_summed = 0.;
		void init() { count = 0; mean = 0.; var_summed = 0.; }

		RunningAggregate() = default;

		void push(double val)
		{
			if constexpr (SkipNaN)
				if (!isfinite(val))
					return;

			count++;
			double delta1 = val - mean;
			mean += delta1 / double(count);
			double delta2 = val - mean;
			var_summed += delta1 * delta2;
		}

		void result(uint64_t* count, double* mean, double* variance)
		{
			if (nullptr != count)
				*count = this->count;
			if (nullptr != mean)
				*mean = this->mean;
			if (nullptr != variance)
				*variance = this->var_summed / double(this->count);
		}
	};
}
#endif //!CIMGPROC_MATH_H