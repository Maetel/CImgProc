//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#include "CLDifference.h"
namespace CIMGPROC::CL
{
	CLDifference::CLDifference()
	{
	}

	CLDifference::~CLDifference()
	{
	}

	void CLDifference::rebuildProgram(std::string const& option)
	{
		program = this->buildProgram("array_difference.cl");
	}
	bool CLDifference::difference(
		cl::CommandQueue& queue,
		cl::Buffer const& input_l, cl::Buffer const& input_r,
		cl::Buffer& output,
		int length
	)
	{
		auto kernel = cl::Kernel(this->program, "array_difference");

		if (nullptr == kernel.get())
			return false;

		cl::EnqueueArgs args(
			queue,
			roundUpGlobalSize(length),
			localSize()
		);

		cl::KernelFunctor<
			cl::Buffer, cl::Buffer, //input_l, input_r
			cl::Buffer,				//input_r
			int						//length
		> functor(kernel);

		cl_int err;
		functor(args,
			input_l, input_r,
			output,
			length,
			err
		);
		return (CL_SUCCESS == err);
	}
}
