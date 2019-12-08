//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#include "CLDiffuse.h"
namespace CIMGPROC::CL
{
	CLDiffuse::CLDiffuse()
	{
	}

	CLDiffuse::~CLDiffuse()
	{
	}

	void CLDiffuse::rebuildProgram()
	{
		program = this->buildProgram("diffuse.cl");
	}
	bool CLDiffuse::diffuse(cl::CommandQueue& queue, cl::Buffer const& input1, cl::Buffer const& input2, cl::Buffer& output, float ratio_1to2, int length)
	{
		auto kernel = cl::Kernel(this->program, "diffuse_ratio");

		if (nullptr == kernel.get())
			return false;

		cl::EnqueueArgs args(
			queue,
			roundUpGlobalSize(length),
			localSize()
		);

		cl::KernelFunctor<
			cl::Buffer, cl::Buffer, //input_l, input_r
			cl::Buffer,				//output
			float,					//ratio_1to2
			int						//length
		> functor(kernel);

		cl_int err;
		functor(args,
			input1, input2,
			output,
			ratio_1to2,
			length,
			err
		);
		return (CL_SUCCESS == err);
	}
	bool CLDiffuse::diffuse(cl::CommandQueue& queue, cl::Buffer const& input1, cl::Buffer const& input2, cl::Buffer& output, cl::Buffer const& mask, int length)
	{
		auto kernel = cl::Kernel(this->program, "diffuse_mask");

		if (nullptr == kernel.get())
			return false;

		cl::EnqueueArgs args(
			queue,
			roundUpGlobalSize(length),
			localSize()
		);

		cl::KernelFunctor<
			cl::Buffer, cl::Buffer, //input_l, input_r
			cl::Buffer,				//output
			cl::Buffer,				//mask
			int						//length
		> functor(kernel);

		cl_int err;
		functor(args,
			input1, input2,
			output,
			mask,
			length,
			err
		);
		return (CL_SUCCESS == err);
	}
}
