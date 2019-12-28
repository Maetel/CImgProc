//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#include "CLConvert2Gray.h"


namespace CIMGPROC::CL
{
	CLConvert2Gray::CLConvert2Gray()
	{
	}

	CLConvert2Gray::~CLConvert2Gray()
	{
	}

	void CLConvert2Gray::rebuildProgram()
	{
		//program = this->buildProgram("convert2gray.cl", "");
		const std::string s(
#include "cl_srcs/convert2gray.cl"
		)
			;
		
		program = this->buildProgramWithSrc(s, "");
	}

	bool CLConvert2Gray::convert2gray(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int length, Convert2Gray cvtType)
	{
		const char* caller =
			(RGB2GRAY  == cvtType) ? "rgb2gray" :
			(BGR2GRAY  == cvtType) ? "bgr2gray" :
			(RGBA2GRAY == cvtType) ? "rgba2gray" :
			(BGRA2GRAY == cvtType) ? "bgra2gray" : "not_defined";

		if (std::string("not_defined") == std::string(caller))
			return false;

		cl::Kernel kernel(program, caller);
		if (nullptr == kernel.get())
			return false;
		cl::EnqueueArgs args(queue, roundUpGlobalSize(length), localSize());

		cl::KernelFunctor<
			cl::Buffer,
			cl::Buffer,
			int
		> functor(kernel);

		cl_int err;
		functor(args,
			input,
			output,
			length,
			err
		);

		return false;
	}
}
